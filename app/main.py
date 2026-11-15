"""FastAPI + Socket.IO server.

    python -m app            (picks a free port; see app/__main__.py)
    uvicorn app.main:app --reload

Modes (per browser session):
  sign    hand-sign alphabet: MediaPipe hands -> built-in ASL rules and/or taught signs
  letter  printed letter: OCR (Tesseract + Apple Vision / Windows OCR)
  text    printed words: OCR

HTTP
  GET  /              live webcam page (Jinja template)
  GET  /api/engines   engine availability
  GET  /api/signs     how many examples of each letter have been taught
  POST /api/ocr       recognise one image: raw JPEG/PNG body, options as query params
                      e.g. curl --data-binary @a.png "localhost:8000/api/ocr?mode=sign"

Socket.IO (same origin, path /socket.io)
  client -> "frame"        {id, image: <binary JPEG>}            -> ack with the result
  client -> "settings"     {partial DetectionOptions}            -> ack with effective options
  client -> "reset"                                              -> clears typing state
  client -> "teach_start"  {letter, samples}                     -> record the next frames as that sign
  client -> "teach_stop"                                         -> stop recording
  client -> "signs_delete" {letter} / "signs_clear"              -> ack with remaining counts
  server -> "status"       engine availability (on connect)
"""

from __future__ import annotations

import asyncio
import logging
import string
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import socketio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from .config import DetectionOptions, Settings
from .hands.gate import HandGate
from .hands.library import SignLibrary
from .hands.motion import MotionTracker
from .hands.recognizer import SignRecognizer
from .hands.tracker import HandTracker
from .ocr.manager import OCRManager
from .stabilizer import Stabilizer

log = logging.getLogger("uvicorn.error")  # uvicorn's configured logger, so startup info is visible
HERE = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 8 * 1024 * 1024
REPLACE_WINDOW_S = 2.5  # a J/Z replaces the static letter typed this recently at its start
LETTERS = string.ascii_uppercase

settings = Settings.from_env()
manager = OCRManager.from_settings(settings)
signs = SignRecognizer(HandTracker(settings.hand_model, min_presence=settings.hand_presence),
                       SignLibrary(settings.signs_path),
                       SignRecognizer.load_alphabets(settings.hand_model.parent))
executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="frame")


def default_options() -> DetectionOptions:
    """Defaults, reduced to what is available on this machine: OCR engine strategy
    (Tesseract + Apple Vision / Windows OCR, or Tesseract alone), and OCR mode if hand
    tracking cannot run."""
    base = DetectionOptions()
    changes = {"strategy": manager.effective_strategy(base.strategy)}
    if not signs.available:
        changes["mode"] = "letter"
    if base.alphabet not in signs.alphabets:  # no trained model for the default alphabet
        changes["alphabet"] = "asl"
    return base.merged(changes)


@dataclass
class Recording:
    letter: str
    total: int
    done: int = 0
    hands: int | None = None  # locked to the hand count of the first recorded frame
    skipped: int = 0  # frames with a different number of hands in view


@dataclass
class Session:
    options: DetectionOptions = field(default_factory=default_options)
    stabilizer: Stabilizer = field(default_factory=Stabilizer)
    motion: MotionTracker = field(default_factory=MotionTracker)
    gate: HandGate = field(default_factory=HandGate)
    recording: Recording | None = None
    busy: bool = False

    def __post_init__(self) -> None:
        o = self.options
        self.stabilizer.configure(o.hold_frames, o.release_frames, o.typing_confidence)

    def apply(self, options: DetectionOptions) -> None:
        if options.mode != self.options.mode:
            self.stabilizer.reset()
            self.motion.reset()
            self.gate.reset()
            self.recording = None
        self.options = options
        self.stabilizer.configure(options.hold_frames, options.release_frames, options.typing_confidence)


sessions: dict[str, Session] = {}


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())


def _engines_status() -> dict:
    return {**manager.status(), "hands": signs.status()}


def _taught() -> dict:
    """Taught letters: example counts and whether each is a one- or two-handed sign."""
    return {"counts": signs.library.counts(), "hands": signs.library.hands_by_letter()}


# --------------------------------------------------------------------- HTTP
@asynccontextmanager
async def lifespan(_app: FastAPI):
    for role, info in _engines_status().items():
        if info:
            state = info["detail"] if info["available"] else f"UNAVAILABLE - {info['reason']}"
            log.info("%-9s %-15s %s", role, info["label"], state)
    for name, model in signs.alphabets.items():
        log.info("signs     %s model: letters %s (trained on %s)", name.upper(), "".join(model.letters),
                 ", ".join(model.meta.get("sources", [])) or "?")
    for name in ("isl", "asl"):
        if name not in signs.alphabets:
            log.warning("signs     no %s model (run: python scripts/train_signs.py --alphabet %s)", name.upper(), name)
    taught = signs.library.counts()
    log.info("signs     taught: %s", ", ".join(f"{k}x{v}" for k, v in taught.items()) or "none yet")
    if signs.library.load_error:
        log.warning("signs     %s", signs.library.load_error)
    yield
    executor.shutdown(wait=False, cancel_futures=True)
    manager.close()


api = FastAPI(title="Alphabet OCR", lifespan=lifespan)
api.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


@api.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"engines": _engines_status(), "defaults": default_options().model_dump(), "letters": LETTERS},
    )


@api.get("/api/engines")
async def engines():
    return _engines_status()


@api.get("/api/signs")
async def taught_signs():
    return {"counts": signs.library.counts(), "error": signs.library.load_error or None}


@api.get("/health")
async def health():
    return {"ok": True, "tesseract": manager.primary.available, "hands": signs.available}


def _recognize(image: bytes, opts: DetectionOptions, session: Session | None = None, wait_for_llm=False):
    """Run the recogniser for the mode. Returns (FrameResult, sign features or None).
    With a live session, hands must persist across frames (HandGate) to count."""
    if opts.mode == "sign":
        if not signs.available:
            raise ValueError(f"hand tracking unavailable: {signs.tracker.reason}")
        if session is None:
            return signs.process(image, opts)
        return signs.process(image, opts, session.motion, session.gate)
    return manager.process(image, opts, wait_for_llm=wait_for_llm), None


@api.post("/api/ocr")
async def ocr_image(request: Request):
    body = await request.body()
    if len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large")
    try:
        opts = default_options().merged(dict(request.query_params))
    except ValidationError as exc:
        raise HTTPException(422, _validation_message(exc)) from exc
    loop = asyncio.get_running_loop()
    try:
        result, _ = await loop.run_in_executor(executor, lambda: _recognize(body, opts, wait_for_llm=True))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result.as_dict()


# ---------------------------------------------------------------- Socket.IO
sio = socketio.AsyncServer(async_mode="asgi", max_http_buffer_size=MAX_IMAGE_BYTES)


@sio.event
async def connect(sid, environ, auth=None):
    sessions[sid] = Session()
    await sio.emit("status", {"engines": _engines_status(), "options": sessions[sid].options.model_dump(),
                              "taught": _taught()}, to=sid)


@sio.event
async def disconnect(sid, reason=None):
    session = sessions.pop(sid, None)
    if session and session.recording:
        await asyncio.get_running_loop().run_in_executor(executor, signs.library.save)


@sio.on("settings")
async def on_settings(sid, data):
    session = sessions.get(sid)
    if session is None:
        return {"error": "unknown session"}
    try:
        session.apply(session.options.merged(data if isinstance(data, dict) else {}))
    except ValidationError as exc:
        return {"error": _validation_message(exc), "options": session.options.model_dump()}
    return {"options": session.options.model_dump()}


@sio.on("reset")
async def on_reset(sid, _data=None):
    session = sessions.get(sid)
    if session:
        session.stabilizer.reset()
        session.motion.reset()
    return {"ok": True}


# ------------------------------------------------------------- teaching signs
@sio.on("teach_start")
async def on_teach_start(sid, data):
    session = sessions.get(sid)
    if session is None:
        return {"error": "unknown session"}
    letter = str((data or {}).get("letter", "")).upper()
    if letter not in LETTERS:
        return {"error": "pick a letter A-Z"}
    try:
        total = max(5, min(200, int((data or {}).get("samples", 30))))
    except (TypeError, ValueError):
        return {"error": "samples must be a number"}
    if not signs.available:
        return {"error": f"hand tracking unavailable: {signs.tracker.reason}"}
    session.recording = Recording(letter, total)
    return {"recording": {"letter": letter, "total": total, "done": 0}}


@sio.on("teach_stop")
async def on_teach_stop(sid, _data=None):
    session = sessions.get(sid)
    if session and session.recording:
        session.recording = None
        await asyncio.get_running_loop().run_in_executor(executor, signs.library.save)
    return _taught()


async def _library_changed(action) -> dict:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, action)
    await loop.run_in_executor(executor, signs.library.save)
    taught = _taught()
    await sio.emit("taught", taught)  # every open page
    return taught


@sio.on("signs_delete")
async def on_signs_delete(sid, data):
    letter = str((data or {}).get("letter", "")).upper()
    if letter not in LETTERS:
        return {"error": "pick a letter A-Z"}
    return await _library_changed(lambda: signs.library.remove(letter))


@sio.on("signs_clear")
async def on_signs_clear(sid, _data=None):
    return await _library_changed(signs.library.clear)


# ------------------------------------------------------------------- frames
@sio.on("frame")
async def on_frame(sid, data):
    session = sessions.get(sid)
    frame_id = data.get("id") if isinstance(data, dict) else None
    if session is None:
        return {"id": frame_id, "error": "unknown session"}
    image = data.get("image") if isinstance(data, dict) else None
    if not isinstance(image, (bytes, bytearray)) or not image:
        return {"id": frame_id, "error": "frame must carry a binary 'image' (JPEG/PNG bytes)"}
    if session.busy:  # the client waits for each ack, so this only happens with misbehaving clients
        return {"id": frame_id, "dropped": True}

    session.busy = True
    options = session.options
    try:
        loop = asyncio.get_running_loop()
        result, features = await loop.run_in_executor(executor, _recognize, bytes(image), options, session)
    except ValueError as exc:
        return {"id": frame_id, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - keep the stream alive, report the failure
        log.exception("frame processing failed")
        return {"id": frame_id, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        session.busy = False

    if options is not session.options:  # settings changed while this frame was in flight
        return {"id": frame_id, "dropped": True}
    payload = result.as_dict()
    payload["id"] = frame_id

    rec = session.recording
    if rec is not None:  # teaching: store the frame instead of typing
        visible = features[0] if features is not None else 0
        if features is not None:
            if rec.hands is None:  # the first frame decides: one- or two-handed sign
                rec.hands = visible
            if visible == rec.hands:
                signs.library.add(rec.letter, *features)
                rec.done += 1
            else:
                rec.skipped += 1
        payload["recording"] = {"letter": rec.letter, "total": rec.total, "done": rec.done,
                                "hands": rec.hands, "visible": visible, "skipped": rec.skipped}
        if rec.done >= rec.total:
            session.recording = None
            await loop.run_in_executor(executor, signs.library.save)
            payload["recording"]["finished"] = True
            payload["taught"] = _taught()
            session.stabilizer.reset()
        payload.update(candidate="", progress=0.0, commit=None)
        return payload

    state = session.stabilizer.update(result.text, result.confidence, present=result.found)
    payload.update(candidate=state.candidate, progress=round(state.progress, 3), commit=state.commit)
    if result.motion:  # J / Z drawn in the air
        stab = session.stabilizer
        recent = time.monotonic() - stab.last_commit_at < REPLACE_WINDOW_S
        payload["replace"] = 1 if (recent and stab.last_commit and stab.last_commit == result.motion_base) else 0
        payload.update(commit=result.motion, candidate=result.motion, progress=1.0)
        stab.force(result.motion)
    if manager.llm:
        payload["llm_status"] = manager.llm.status()
    return payload


# uvicorn entry point: Socket.IO handles /socket.io, everything else goes to FastAPI.
app = socketio.ASGIApp(sio, other_asgi_app=api)
