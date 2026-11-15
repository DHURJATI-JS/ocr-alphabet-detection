"""Run the primary (Tesseract) and secondary (Apple Vision / Windows OCR) engines.

Strategies:

* consensus - run both engines in parallel on every frame. If they agree the reading gets
              a confidence boost; if they disagree the more confident one wins. (default)
* fallback  - run Tesseract; only if it finds nothing or is below `fallback_below`, also
              run the secondary engine and keep the better reading.
* primary   - Tesseract only.
* secondary - Apple Vision / Windows OCR only.

The secondary engine is used only if it is available: the platform must have it, its Python
bindings must import, and it must pass a self-test at startup. Otherwise every strategy
quietly runs on Tesseract alone (see `effective_strategy`).

In letter mode an optional LLM arbiter (app/ocr/llm.py) settles frames the engines are
unsure about.
"""

from __future__ import annotations

import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, replace

import numpy as np

from ..config import DetectionOptions, Settings
from ..faces import FaceGuard
from ..letters import normalize_letter, normalize_text
from ..results import EngineReading, FrameResult
from ..preprocess import Prepared, decode_image, prepare_letter, prepare_text, thumbnail_png
from .apple import AppleVisionEngine
from .base import OCREngine
from .llm import LLMArbiter
from .tesseract import TesseractEngine
from .windows import WindowsOCREngine

AGREEMENT_BOOST = 0.15
SELF_TEST_TIMEOUT = 20.0  # seconds; first Vision / WinRT call may load models
# A frame never waits on an engine for longer than this. An engine that times out
# STRIKES times in a row is benched for BENCH_S seconds (Tesseract carries on alone), so
# a stalled OS service cannot freeze the live view.
ENGINE_TIMEOUT_S = 3.0
STRIKES = 2
BENCH_S = 60.0
LLM_CONFIRM_CONFIDENCE = 0.85  # LLM picked one of the engine readings
LLM_ONLY_CONFIDENCE = 0.70  # LLM read a letter none of the engines produced


class OCRManager:
    def __init__(self, primary: OCREngine, secondary: OCREngine | None, llm: LLMArbiter | None = None,
                 face_guard: FaceGuard | None = None) -> None:
        self.primary = primary
        self.secondary = secondary
        self.llm = llm
        self.face_guard = face_guard
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr-engine")
        self.engine_timeout = ENGINE_TIMEOUT_S
        self._strikes: dict[str, int] = {}
        self._benched_until: dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "OCRManager":
        primary = TesseractEngine(settings.tessdata_dir, settings.tesseract_lang, settings.tesseract_cmd)
        choice = settings.secondary
        if choice == "auto":
            choice = {"darwin": "apple", "win32": "windows"}.get(sys.platform, "none")
        secondary: OCREngine | None = None
        if choice == "apple":
            secondary = AppleVisionEngine()
        elif choice == "windows":
            secondary = WindowsOCREngine(settings.windows_language)
        for engine in (primary, secondary):
            if engine is not None:
                self_test(engine)
        llm = None
        if settings.llm_base_url:
            llm = LLMArbiter(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout)
        return cls(primary, secondary, llm, FaceGuard(settings.face_model))

    def _usable(self, engine: OCREngine | None) -> bool:
        return bool(engine and engine.available and time.monotonic() >= self._benched_until.get(engine.name, 0.0))

    @property
    def secondary_available(self) -> bool:
        return bool(self.secondary and self.secondary.available)

    def effective_strategy(self, requested: str) -> str:
        """The strategy that can actually run with the engines that are usable right now."""
        has_primary, has_secondary = self._usable(self.primary), self._usable(self.secondary)
        if requested in ("consensus", "fallback") and not (has_primary and has_secondary):
            return "primary" if has_primary else "secondary"
        if requested == "secondary" and not has_secondary:
            return "primary"
        if requested == "primary" and not has_primary and has_secondary:
            return "secondary"
        return requested

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.llm:
            self.llm.close()

    def status(self) -> dict:
        return {
            "primary": self.primary.status(),
            "secondary": self.secondary.status() if self.secondary else None,
            "llm": self.llm.status() if self.llm else None,
        }

    # ----------------------------------------------------------------- process
    def process(self, image_bytes: bytes, opts: DetectionOptions, wait_for_llm: bool = False) -> FrameResult:
        """OCR one image. Raises ValueError for undecodable input. Thread-safe."""
        if opts.mode not in ("letter", "text"):
            raise ValueError(f"OCR handles 'letter' and 'text' modes, not {opts.mode!r}")
        t0 = time.perf_counter()
        bgr = decode_image(image_bytes)
        # Faces are not text: mask them out (and skip shapes on skin) unless switched off.
        faces = self.face_guard.boxes(bgr) if (self.face_guard and opts.ignore_faces) else []
        prepare = prepare_letter if opts.mode == "letter" else prepare_text
        prepared = prepare(bgr, faces, ignore_skin=opts.ignore_faces)
        t_prep = (time.perf_counter() - t0) * 1000

        strategy = self.effective_strategy(opts.strategy)
        readings = self._run_strategy(prepared, opts, strategy) if prepared.found else []
        best, agreement = _choose(readings)

        if self._needs_arbiter(prepared, opts, best, agreement):
            best, agreement = self._arbitrate(prepared, opts, readings, best, agreement, wait_for_llm)

        result = FrameResult(
            mode=opts.mode,
            strategy=strategy,
            text=best.text if best else "",
            confidence=round(best.confidence, 3) if best else 0.0,
            engine=best.engine if best else None,
            agreement=agreement,
            found=prepared.found,
            glyph=prepared.glyph.as_dict() if prepared.glyph else None,
            readings=readings,
            timings={"preprocess_ms": round(t_prep, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
        )
        if opts.debug_image and prepared.found:
            # Side by side: Tesseract's thresholded input | the secondary engine's gray input.
            sep = np.full((prepared.gray.shape[0], 4), 128, np.uint8)
            png = thumbnail_png(np.hstack([prepared.binary, sep, prepared.gray]), max_side=320)
            result.debug_image = "data:image/png;base64," + base64.b64encode(png).decode()
        return result

    def _run_strategy(self, prepared: Prepared, opts: DetectionOptions, strategy: str) -> list[EngineReading]:
        # effective_strategy() only returns consensus/fallback when both engines are available.
        primary, secondary = self.primary, self.secondary
        if strategy == "consensus":
            futures = [(e, self._pool.submit(self._read, e, prepared, opts)) for e in (primary, secondary)]
            return [self._collect(e, f) for e, f in futures]
        if strategy == "fallback":
            first = self._read_bounded(primary, prepared, opts)
            if first.text and first.confidence >= opts.fallback_below:
                return [first]
            return [first, self._read_bounded(secondary, prepared, opts)]
        engine = primary if strategy == "primary" else secondary
        if not self._usable(engine):
            return []
        return [self._read_bounded(engine, prepared, opts)]

    def _read_bounded(self, engine: OCREngine, prepared: Prepared, opts: DetectionOptions) -> EngineReading:
        return self._collect(engine, self._pool.submit(self._read, engine, prepared, opts))

    def _collect(self, engine: OCREngine, future) -> EngineReading:
        """Wait for an engine at most engine_timeout; bench engines that keep stalling."""
        try:
            reading = future.result(timeout=self.engine_timeout)
        except FutureTimeout:
            strikes = self._strikes.get(engine.name, 0) + 1
            self._strikes[engine.name] = strikes
            note = None
            if strikes >= STRIKES:
                self._benched_until[engine.name] = time.monotonic() + BENCH_S
                self._strikes[engine.name] = 0
                note = f"benched for {BENCH_S:.0f}s"
            return EngineReading(engine.name, engine.label, "", "", 0.0, False, round(self.engine_timeout * 1000, 1),
                                 error=f"no answer within {self.engine_timeout:g}s", note=note)
        self._strikes[engine.name] = 0
        return reading

    def _read(self, engine: OCREngine, prepared: Prepared, opts: DetectionOptions) -> EngineReading:
        image = prepared.binary if engine.input_kind == "binary" else prepared.gray
        t0 = time.perf_counter()
        try:
            out = engine.recognize(image, opts.mode)
        except Exception as exc:  # noqa: BLE001 - one engine failing must not kill the frame
            return EngineReading(engine.name, engine.label, "", "", 0.0, False,
                                 round((time.perf_counter() - t0) * 1000, 1), error=f"{type(exc).__name__}: {exc}")
        latency = round((time.perf_counter() - t0) * 1000, 1)
        normalize = normalize_letter if opts.mode == "letter" else normalize_text
        text = normalize(out.text, opts.case)
        boxes = []
        for wd in out.words:
            box = prepared.to_source(wd.x, wd.y, wd.w, wd.h).as_dict()
            box["text"] = normalize_text(wd.text, opts.case)
            box["confidence"] = round(wd.confidence, 3)
            boxes.append(box)
        return EngineReading(
            engine=engine.name,
            label=engine.label,
            text=text,
            raw=out.text,
            confidence=round(out.confidence if text else 0.0, 3),
            confidence_estimated=out.confidence_estimated,
            latency_ms=latency,
            boxes=boxes,
        )

    # --------------------------------------------------------------- arbiter
    def _needs_arbiter(self, prepared: Prepared, opts: DetectionOptions,
                       best: EngineReading | None, agreement: bool | None) -> bool:
        if not (self.llm and opts.llm_arbiter and opts.mode == "letter" and prepared.found):
            return False
        return best is None or best.confidence < opts.min_confidence or agreement is False

    def _arbitrate(self, prepared: Prepared, opts: DetectionOptions, readings: list[EngineReading],
                   best: EngineReading | None, agreement: bool | None, wait: bool):
        assert self.llm is not None
        t0 = time.perf_counter()
        verdict = self.llm.lookup(prepared.binary, prepared.gray, wait=wait)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        if verdict is None:
            note = "paused" if self.llm.status()["reason"] else "asking…"
            readings.append(EngineReading(self.llm.name, self.llm.label, "", "", 0.0, True, latency,
                                          error=self.llm.last_error or None, note=note))
            return best, agreement

        letter = normalize_letter(verdict.letter, opts.case)
        llm_reading = EngineReading(
            engine=self.llm.name, label=self.llm.label, text=letter, raw=verdict.raw,
            confidence=LLM_ONLY_CONFIDENCE if letter else 0.0, confidence_estimated=True,
            latency_ms=verdict.latency_ms if not verdict.cached else latency,
            note="cached" if verdict.cached else None,
        )
        readings.append(llm_reading)
        if not letter:
            return best, agreement
        match = next((r for r in readings[:-1] if r.text and r.text.casefold() == letter.casefold()), None)
        if match:
            confirmed = replace(match, confidence=max(match.confidence, LLM_CONFIRM_CONFIDENCE),
                                engine=f"{match.engine}+llm", label=f"{match.label} + LLM")
            return confirmed, True
        return llm_reading, False


def self_test(engine: OCREngine) -> None:
    """Run the engine once on a blank image; if that fails, mark it unavailable.

    An engine can import fine and still not work (missing OS feature, broken language pack,
    permissions). Better to find out once at startup than to error on every frame. Runs in a
    worker thread: Windows OCR starts its own asyncio loop, which must not happen inside
    uvicorn's running loop.
    """
    if not engine.available:
        return
    probe = np.full((96, 96), 255, np.uint8)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"selftest-{engine.name}")
    try:
        pool.submit(engine.recognize, probe, "letter").result(timeout=SELF_TEST_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        engine.available = False
        engine.reason = f"failed its startup self-test: {type(exc).__name__}: {exc}"[:300]
    finally:
        pool.shutdown(wait=False)


def _choose(readings: list[EngineReading]) -> tuple[EngineReading | None, bool | None]:
    usable = [r for r in readings if r.text]
    if not usable:
        return None, None
    if len(usable) < 2:
        return usable[0], None  # only one source read anything: nothing to compare
    a, b = usable[0], usable[1]
    if a.text.casefold() == b.text.casefold():
        winner = max((a, b), key=lambda r: r.confidence)
        boosted = replace(winner, confidence=round(min(1.0, winner.confidence + AGREEMENT_BOOST), 3),
                          engine=f"{a.engine}+{b.engine}", label=f"{a.label} + {b.label}")
        return boosted, True
    return max(usable, key=lambda r: r.confidence), False
