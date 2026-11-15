"""Tesseract engine (primary).

Two backends, picked automatically:

* ``tesserocr`` (preferred): Tesseract compiled into the Python wheel, no system install
  needed. Only the language model (``tessdata/eng.traineddata``) has to be present.
* ``pytesseract``: drives a system ``tesseract`` executable (e.g. Homebrew or the
  UB Mannheim installer on Windows). Slower, since it spawns a process per call.

Single letters (benchmarked on synthetic webcam frames, see scripts/benchmark.py):

* The LSTM recogniser is weak on an isolated glyph in single-char mode (~85%): it often
  returns nothing or stutters ("WWW" for "W").
* The legacy character classifier reads isolated glyphs well (~97%, <1 ms). It needs the
  standard ``tessdata`` model (tessdata_fast/best ship LSTM only).
* The LSTM is very precise when the glyph is tiled into a short line ("A A A", PSM 7)
  and the characters are majority-voted.

Letter mode runs both and cross-checks them; text mode uses the LSTM on the whole block.
"""

from __future__ import annotations

import shutil
import string
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from .base import EngineOutput, EngineUnavailable, OCREngine, Word

WHITELIST = string.ascii_uppercase + string.ascii_lowercase
OEM_LEGACY, OEM_LSTM = 0, 1
PSM_SINGLE_BLOCK, PSM_SINGLE_LINE, PSM_SINGLE_CHAR = 6, 7, 10
TILES = 3
DISAGREE_PENALTY = 0.5  # confidence multiplier when legacy and LSTM read different letters
SINGLE_SOURCE_PENALTY = 0.85  # only one of the two produced a letter

# recognition passes: name -> (oem, psm)
PASSES = {
    "legacy_char": (OEM_LEGACY, PSM_SINGLE_CHAR),
    "lstm_line": (OEM_LSTM, PSM_SINGLE_LINE),
    "lstm_block": (OEM_LSTM, PSM_SINGLE_BLOCK),
}


class TesseractEngine(OCREngine):
    name = "tesseract"
    label = "Tesseract"
    input_kind = "binary"

    def __init__(self, tessdata_dir: Path, lang: str = "eng", tesseract_cmd: str | None = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._backend = None
        self.lang = lang
        self.tessdata_dir = Path(tessdata_dir)
        self.has_legacy = False
        errors = []
        for setup in (self._setup_tesserocr, self._setup_pytesseract):
            try:
                setup(tesseract_cmd)
                self.available = True
                if not self.has_legacy:
                    self.detail += " - LSTM only (run scripts/download_models.py for better single letters)"
                return
            except EngineUnavailable as exc:
                errors.append(str(exc))
        self.reason = "; ".join(errors)

    # ---------------------------------------------------------------- backends
    def _setup_tesserocr(self, _cmd: str | None) -> None:
        try:
            import tesserocr
            from tesserocr import PyTessBaseAPI
        except ImportError as exc:
            raise EngineUnavailable(f"tesserocr not installed ({exc})") from exc
        model = self.tessdata_dir / f"{self.lang}.traineddata"
        if not model.is_file():
            raise EngineUnavailable(f"missing {model} (run: python scripts/download_models.py)")
        path = str(self.tessdata_dir) + "/"
        self._apis = {}
        for name, (oem, psm) in PASSES.items():
            try:
                api = PyTessBaseAPI(path=path, lang=self.lang, psm=psm, oem=oem)
            except RuntimeError:
                if oem == OEM_LEGACY:
                    continue  # LSTM-only model (tessdata_fast / tessdata_best)
                raise EngineUnavailable(f"tesseract failed to load {model}")
            api.SetVariable("tessedit_char_whitelist", WHITELIST)
            api.SetVariable("load_system_dawg", "0")  # alphabet cards are not dictionary words
            api.SetVariable("load_freq_dawg", "0")
            # The legacy classifier adapts to every image it sees, so readings would drift
            # with the history of the live stream (and one misread would train the next).
            api.SetVariable("classify_enable_learning", "0")
            self._apis[name] = api
        self.has_legacy = "legacy_char" in self._apis
        self._backend = "tesserocr"
        version = tesserocr.tesseract_version().splitlines()[0]
        self.detail = f"{version} (embedded via tesserocr)"

    def _setup_pytesseract(self, cmd: str | None) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise EngineUnavailable(f"pytesseract not installed ({exc})") from exc
        binary = cmd or shutil.which("tesseract")
        if not binary:
            raise EngineUnavailable("no tesseract executable on PATH (set TESSERACT_CMD)")
        pytesseract.pytesseract.tesseract_cmd = binary
        try:
            version = pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            raise EngineUnavailable(f"tesseract at {binary} failed: {exc}") from exc
        self._pytesseract = pytesseract
        self._backend = "pytesseract"
        try:  # probe whether the installed model includes the legacy engine
            self._run_pytesseract("legacy_char", np.full((40, 40), 255, np.uint8))
            self.has_legacy = True
        except pytesseract.TesseractError:
            self.has_legacy = False
        self.detail = f"tesseract {version} (CLI: {binary})"

    # ---------------------------------------------------------------- passes
    def _run(self, pass_name: str, image: np.ndarray) -> list[Word]:
        if self._backend == "tesserocr":
            return self._run_tesserocr(pass_name, image)
        return self._run_pytesseract(pass_name, image)

    def _run_tesserocr(self, pass_name: str, image: np.ndarray) -> list[Word]:
        from tesserocr import RIL, iterate_level

        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        level = RIL.WORD if pass_name == "lstm_block" else RIL.SYMBOL
        words: list[Word] = []
        with self._lock:  # PyTessBaseAPI instances are not thread-safe
            api = self._apis[pass_name]
            api.SetImageBytes(image.tobytes(), w, h, 1, w)
            api.Recognize()
            iterator = api.GetIterator()
            if iterator is not None:
                for item in iterate_level(iterator, level):
                    try:
                        text = item.GetUTF8Text(level)
                    except RuntimeError:  # tesserocr raises for an empty result element
                        continue
                    box = item.BoundingBox(level)
                    if not text or not text.strip() or box is None:
                        continue
                    x0, y0, x1, y1 = box
                    words.append(Word(text.strip(), item.Confidence(level) / 100.0, x0, y0, x1 - x0, y1 - y0))
            api.Clear()
        return words

    def _run_pytesseract(self, pass_name: str, image: np.ndarray) -> list[Word]:
        pt = self._pytesseract
        oem, psm = PASSES[pass_name]
        config = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={WHITELIST} -c classify_enable_learning=0"
        if (self.tessdata_dir / f"{self.lang}.traineddata").is_file():
            config += f' --tessdata-dir "{self.tessdata_dir}"'
        data = pt.image_to_data(image, lang=self.lang, config=config, output_type=pt.Output.DICT)
        words = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if text and text.strip() and conf >= 0:
                words.append(Word(text.strip(), conf / 100.0, data["left"][i], data["top"][i],
                                  data["width"][i], data["height"][i]))
        return words

    # ---------------------------------------------------------------- recognise
    def recognize(self, image: np.ndarray, mode: str) -> EngineOutput:
        if not self.available:
            raise EngineUnavailable(self.reason)
        if mode == "letter":
            return self._recognize_letter(image)
        words = self._run("lstm_block", image)
        if not words:
            return EngineOutput()
        weights = [max(len(wd.text), 1) for wd in words]
        conf = sum(wd.confidence * n for wd, n in zip(words, weights)) / sum(weights)
        return EngineOutput(words=words, text=" ".join(wd.text for wd in words), confidence=conf)

    def _recognize_letter(self, image: np.ndarray) -> EngineOutput:
        h, w = image.shape[:2]
        legacy = self._run("legacy_char", image) if self.has_legacy else []
        lstm = _vote(self._run("lstm_line", _tile(image, TILES)))

        leg_text = "".join(wd.text for wd in legacy)
        leg_conf = min((wd.confidence for wd in legacy), default=0.0)
        if len(leg_text) != 1:
            leg_text = ""
        box = legacy[0] if len(legacy) == 1 else Word("", 0.0, 0, 0, w, h)

        def out(text: str, conf: float) -> EngineOutput:
            return EngineOutput(words=[Word(text, conf, box.x, box.y, box.w, box.h)], text=text, confidence=conf)

        if leg_text and lstm:
            lstm_text, lstm_conf = lstm
            if leg_text.casefold() == lstm_text.casefold():
                return out(lstm_text if lstm_conf >= leg_conf else leg_text, max(leg_conf, lstm_conf))
            text, conf = (leg_text, leg_conf) if leg_conf >= lstm_conf else (lstm_text, lstm_conf)
            return out(text, conf * DISAGREE_PENALTY)
        if leg_text:
            return out(leg_text, leg_conf * SINGLE_SOURCE_PENALTY)
        if lstm:
            return out(lstm[0], lstm[1] * SINGLE_SOURCE_PENALTY)
        return EngineOutput()


def _tile(image: np.ndarray, n: int) -> np.ndarray:
    """Repeat a glyph crop n times side by side, so the LSTM sees a short line of text."""
    h, w = image.shape[:2]
    gap = np.full((h, max(4, w // 6)), 255, np.uint8)
    parts: list[np.ndarray] = []
    for _ in range(n):
        parts += [image, gap]
    return np.hstack(parts[:-1])


def _vote(words: list[Word]) -> tuple[str, float] | None:
    """Majority letter over the tiled reading: (letter, confidence) or None if no majority."""
    chars = [(ch, wd.confidence) for wd in words for ch in wd.text if ch.isalpha()]
    if not chars:
        return None
    counts = Counter(ch.casefold() for ch, _ in chars)
    letter, n = counts.most_common(1)[0]
    if n < 2:
        return None
    matching = [(ch, c) for ch, c in chars if ch.casefold() == letter]
    form = Counter(ch for ch, _ in matching).most_common(1)[0][0]  # most frequent case variant
    return form, sum(c for _, c in matching) / len(matching)
