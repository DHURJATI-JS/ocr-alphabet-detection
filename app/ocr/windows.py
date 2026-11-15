"""Windows OCR (secondary engine on Windows 10/11), via Windows.Media.Ocr.

Uses the PyWinRT projection (``winrt-Windows.Media.Ocr`` and friends), falling back to the
older all-in-one ``winsdk`` package. Requires an OCR language pack, which ships with the
matching Windows display language (Settings > Time & language > Language).

Windows OCR reports no confidence scores, so a fixed estimate is used and flagged as such.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import cv2
import numpy as np

from .base import EngineOutput, EngineUnavailable, OCREngine, Word

ESTIMATED_CONFIDENCE = 0.80


def _import_winrt():
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter

        return "winrt", Language, BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap, OcrEngine, DataWriter
    except ImportError:
        from winsdk.windows.globalization import Language
        from winsdk.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage.streams import DataWriter

        return "winsdk", Language, BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap, OcrEngine, DataWriter


class WindowsOCREngine(OCREngine):
    name = "windows"
    label = "Windows OCR"
    input_kind = "gray"

    def __init__(self, language: str = "en-US") -> None:
        super().__init__()
        self._lock = threading.Lock()
        if sys.platform != "win32":
            self.reason = "only available on Windows 10/11"
            return
        try:
            (pkg, Language, self._AlphaMode, self._PixelFormat, self._SoftwareBitmap, OcrEngine,
             self._DataWriter) = _import_winrt()
        except ImportError as exc:
            self.reason = f"WinRT OCR bindings not installed ({exc}); pip install winrt-Windows.Media.Ocr"
            return
        lang = Language(language)
        engine = OcrEngine.try_create_from_language(lang) if OcrEngine.is_language_supported(lang) else None
        if engine is None:
            engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            self.reason = f"no Windows OCR language pack installed for {language}"
            return
        self._engine = engine
        self._max_side = int(OcrEngine.max_image_dimension)
        self.available = True
        self.detail = f"Windows.Media.Ocr [{engine.recognizer_language.language_tag}] via {pkg}"

    def _bitmap(self, gray: np.ndarray):
        h, w = gray.shape[:2]
        if max(h, w) > self._max_side:
            f = self._max_side / float(max(h, w))
            gray = cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
            h, w = gray.shape[:2]
        bgra = np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
        writer = self._DataWriter()
        try:
            writer.write_bytes(bgra.tobytes())
        except TypeError:  # older projections want a list of ints
            writer.write_bytes(list(bgra.tobytes()))
        return self._SoftwareBitmap.create_copy_from_buffer(
            writer.detach_buffer(), self._PixelFormat.BGRA8, w, h, self._AlphaMode.PREMULTIPLIED
        ), (gray.shape[1], gray.shape[0])

    def recognize(self, image: np.ndarray, mode: str) -> EngineOutput:
        if not self.available:
            raise EngineUnavailable(self.reason)
        orig_h, orig_w = image.shape[:2]
        bitmap, (bw, bh) = self._bitmap(image)

        async def run():
            return await self._engine.recognize_async(bitmap)

        with self._lock:
            result = asyncio.run(run())  # called from a worker thread, which has no running loop
        sx, sy = orig_w / float(bw), orig_h / float(bh)
        words: list[Word] = []
        for line in result.lines:
            for word in line.words:
                r = word.bounding_rect
                words.append(Word(word.text, ESTIMATED_CONFIDENCE, r.x * sx, r.y * sy, r.width * sx, r.height * sy))
        if not words:
            return EngineOutput()
        if mode == "letter":
            best = max(words, key=lambda wd: wd.w * wd.h)
            return EngineOutput(words=[best], text=best.text, confidence=ESTIMATED_CONFIDENCE, confidence_estimated=True)
        return EngineOutput(
            words=words, text=" ".join(wd.text for wd in words), confidence=ESTIMATED_CONFIDENCE,
            confidence_estimated=True,
        )
