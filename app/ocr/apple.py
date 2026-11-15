"""Apple Vision OCR (secondary engine on macOS), via PyObjC.

Uses VNRecognizeTextRequest. Notes from testing on macOS:

* The "accurate" recognition level ignores isolated single characters entirely, so letter
  mode uses the "fast" level, which reads a tightly cropped glyph well.
* "accurate" may report Latin capitals as look-alike Cyrillic (e.g. "АВС"), even with
  en-US requested; app.letters maps those back to Latin.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from .base import EngineOutput, EngineUnavailable, OCREngine, Word

# The "fast" level reports only 1.0 or 0.5 for single letters, and is confidently wrong on
# a few shapes (it reads 'Q' as 'O'). Scale it so a cross-checked Tesseract reading
# (legacy + LSTM agreeing, ~0.9) wins a disagreement.
LETTER_CONFIDENCE_SCALE = 0.85


class AppleVisionEngine(OCREngine):
    name = "apple"
    label = "Apple Vision"
    input_kind = "gray"

    def __init__(self, languages: tuple[str, ...] = ("en-US",)) -> None:
        super().__init__()
        self.languages = list(languages)
        if sys.platform != "darwin":
            self.reason = "only available on macOS"
            return
        try:
            import objc
            import Vision
            from Foundation import NSData
        except ImportError as exc:
            self.reason = f"PyObjC Vision bindings not installed ({exc}); pip install pyobjc-framework-Vision"
            return
        if not hasattr(Vision, "VNRecognizeTextRequest"):
            self.reason = "this macOS has no Vision text recognition (needs macOS 10.15+)"
            return
        self._objc, self._vision, self._nsdata = objc, Vision, NSData
        self.available = True
        self.detail = "Vision VNRecognizeTextRequest (fast for letters, accurate for text)"

    def recognize(self, image: np.ndarray, mode: str) -> EngineOutput:
        if not self.available:
            raise EngineUnavailable(self.reason)
        Vision = self._vision
        ok, png = cv2.imencode(".png", image)
        if not ok:
            return EngineOutput()
        h, w = image.shape[:2]
        words: list[Word] = []
        with self._objc.autorelease_pool():
            data = self._nsdata.dataWithBytes_length_(png.tobytes(), len(png))
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
            request = Vision.VNRecognizeTextRequest.alloc().init()
            level = (
                Vision.VNRequestTextRecognitionLevelFast
                if mode == "letter"
                else Vision.VNRequestTextRecognitionLevelAccurate
            )
            request.setRecognitionLevel_(level)
            request.setUsesLanguageCorrection_(False)
            request.setRecognitionLanguages_(self.languages)
            if request.respondsToSelector_(b"setAutomaticallyDetectsLanguage:"):
                request.setAutomaticallyDetectsLanguage_(False)
            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise RuntimeError(f"Vision request failed: {error}")
            for obs in request.results() or []:
                candidates = obs.topCandidates_(1)
                if not candidates:
                    continue
                cand = candidates[0]
                bb = obs.boundingBox()  # normalised, origin at bottom-left
                words.append(
                    Word(
                        text=str(cand.string()),
                        confidence=float(cand.confidence()),
                        x=bb.origin.x * w,
                        y=(1.0 - bb.origin.y - bb.size.height) * h,
                        w=bb.size.width * w,
                        h=bb.size.height * h,
                    )
                )
        if not words:
            return EngineOutput()
        words.sort(key=lambda wd: (round(wd.y / max(wd.h, 1.0)), wd.x))  # reading order
        if mode == "letter":
            text = "".join(wd.text for wd in words)
            conf = min(wd.confidence for wd in words) * LETTER_CONFIDENCE_SCALE
            return EngineOutput(words=words, text=text, confidence=conf)
        conf = sum(wd.confidence for wd in words) / len(words)
        return EngineOutput(words=words, text=" ".join(wd.text for wd in words), confidence=conf)
