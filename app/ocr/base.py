"""Common interface for OCR engines."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Word:
    """One recognised piece of text with its box in the engine's input-image pixels."""

    text: str
    confidence: float  # 0..1
    x: float
    y: float
    w: float
    h: float


@dataclass
class EngineOutput:
    """Raw engine output, before letter normalisation."""

    words: list[Word] = field(default_factory=list)
    text: str = ""  # full raw text in reading order
    confidence: float = 0.0  # 0..1
    confidence_estimated: bool = False  # engine gives no scores; confidence is a fixed guess


class OCREngine(abc.ABC):
    name: str = ""  # stable id used in the API: tesseract | apple | windows
    label: str = ""  # human readable
    # Which prepared image this engine reads best: "binary" (thresholded) or "gray".
    input_kind: str = "gray"

    def __init__(self) -> None:
        self.available = False
        self.reason = ""  # why the engine is unavailable
        self.detail = ""  # version / backend info for the UI

    @abc.abstractmethod
    def recognize(self, image: np.ndarray, mode: str) -> EngineOutput:
        """Recognise text in a uint8 grayscale image (dark text on light background).

        mode == "letter": the image is a tight crop around one glyph.
        mode == "text":   the image is a whole region that may hold words or lines.
        Must be thread-safe.
        """

    def status(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
            "detail": self.detail,
        }


class EngineUnavailable(RuntimeError):
    pass
