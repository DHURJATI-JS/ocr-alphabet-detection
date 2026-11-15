"""Keep faces out of printed-letter OCR.

A mouth, nostrils or eyes are dark blobs on a lighter background: exactly what the glyph
picker looks for. Measured on 84 face-only frames, OCR "read" a letter in 79 and was
confident enough to type it in 23. Two guards:

* FaceGuard - MediaPipe's face detector (BlazeFace); detected faces are masked out.
* skin_like  - backup for close-ups where too little of the face is visible for the
  detector: a printed letter sits on paper, so a shape whose surroundings are skin-toned
  is not one.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")

# Face box expansion: BlazeFace boxes run roughly brows-to-chin; include hair and chin.
EXPAND_X, EXPAND_TOP, EXPAND_BOTTOM = 0.15, 0.35, 0.25
# BlazeFace also "finds faces" in round printed letters (G, S, Q, B ... 291 times in 936
# letter images, scores up to 0.72, so the score cannot tell). Those boxes contain no skin;
# real faces measured >= 17% skin-coloured pixels.
MIN_FACE_SKIN = 0.12


class FaceGuard:
    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        self.available = False
        self.reason = ""
        self._lock = threading.Lock()
        model_path = Path(model_path)
        if not model_path.is_file():
            self.reason = f"missing {model_path} (run: python scripts/download_models.py)"
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            self._detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE, min_detection_confidence=min_confidence))
        except Exception as exc:  # noqa: BLE001 - OCR works without the guard
            self.reason = f"face detector unavailable: {exc}"
            return
        self._mp = mp
        self.available = True

    def boxes(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Face regions (x0, y0, x1, y1) in pixels, generously expanded."""
        if not self.available:
            return []
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self._lock:
            result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        out = []
        skin = None
        for det in result.detections:
            b = det.bounding_box
            if skin is None:
                skin = skin_mask(bgr)
            region = skin[max(0, b.origin_y):max(0, b.origin_y + b.height), max(0, b.origin_x):max(0, b.origin_x + b.width)]
            if region.size == 0 or float((region > 0).mean()) < MIN_FACE_SKIN:
                continue  # not a face: no skin in it (e.g. a printed 'G')
            x0 = b.origin_x - EXPAND_X * b.width
            x1 = b.origin_x + (1 + EXPAND_X) * b.width
            y0 = b.origin_y - EXPAND_TOP * b.height
            y1 = b.origin_y + (1 + EXPAND_BOTTOM) * b.height
            out.append((max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))))
        return out


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels whose colour is in the usual skin range (YCrCb), 255 = skin."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, np.array([40, 138, 77], np.uint8), np.array([250, 175, 127], np.uint8))


def skin_like(skin: np.ndarray, ink: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Is the area around a candidate glyph mostly skin rather than paper?"""
    pad = max(4, int(0.35 * max(w, h)))
    H, W = skin.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    around = ink[y0:y1, x0:x1] == 0  # background pixels only
    if not around.any():
        return False
    return float((skin[y0:y1, x0:x1][around] > 0).mean()) > 0.4
