"""MediaPipe hand landmark detection (up to two hands)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .geometry import Hand

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's C++ INFO logging


# Two thresholds, deliberately different (measured on 84 face-only frames incl. mouth
# close-ups, and 9 real hands incl. a fist in front of a face; see tests/test_hands.py):
# * palm detection 0.5 - raising it only loses real hands (0.6 already missed one);
# * hand presence 0.7 - the landmark model's own "is this really a hand" score. Things
#   that are not hands (faces, mouths) score below 0.5 even when the palm detector is
#   forced to fire; real hands kept passing up to 0.8.
DETECTION_CONFIDENCE = 0.5
PRESENCE_CONFIDENCE = 0.7


class HandTracker:
    name = "mediapipe"
    label = "MediaPipe Hands"

    def __init__(self, model_path: Path, max_hands: int = 2, min_confidence: float = DETECTION_CONFIDENCE,
                 min_presence: float = PRESENCE_CONFIDENCE) -> None:
        self.available = False
        self.reason = ""
        self.detail = ""
        self._lock = threading.Lock()
        model_path = Path(model_path)
        if not model_path.is_file():
            self.reason = f"missing {model_path} (run: python scripts/download_models.py)"
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            self.reason = f"mediapipe not installed ({exc})"
            return
        try:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_presence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001
            self.reason = f"could not load hand model: {exc}"
            return
        self._mp = mp
        self.available = True
        self.detail = (f"mediapipe {mp.__version__}, up to {max_hands} hands, "
                       f"detection >= {min_confidence:g}, hand presence >= {min_presence:g}")

    def detect(self, bgr: np.ndarray) -> list[Hand]:
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:  # one landmarker instance, not re-entrant
            result = self._landmarker.detect(image)
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[i][0]
            hands.append(
                Hand(
                    label=category.category_name,
                    score=float(category.score),
                    image=np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64),
                    world=np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]], dtype=np.float64),
                    aspect=w / float(h),
                )
            )
        return hands

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "available": self.available,
                "reason": self.reason, "detail": self.detail}
