"""Debounce noisy per-frame OCR readings into deliberate "type this" events.

Webcam OCR flickers: a letter held in front of the camera is read correctly in most frames
but not all. A reading is typed only after it has been seen in `hold_frames` frames in a
row. To type the same letter twice, the letter must leave the view (nothing in the guide
box for `release_frames` frames) and come back. A different letter can be typed straight away.

Three kinds of frame:
* confident reading - counts towards the streak (a different letter restarts it);
* unsure            - something is in view but no reading passed `min_confidence`; it
                      neither advances nor breaks the streak, and does NOT count as the
                      letter leaving (otherwise a wobbly read retypes a held letter);
* empty             - nothing letter-like in view; enough of these re-arm repeats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StabilizerState:
    candidate: str  # what is currently being held ('' when nothing confident is visible)
    progress: float  # 0..1 towards being typed
    commit: str | None  # set on the frame where the candidate gets typed


class Stabilizer:
    def __init__(self, hold_frames: int = 3, release_frames: int = 2, min_confidence: float = 0.55):
        self.configure(hold_frames, release_frames, min_confidence)
        self.reset()

    def configure(self, hold_frames: int, release_frames: int, min_confidence: float) -> None:
        self.hold_frames = max(1, hold_frames)
        self.release_frames = max(1, release_frames)
        self.min_confidence = min_confidence

    def reset(self) -> None:
        self._candidate = ""
        self._streak = 0
        self._empty_streak = 0
        self._last_commit = ""
        self._armed = True  # True when the last committed reading may be typed again
        self.last_commit_at = 0.0  # time.monotonic() of the last commit

    @property
    def last_commit(self) -> str:
        return self._last_commit

    def force(self, text: str) -> None:
        """Record a commit made outside the hold logic (a letter drawn in the air)."""
        self._last_commit, self._armed = text, False
        self._candidate, self._streak = "", 0
        self.last_commit_at = time.monotonic()

    def update(self, text: str, confidence: float, present: bool | None = None) -> StabilizerState:
        """Feed one frame. `present`: whether anything letter-like was in view
        (defaults to whether there is any text at all)."""
        if present is None:
            present = bool(text)
        reading = text if text and confidence >= self.min_confidence else ""

        if not present:
            self._empty_streak += 1
            self._candidate, self._streak = "", 0
            if self._empty_streak >= self.release_frames:
                self._armed = True
            return StabilizerState(candidate="", progress=0.0, commit=None)
        self._empty_streak = 0

        if not reading:  # unsure: keep the current candidate and streak as they are
            return StabilizerState(candidate=self._candidate, progress=self._progress(), commit=None)

        if reading == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = reading, 1

        if self._streak >= self.hold_frames and self._typeable(reading):
            self._last_commit = reading
            self._armed = False
            self.last_commit_at = time.monotonic()
            return StabilizerState(candidate=reading, progress=1.0, commit=reading)
        return StabilizerState(candidate=reading, progress=self._progress(), commit=None)

    def _typeable(self, reading: str) -> bool:
        return reading != self._last_commit or self._armed

    def _progress(self) -> float:
        if not self._candidate:
            return 0.0
        if not self._typeable(self._candidate):
            return 1.0  # already typed and still held
        return min(1.0, self._streak / self.hold_frames)
