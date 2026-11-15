"""J and Z: the two ASL letters that are drawn in the air rather than held still.

  J - the I handshape (little finger up), the little fingertip draws a J: down, then curving.
  Z - the index finger alone, its tip draws a Z: across, diagonally back, across again.

The tracker keeps the last ~1.6 s of fingertip positions (measured in palm widths, so it
works at any distance from the camera) and looks for those paths.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .geometry import HandPose

WINDOW_S = 1.6
COOLDOWN_S = 1.0
MIN_FRAMES = 5


@dataclass
class _Sample:
    t: float
    shape: str  # "index" (only index straight), "pinky" (only pinky straight) or ""
    index_tip: np.ndarray  # image coords / palm size
    pinky_tip: np.ndarray
    static: str  # what the static recogniser said on this frame


def _shape(pose: HandPose) -> str:
    straight = {k: f.total < 75 for k, f in pose.fingers.items()}
    folded = {k: f.total > 95 for k, f in pose.fingers.items()}
    if straight["index"] and folded["middle"] and folded["ring"] and folded["pinky"]:
        return "index"
    if straight["pinky"] and folded["index"] and folded["middle"] and folded["ring"]:
        return "pinky"
    return ""


def _reversals(xs: np.ndarray, min_stroke: float) -> int:
    """Number of direction changes along xs, ignoring wiggles shorter than min_stroke."""
    count, direction, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        delta = x - anchor
        if direction >= 0 and delta <= -min_stroke:
            count += direction == 1
            direction, anchor = -1, x
        elif direction <= 0 and delta >= min_stroke:
            count += direction == -1
            direction, anchor = 1, x
        elif (direction == 1 and x > anchor) or (direction == -1 and x < anchor):
            anchor = x
    return count


class MotionTracker:
    def __init__(self) -> None:
        self._history: deque[_Sample] = deque()
        self._cooldown_until = 0.0

    def reset(self) -> None:
        self._history.clear()

    def update(self, pose: HandPose | None, static_letter: str, now: float | None = None) -> tuple[str, str] | None:
        """Feed one frame; returns (letter, base handshape letter) when a J or Z completes."""
        now = time.monotonic() if now is None else now
        if pose is None:
            self._history.clear()
            return None
        scale = max(pose.palm_img, 1e-3)
        xy = pose.hand.image[:, :2] * np.array([pose.hand.aspect, 1.0])
        self._history.append(_Sample(now, _shape(pose), xy[8] / scale, xy[20] / scale, static_letter))
        while self._history and now - self._history[0].t > WINDOW_S:
            self._history.popleft()
        if now < self._cooldown_until or len(self._history) < MIN_FRAMES:
            return None

        found = self._detect_z() or self._detect_j()
        if found:
            statics = Counter(s.static for s in self._history if s.static)
            base = statics.most_common(1)[0][0] if statics else ""
            self._history.clear()
            self._cooldown_until = now + COOLDOWN_S
            return found, base
        return None

    def _frames(self, shape: str) -> list[_Sample]:
        frames = [s for s in self._history if s.shape == shape]
        return frames if len(frames) >= max(MIN_FRAMES, 0.7 * len(self._history)) else []

    def _detect_z(self) -> str:
        frames = self._frames("index")
        if not frames:
            return ""
        path = np.array([s.index_tip for s in frames])
        if _reversals(path[:, 0], 0.45) >= 2 and path[-1, 1] - path[0, 1] >= 0.3:
            return "Z"
        return ""

    def _detect_j(self) -> str:
        frames = self._frames("pinky")
        if not frames:
            return ""
        path = np.array([s.pinky_tip for s in frames])
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        lowest = int(np.argmax(path[:, 1]))
        drop = path[lowest, 1] - path[0, 1]
        hook = abs(path[-1, 0] - path[lowest, 0]) + abs(path[lowest, 0] - path[0, 0])
        if length >= 0.9 and drop >= 0.35 and hook >= 0.3:
            return "J"
        return ""
