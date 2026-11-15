"""Only use hands that are really there: a detection must persist across frames.

A false detection (a face, a cushion, a patch of skin) tends to flicker into a single frame.
A real hand is seen again in the next frame, at about the same place. A hand is therefore
confirmed only when a hand was also detected within CONFIRM_RADIUS palm widths of it in
the previous frame. Unconfirmed hands are neither drawn nor recognised. Cost: a hand
entering the view is used one frame (~0.1 s) later.
"""

from __future__ import annotations

import numpy as np

from .geometry import Hand

CONFIRM_RADIUS = 1.5  # palm widths a hand may move between consecutive frames


def _centre_and_size(hand: Hand) -> tuple[np.ndarray, float]:
    xy = hand.image[:, :2] * np.array([hand.aspect, 1.0])
    return xy.mean(axis=0), max(float(np.linalg.norm(xy[9] - xy[0])), 1e-3)


class HandGate:
    def __init__(self) -> None:
        self._previous: list[tuple[np.ndarray, float]] = []

    def reset(self) -> None:
        self._previous = []

    def filter(self, hands: list[Hand]) -> list[Hand]:
        """Hands confirmed by the previous frame. Remembers this frame's detections."""
        current = [_centre_and_size(hd) for hd in hands]
        confirmed = [
            hd for hd, (centre, size) in zip(hands, current)
            if any(np.linalg.norm(centre - pc) <= CONFIRM_RADIUS * max(size, ps) for pc, ps in self._previous)
        ]
        self._previous = current
        return confirmed
