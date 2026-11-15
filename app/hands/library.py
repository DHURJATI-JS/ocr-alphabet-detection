"""Signs taught by the user: recorded examples per letter, nearest-neighbour matching.

Teaching makes recognition fit one person's hands, camera and lighting, and works for any
sign alphabet, including two-handed ones such as ISL. Examples are saved to a JSON file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURE_VERSION = 3  # bump when geometry features change; old examples are then ignored (v3: 3D orientation)
K = 7
DEFAULT_SCALE = 0.3  # typical spread of one sign's examples, before there is data to measure it
MIN_SCALE = 0.2  # a very still recording must not make matching brittle later
REJECT_FACTOR = 4.0  # further than this x spread from every example: not a taught sign


class SignLibrary:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._samples: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self._scale: float | None = None
        self.load_error = ""
        self.load()

    # ------------------------------------------------------------- persistence
    def load(self) -> None:
        with self._lock:
            self._samples.clear()
            self._scale = None
            if not self.path.is_file():
                return
            try:
                data = json.loads(self.path.read_text())
                if data.get("feature_version") != FEATURE_VERSION:
                    self.load_error = "saved signs were recorded with an older version; please re-teach them"
                    return
                for letter, items in data.get("signs", {}).items():
                    for item in items:
                        self._samples[letter].append((int(item["hands"]), np.asarray(item["v"], dtype=np.float32)))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.load_error = f"could not read {self.path}: {exc}"

    def save(self) -> None:
        with self._lock:
            data = {
                "feature_version": FEATURE_VERSION,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "signs": {
                    letter: [{"hands": h, "v": [round(float(x), 4) for x in v]} for h, v in items]
                    for letter, items in sorted(self._samples.items()) if items
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, self.path)

    # ----------------------------------------------------------------- editing
    def add(self, letter: str, hands: int, vector: np.ndarray) -> int:
        with self._lock:
            self._samples[letter].append((hands, np.asarray(vector, dtype=np.float32)))
            self._scale = None
            return len(self._samples[letter])

    def remove(self, letter: str) -> None:
        with self._lock:
            self._samples.pop(letter, None)
            self._scale = None

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._scale = None

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {letter: len(items) for letter, items in sorted(self._samples.items()) if items}

    def hands_by_letter(self) -> dict[str, list[int]]:
        """Which hand counts each taught letter was recorded with, e.g. {"A": [2]}."""
        with self._lock:
            return {letter: sorted({h for h, _ in items}) for letter, items in sorted(self._samples.items()) if items}

    def hand_counts(self) -> set[int]:
        with self._lock:
            return {h for items in self._samples.values() for h, _ in items}

    def __len__(self) -> int:
        with self._lock:
            return sum(len(items) for items in self._samples.values())

    # ---------------------------------------------------------------- matching
    def _typical_distance(self) -> float:
        """How much one sign varies: median distance of examples from their letter's average.

        (Not nearest-neighbour distance: a recording is a run of near-identical consecutive
        frames, which would make that tiny and matching brittle.)
        """
        if self._scale is not None:
            return self._scale
        spreads = []
        for items in self._samples.values():
            for hands in {h for h, _ in items}:
                vecs = np.stack([v for h, v in items if h == hands])
                if len(vecs) >= 3:
                    spreads.append(float(np.median(np.linalg.norm(vecs - vecs.mean(axis=0), axis=1))))
        self._scale = max(float(np.median(spreads)), MIN_SCALE) if spreads else DEFAULT_SCALE
        return self._scale

    def classify(self, hands: int, vector: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """(letter or '', confidence 0..1, top candidates with vote share)."""
        with self._lock:
            pool = [(letter, v) for letter, items in self._samples.items() for h, v in items if h == hands]
            if not pool:
                return "", 0.0, []
            scale = self._typical_distance()
            letters = [letter for letter, _ in pool]
            dists = np.linalg.norm(np.stack([v for _, v in pool]) - vector[None, :], axis=1)
        order = np.argsort(dists)[: min(K, len(pool))]
        votes: dict[str, float] = defaultdict(float)
        for i in order:
            votes[letters[i]] += 1.0 / (float(dists[i]) + 0.05)
        total = sum(votes.values())
        ranked = sorted(((letter, w / total) for letter, w in votes.items()), key=lambda x: x[1], reverse=True)
        best, share = ranked[0]
        nearest = min(float(dists[i]) for i in order if letters[i] == best)
        closeness = float(np.exp(-((nearest / (2.5 * scale)) ** 2)))
        confidence = share * closeness
        if nearest > REJECT_FACTOR * scale:
            return "", confidence, ranked[:3]
        return best, confidence, ranked[:3]
