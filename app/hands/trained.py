"""Sign-alphabet recognisers trained on real photos (see scripts/train_signs.py).

Per number of hands in view (1 or 2) there is a small neural network (MLP) over the app's
pose features. Training uses scikit-learn; the saved model is plain numpy arrays, so the
app needs no ML library to run it.

A classifier always answers *something*, even for a hand that isn't signing. So each
model also keeps reference poses per letter: if the live pose is further from every real
example of the predicted letter than held-out examples ever were, the answer is rejected
rather than typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HIDDEN = (256, 128)
REFS_PER_LETTER = 300  # reference poses kept per letter for the "is this a real sign" check
REJECT_PERCENTILE = 99.0  # of held-out distances; beyond (x margin) = not a known sign
REJECT_MARGIN = 1.25
NONE = "_"  # class of relaxed / non-signing hands


@dataclass
class HandModel:
    """One MLP for a fixed number of hands."""

    hands: int
    classes: np.ndarray  # letters
    mean: np.ndarray
    std: np.ndarray
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    refs: np.ndarray  # standardised reference poses
    ref_labels: np.ndarray
    reject_distance: float

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)

    def nearest_same_letter(self, X: np.ndarray, letters: np.ndarray) -> np.ndarray:
        """Distance from each pose to the closest reference pose of the given letter."""
        Z = (X - self.mean) / self.std
        out = np.full(len(X), np.inf)
        for i, (z, letter) in enumerate(zip(Z, letters)):
            refs = self.refs[self.ref_labels == letter]
            if len(refs):
                out[i] = float(np.sqrt(((refs - z) ** 2).sum(axis=1)).min())
        return out

    def predict(self, vec: np.ndarray) -> tuple[str, float, list[tuple[str, float]], bool]:
        """(letter, probability, top-3, known_pose)."""
        p = self.probabilities(vec[None, :])[0]
        order = np.argsort(p)[::-1]
        letter = str(self.classes[order[0]])
        top = [(str(self.classes[i]), float(p[i])) for i in order if self.classes[i] != NONE][:3]
        if letter == NONE:  # a hand that is not signing
            return letter, float(p[order[0]]), top, False
        known = bool(self.nearest_same_letter(vec[None, :], np.array([letter]))[0] <= self.reject_distance)
        return letter, float(p[order[0]]), top, known


class TrainedAlphabet:
    def __init__(self, models: dict[int, HandModel], meta: dict | None = None) -> None:
        self.models = models
        self.meta = meta or {}

    @property
    def letters(self) -> list[str]:
        return sorted({str(c) for m in self.models.values() for c in m.classes} - {NONE})

    # ----------------------------------------------------------------- inference
    def predict(self, poses) -> tuple[str, float, list[tuple[str, float]]]:
        """Best letter for the hands in view: ('', p, top) when it is not a known sign.

        With two hands in view, both the two-handed model and the one-handed model on each
        hand are asked (a one-handed letter with the other hand resting in view)."""
        from .geometry import frame_features, pose_features

        candidates = []
        n, vec = frame_features(poses)
        if n in self.models:
            candidates.append(self.models[n].predict(vec))
        if len(poses) == 2 and 1 in self.models:
            for pose in poses:
                letter, p, top, known = self.models[1].predict(pose_features(pose))
                candidates.append((letter, p * 0.9, top, known))  # prefer the full two-hand reading
        known = [c for c in candidates if c[3]]
        if not known:
            best = max(candidates, key=lambda c: c[1], default=("", 0.0, [], False))
            return "", best[1], best[2]
        letter, p, top, _ = max(known, key=lambda c: c[1])
        return letter, p, top

    # --------------------------------------------------------------- persistence
    def save(self, path: Path, meta: dict | None = None) -> None:
        arrays: dict[str, np.ndarray] = {}
        for n, m in self.models.items():
            arrays[f"h{n}_classes"] = m.classes.astype(str)
            arrays[f"h{n}_mean"], arrays[f"h{n}_std"] = m.mean, m.std
            for i, (W, b) in enumerate(zip(m.weights, m.biases)):
                arrays[f"h{n}_W{i}"], arrays[f"h{n}_b{i}"] = W.astype(np.float32), b.astype(np.float32)
            arrays[f"h{n}_refs"] = m.refs.astype(np.float16)
            arrays[f"h{n}_ref_labels"] = m.ref_labels.astype(str)
            arrays[f"h{n}_reject"] = np.array(m.reject_distance, np.float32)
        arrays["meta"] = np.array(json.dumps({**self.meta, **(meta or {})}))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "TrainedAlphabet":
        z = np.load(path, allow_pickle=False)
        models = {}
        for n in (1, 2):
            if f"h{n}_classes" not in z:
                continue
            layers = sorted({int(k[len(f"h{n}_W"):]) for k in z.files if k.startswith(f"h{n}_W")})
            models[n] = HandModel(
                hands=n, classes=z[f"h{n}_classes"], mean=z[f"h{n}_mean"], std=z[f"h{n}_std"],
                weights=[z[f"h{n}_W{i}"] for i in layers], biases=[z[f"h{n}_b{i}"] for i in layers],
                refs=z[f"h{n}_refs"].astype(np.float32), ref_labels=z[f"h{n}_ref_labels"],
                reject_distance=float(z[f"h{n}_reject"]),
            )
        return cls(models, json.loads(str(z["meta"])))


# ------------------------------------------------------------------------ training
def _fit_hand_model(X: np.ndarray, y: np.ndarray, n: int, seed: int = 0):
    from sklearn.neural_network import MLPClassifier

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-3
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=1e-3, batch_size=256, max_iter=300,
                        early_stopping=True, n_iter_no_change=12, random_state=seed)
    clf.fit((X - mean) / std, y)
    rng = np.random.default_rng(seed)
    refs, ref_labels = [], []
    for letter in clf.classes_:
        idx = np.flatnonzero(y == letter)
        pick = rng.choice(idx, size=min(REFS_PER_LETTER, len(idx)), replace=False)
        refs.append((X[pick] - mean) / std)
        ref_labels += [letter] * len(pick)
    return HandModel(hands=n, classes=clf.classes_, mean=mean, std=std, weights=list(clf.coefs_),
                     biases=list(clf.intercepts_), refs=np.vstack(refs), ref_labels=np.array(ref_labels),
                     reject_distance=np.inf)


def _calibrate_reject(model: HandModel, X: np.ndarray, y: np.ndarray) -> float:
    """Reject distance from held-out poses of the true letter."""
    d = model.nearest_same_letter(X, y)
    d = d[np.isfinite(d)]
    return float(np.percentile(d, REJECT_PERCENTILE) * REJECT_MARGIN) if len(d) else np.inf


def fit(data: dict[int, tuple[np.ndarray, np.ndarray]], seed: int = 0) -> TrainedAlphabet:
    """Train one model per hand count from {hands: (features, letters)}.

    10% of the poses are kept aside to calibrate the "not a known sign" distance."""
    models = {}
    rng = np.random.default_rng(seed)
    for n, (X, y) in sorted(data.items()):
        if len(X) < 50 or len(np.unique(y)) < 2:
            continue
        order = rng.permutation(len(X))
        cut = max(1, len(X) // 10)
        cal, tr = order[:cut], order[cut:]
        model = _fit_hand_model(X[tr], y[tr], n, seed)
        model.reject_distance = _calibrate_reject(model, X[cal], y[cal])
        models[n] = model
    return TrainedAlphabet(models)
