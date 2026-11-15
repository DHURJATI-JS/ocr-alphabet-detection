"""The trained-recogniser machinery (app/hands/trained.py) and the shipped models."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.hands.geometry import analyze, frame_features, order_hands
from app.hands.trained import TrainedAlphabet, fit
from tests.handmodel import RELAXED_OPEN, letter_hand, make_hand

DATA = Path(__file__).parent / "data"
MODELS = Path(Settings().hand_model).parent
TRAIN_LETTERS = "ABLVY"


def synthetic_set(letters, seeds, none_seeds=()):
    X, y = [], []
    samples = [(letter, letter_hand, s) for letter in letters for s in seeds]
    samples += [("_", None, s) for s in none_seeds]  # "_" = relaxed / not signing
    for label, make, s in samples:
        for left in (False, True):
            hand = make(label, seed=s, noise=0.03, tilt=10, left=left) if make else \
                make_hand(RELAXED_OPEN, seed=s, noise=0.03, tilt=10, left=left)
            n, vec = frame_features([analyze(hand)])
            X.append(vec)
            y.append(label)
    return {1: (np.array(X, np.float32), np.array(y))}


@pytest.fixture(scope="module")
def small_model():
    return fit(synthetic_set(TRAIN_LETTERS, range(40), none_seeds=range(40)))


def predict(model, hands):
    return model.predict(order_hands([analyze(h) for h in hands]))


def test_fit_learns_and_generalises_to_new_samples(small_model):
    results = [predict(small_model, [letter_hand(letter, seed=seed, noise=0.03, tilt=10)])[0] == letter
               for letter in TRAIN_LETTERS for seed in (500, 501, 502)]
    assert sum(results) >= 13  # of 15 unseen samples (small, noisy synthetic training set)


def test_not_a_sign_class_is_never_typed(small_model):
    """A classifier always picks *some* class; with a 'not a sign' class a relaxed hand
    comes back as nothing instead of a forced letter."""
    assert "_" not in small_model.letters
    rejected = sum(predict(small_model, [make_hand(RELAXED_OPEN, seed=s, noise=0.03)])[0] == "" for s in range(100, 120))
    assert rejected >= 19


def test_save_and_load_round_trip(small_model, tmp_path):
    path = tmp_path / "m.npz"
    small_model.save(path, meta={"alphabet": "test"})
    loaded = TrainedAlphabet.load(path)
    assert loaded.meta["alphabet"] == "test" and loaded.letters == small_model.letters
    for letter in TRAIN_LETTERS:
        hands = [letter_hand(letter, seed=7, noise=0.03)]
        a, b = predict(small_model, hands), predict(loaded, hands)
        assert a[0] == b[0] and abs(a[1] - b[1]) < 1e-4


@pytest.mark.parametrize("alphabet", ["asl", "isl"])
def test_shipped_model_loads(alphabet):
    path = MODELS / f"{alphabet}_signs.npz"
    if not path.is_file():
        pytest.skip(f"no {path.name}; run scripts/train_signs.py --alphabet {alphabet}")
    model = TrainedAlphabet.load(path)
    assert len(model.letters) >= 20 and 1 in model.models


def test_shipped_asl_model_reads_a_real_photo():
    path = MODELS / "asl_signs.npz"
    if not path.is_file():
        pytest.skip("no asl model")
    from app.hands.tracker import HandTracker

    tracker = HandTracker(Settings().hand_model)
    hands = tracker.detect(cv2.imread(str(DATA / "victory.jpg")))
    assert predict(TrainedAlphabet.load(path), hands)[0] == "V"
