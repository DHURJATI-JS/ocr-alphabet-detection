"""Only hands count as hands, and only letters on paper count as letters - never a face,
mouth or anything else."""

import string

import cv2
import numpy as np
import pytest

from app.config import DetectionOptions, Settings
from app.faces import FaceGuard
from app.hands.gate import HandGate
from app.hands.library import SignLibrary
from app.hands.recognizer import SignRecognizer
from app.hands.tracker import HandTracker
from tests.faces import DATA, centre_square, face_frames, fit, jpeg
from tests.handmodel import letter_hand
from tests.synthetic import available_fonts, render

FACES = face_frames()


@pytest.fixture(scope="module")
def tracker():
    t = HandTracker(Settings().hand_model)
    if not t.available:
        pytest.skip(f"hand tracking unavailable: {t.reason}")
    return t


# ----------------------------------------------------------------- hand signs
def test_faces_and_mouths_are_never_hands(tracker):
    assert len(FACES) == 84
    false_hands = [name for name, img in FACES if tracker.detect(img)]
    assert false_hands == []


def test_presence_threshold_rejects_what_the_palm_detector_is_forced_to_find():
    """Forcing the palm detector to fire (0.05) finds 'hands' on a few faces; the landmark
    model's hand-presence check removes every one of them."""
    forced = HandTracker(Settings().hand_model, min_confidence=0.05, min_presence=0.05)
    guarded = HandTracker(Settings().hand_model, min_confidence=0.05)
    assert sum(len(forced.detect(img)) for _, img in FACES) > 0
    assert sum(len(guarded.detect(img)) for _, img in FACES) == 0


@pytest.mark.parametrize("photo, hands", [
    ("victory.jpg", 1), ("pointing_up.jpg", 1), ("thumbs_up.jpg", 1), ("thumbs_down.jpg", 1),
    ("woman_hands.jpg", 2), ("portrait.jpg", 1), ("business_person.jpg", 2),
])
def test_real_hands_are_still_found(tracker, photo, hands):
    """Including a fist held in front of the face (thumbs_down) and crossed arms (portrait)."""
    assert len(tracker.detect(fit(cv2.imread(str(DATA / photo))))) == hands


def test_hand_must_persist_across_frames():
    gate = HandGate()
    here = letter_hand("A", center=(0.4, 0.5))
    moved = letter_hand("A", center=(0.45, 0.52))  # a hand moves a little between frames
    elsewhere = letter_hand("A", center=(0.9, 0.1))
    assert gate.filter([here]) == []  # first sighting: not used yet
    assert gate.filter([moved]) == [moved]
    assert gate.filter([elsewhere]) == []  # popped up somewhere else: not confirmed
    assert gate.filter([]) == []
    assert gate.filter([here]) == []  # after a gap it must be confirmed again
    assert gate.filter([here]) == [here]


def test_one_frame_flicker_never_reaches_the_letters(tracker, tmp_path):
    rec = SignRecognizer(tracker, SignLibrary(tmp_path / "signs.json"))
    gate = HandGate()
    victory = (DATA / "victory.jpg").read_bytes()
    face = jpeg(FACES[0][1])
    asl = DetectionOptions(alphabet="asl")  # no trained model passed: built-in ASL rules
    first = rec.process(victory, asl, gate=gate)[0]
    assert first.hands == [] and first.text == ""
    assert rec.process(victory, asl, gate=gate)[0].text == "V"
    assert rec.process(face, asl, gate=gate)[0].hands == []


# ------------------------------------------------------------ printed letters
@pytest.fixture(scope="module")
def manager():
    from app.ocr.manager import OCRManager

    m = OCRManager.from_settings(Settings())
    if not m.primary.available:
        pytest.skip("Tesseract unavailable")
    yield m
    m.close()


def typed(result) -> bool:
    return bool(result.text) and result.confidence >= DetectionOptions().min_confidence


def test_ocr_does_not_read_faces_as_letters(manager):
    letter_mode = sum(typed(manager.process(jpeg(centre_square(img)), DetectionOptions(mode="letter")))
                      for _, img in FACES)
    text_mode = sum(typed(manager.process(jpeg(img), DetectionOptions(mode="text"))) for _, img in FACES)
    assert letter_mode == 0  # without the guard: 23 of 84 faces typed a letter
    assert text_mode <= 2


def test_face_guard_ignores_round_letters():
    """BlazeFace "sees faces" in printed G / S / Q ...; those contain no skin and are dropped."""
    guard = FaceGuard(Settings().face_model)
    if not guard.available:
        pytest.skip(guard.reason)
    font = available_fonts()[0]
    for ch in "GSQBO":
        img = cv2.imdecode(np.frombuffer(render(ch, font=font), np.uint8), cv2.IMREAD_COLOR)
        assert guard.boxes(img) == []
    real = guard.boxes(fit(cv2.imread(str(DATA / "portrait.jpg"))))
    assert len(real) == 1


def test_guard_keeps_printed_letters(manager):
    font = available_fonts()[0]
    rough = {"noise": 14, "blur": 2.0, "gradient": 55, "rotate": -8, "clutter": True, "text_px": 120}
    for ch in string.ascii_uppercase:
        assert manager.process(render(ch, font=font, **rough), DetectionOptions(mode="letter")).text == ch
