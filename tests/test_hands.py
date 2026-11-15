from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import DetectionOptions, Settings
from app.hands import asl
from app.hands.geometry import LABEL_PALM_SIGN, analyze, frame_features, palm_frame, pose_features
from app.hands.library import SignLibrary
from app.hands.motion import MotionTracker
from app.hands.recognizer import SignRecognizer
from app.hands.tracker import HandTracker
from tests.handmodel import LETTERS, RELAXED_OPEN, letter_hand, make_hand, two_hand_vowel

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def tracker():
    t = HandTracker(Settings().hand_model)
    if not t.available:
        pytest.skip(f"hand tracking unavailable: {t.reason}")
    return t


def photo_hands(tracker, name):
    return tracker.detect(cv2.imread(str(DATA / name)))


# ------------------------------------------------------------------ real photos
def test_tracker_finds_one_and_two_hands(tracker):
    assert len(photo_hands(tracker, "victory.jpg")) == 1
    assert len(photo_hands(tracker, "woman_hands.jpg")) == 2
    assert photo_hands(tracker, "victory.jpg")[0].image.shape == (21, 3)


def test_palm_side_matches_label(tracker):
    """Where curled fingers reveal the palm side, it agrees with the label-based fallback."""
    for name in ("victory.jpg", "pointing_up.jpg", "thumbs_down.jpg"):
        for hand in photo_hands(tracker, name):
            _, _, source = palm_frame(hand.world, hand.label)
            assert source == "fingers"
            w = hand.world - hand.world[0]
            u = w[9] / np.linalg.norm(w[9])
            across = w[5] - w[17]
            a = across - np.dot(across, u) * u
            n = np.cross(a / np.linalg.norm(a), u)
            curled_tip_side = np.sign(np.dot(w[16] - w[13], n))  # ring finger is curled in all three
            assert curled_tip_side == LABEL_PALM_SIGN[hand.label], name


@pytest.mark.parametrize("name, expected", [("victory.jpg", "V"), ("pointing_up.jpg", "D"), ("thumbs_down.jpg", "A")])
def test_real_photos_read_as_asl(tracker, name, expected):
    letter, conf, _ = asl.classify(analyze(photo_hands(tracker, name)[0]))
    assert letter == expected and conf >= 0.55


def test_relaxed_real_hands_are_not_letters(tracker):
    for hand in photo_hands(tracker, "woman_hands.jpg"):
        assert asl.classify(analyze(hand))[0] == ""


# -------------------------------------------------------------- synthetic hands
@pytest.mark.parametrize("letter", sorted(LETTERS))
def test_every_static_letter_clean(letter):
    for left in (False, True):
        got, conf, top = asl.classify(analyze(letter_hand(letter, left=left)))
        assert got == letter, top


def test_letters_with_landmark_noise_and_tilt():
    ok = total = 0
    for letter in LETTERS:
        for seed in range(12):
            for left in (False, True):
                got, _, _ = asl.classify(analyze(letter_hand(letter, seed=seed, noise=0.03, tilt=10, left=left)))
                ok += got == letter
                total += 1
    assert ok / total >= 0.92


def test_relaxed_open_hand_types_nothing():
    for seed in range(10):
        assert asl.classify(analyze(make_hand(RELAXED_OPEN, seed=seed, noise=0.03, tilt=10)))[0] == ""


def test_features_ignore_position_and_size_but_keep_orientation():
    a = pose_features(analyze(letter_hand("U", center=(0.3, 0.4), palm_img=0.12)))
    b = pose_features(analyze(letter_hand("U", center=(0.7, 0.6), palm_img=0.25)))
    assert np.allclose(a, b, atol=1e-4)
    h = pose_features(analyze(make_hand(LETTERS["U"], "side")))  # same shape, pointing sideways (= ASL H)
    assert np.linalg.norm(a - h) > 1.0


# ------------------------------------------------------------------- teaching
def teach(lib, letter, n=15, **kw):
    for seed in range(n):
        lib.add(letter, *frame_features([analyze(letter_hand(letter, seed=seed, noise=0.02, tilt=8, **kw))]))


def test_library_learns_and_rejects_untaught_shapes(tmp_path):
    lib = SignLibrary(tmp_path / "signs.json")
    for letter in "ABLY":
        teach(lib, letter)
    for letter in "ABLY":
        got, conf, _ = lib.classify(*frame_features([analyze(letter_hand(letter, seed=99, noise=0.02))]))
        assert got == letter and conf > 0.5
    # A shape nothing was taught for is not forced onto the nearest letter.
    got, _, _ = lib.classify(*frame_features([analyze(letter_hand("W", seed=5))]))
    assert got == ""


def test_library_persists_and_edits(tmp_path):
    path = tmp_path / "signs.json"
    lib = SignLibrary(path)
    teach(lib, "C", n=6)
    teach(lib, "O", n=4)
    lib.save()
    again = SignLibrary(path)
    assert again.counts() == {"C": 6, "O": 4}
    again.remove("C")
    assert again.counts() == {"O": 4}
    again.clear()
    assert again.counts() == {} and len(again) == 0


def test_library_ignores_incompatible_file(tmp_path):
    path = tmp_path / "signs.json"
    path.write_text('{"feature_version": 0, "signs": {"A": [{"hands": 1, "v": [1, 2]}]}}')
    lib = SignLibrary(path)
    assert lib.counts() == {} and "re-teach" in lib.load_error


def test_two_handed_signs_can_be_taught(tmp_path):
    """Two-handed alphabets (e.g. ISL): both hands and their relative position are matched."""
    lib = SignLibrary(tmp_path / "signs.json")

    def pair(letter_left, letter_right, seed, gap=0.25):
        return [analyze(letter_hand(letter_left, seed=seed, noise=0.02, left=True, center=(0.5 - gap, 0.55))),
                analyze(letter_hand(letter_right, seed=seed + 50, noise=0.02, center=(0.5 + gap, 0.55)))]

    for s in range(10):
        lib.add("A", *frame_features(pair("B", "L", s)))
        lib.add("B", *frame_features(pair("B", "B", s)))
    assert lib.classify(*frame_features(pair("B", "L", 77)))[0] == "A"
    assert lib.classify(*frame_features(pair("B", "B", 77)))[0] == "B"


# ---------------------------------------------------------------------- motion
def moving(letter, path, start=0.0, dt=0.1, **kw):
    """Hands for the letter's handshape following `path` (offsets in palm widths)."""
    out = []
    for i, (dx, dy) in enumerate(path):
        hand = letter_hand(letter, center=(0.5 + dx * 0.18, 0.4 + dy * 0.18), **kw)
        out.append((start + i * dt, hand))
    return out


def run_motion(frames):
    tracker, events = MotionTracker(), []
    for t, hand in frames:
        pose = analyze(hand)
        letter = asl.classify(pose)[0]
        ev = tracker.update(pose, letter, now=t)
        if ev:
            events.append(ev)
    return events


Z_PATH = [(0, 0), (0.4, 0), (0.8, 0), (1.2, 0), (0.8, 0.3), (0.4, 0.6), (0, 0.9), (0.4, 0.9), (0.8, 0.9), (1.2, 0.9)]
J_PATH = [(0, 0), (0, 0.25), (0, 0.5), (0, 0.75), (-0.15, 0.95), (-0.4, 1.05), (-0.65, 1.0)]


def test_z_drawn_with_index_finger():
    # Z's handshape is the index alone, pointing up (like D without the thumb contact).
    assert run_motion(moving("D", Z_PATH)) and run_motion(moving("D", Z_PATH))[0][0] == "Z"


def test_j_drawn_with_little_finger():
    events = run_motion(moving("I", J_PATH))
    assert events and events[0] == ("J", "I")


def test_held_still_or_wrong_shape_is_not_motion():
    assert run_motion(moving("I", [(0, 0)] * 12)) == []
    assert run_motion(moving("B", Z_PATH)) == []  # open hand waving is not a Z


# -------------------------------------------------------------- recogniser
def test_recognizer_on_jpeg_bytes(tracker, tmp_path):
    rec = SignRecognizer(tracker, SignLibrary(tmp_path / "signs.json"))
    data = (DATA / "victory.jpg").read_bytes()
    result, features = rec.process(data, DetectionOptions(mode="sign", alphabet="asl"))
    assert result.text == "V" and result.engine == "asl-rules" and result.found  # no trained model given
    assert len(result.hands) == 1 and len(result.hands[0]["points"]) == 21
    assert features[0] == 1

    # Teach the same photo as "K": taught signs take priority over the built-in rules.
    for _ in range(5):
        rec.library.add("K", *features)
    taught = rec.process(data, DetectionOptions(mode="sign", alphabet="asl"))[0]
    assert taught.text == "K" and taught.engine == "taught"
    only_taught = rec.process(data, DetectionOptions(mode="sign", alphabet="custom"))[0]
    assert only_taught.text == "K" and [r.engine for r in only_taught.readings] == ["taught"]


def test_recognizer_without_hands(tracker, tmp_path):
    from tests.synthetic import blank

    rec = SignRecognizer(tracker, SignLibrary(tmp_path / "signs.json"))
    result, features = rec.process(blank(), DetectionOptions(mode="sign"))
    assert not result.found and result.text == "" and result.hands == [] and features is None


# ----------------------------------------------------------------- two hands
VOWEL_TIPS = {"A": 4, "E": 8, "I": 12, "O": 16, "U": 20}  # thumb, index, middle, ring, little


def test_two_handed_vowels_are_told_apart_by_the_touched_fingertip(tmp_path):
    """Same two hand shapes; the sign is which fingertip of the open hand is touched. With
    hands at varying distances and the open hand tilted, their relative position is
    unreliable; the hand-to-hand contact features carry it (without them: ~71-88%)."""
    lib = SignLibrary(tmp_path / "signs.json")

    def feats(tip, seed):
        return frame_features([analyze(h) for h in two_hand_vowel(tip, seed=seed, noise=0.02, tilt=25, vary_size=True)])

    for vowel, tip in VOWEL_TIPS.items():
        for seed in range(15):
            lib.add(vowel, *feats(tip, seed))
    results = [lib.classify(*feats(tip, seed))[0] == vowel for vowel, tip in VOWEL_TIPS.items() for seed in range(100, 130)]
    assert sum(results) / len(results) >= 0.95
    # One hand alone is not a two-handed sign.
    assert lib.classify(*frame_features([analyze(two_hand_vowel(8)[0])]))[0] == ""


def test_hands_setting_keeps_most_prominent_hand(tracker, tmp_path):
    rec = SignRecognizer(tracker, SignLibrary(tmp_path / "signs.json"))
    data = (DATA / "woman_hands.jpg").read_bytes()
    both = rec.process(data, DetectionOptions(mode="sign"))[0]
    one = rec.process(data, DetectionOptions(mode="sign", hands=1))[0]
    assert DetectionOptions().hands == 2
    assert len(both.hands) == 2 and len(one.hands) == 1
    assert both.hands[0]["points"][0][0] < both.hands[1]["points"][0][0]  # ordered left to right
    assert one.hands[0]["points"] in [h["points"] for h in both.hands]


class StubTracker:
    available = True

    def __init__(self, hands):
        self.hands = hands

    def detect(self, bgr):
        return self.hands


def test_hand_side_tags_hidden_when_handedness_contradicts(tmp_path):
    from tests.synthetic import blank

    a = letter_hand("B", center=(0.3, 0.5))
    b = letter_hand("L", center=(0.7, 0.5))
    b_left = letter_hand("L", center=(0.7, 0.5), left=True)
    same = SignRecognizer(StubTracker([a, b]), SignLibrary(tmp_path / "s.json"))  # both labelled "Right"
    assert [h["side"] for h in same.process(blank(), DetectionOptions())[0].hands] == ["", ""]
    mixed = SignRecognizer(StubTracker([a, b_left]), SignLibrary(tmp_path / "s.json"))
    assert sorted(h["side"] for h in mixed.process(blank(), DetectionOptions())[0].hands) == ["left", "right"]
