import pytest

from app.letters import normalize_letter, normalize_text
from app.ocr.llm import parse_letter
from app.stabilizer import Stabilizer


@pytest.mark.parametrize(
    "raw, case, expected",
    [
        ("A", "auto", "A"),
        (" k\n", "upper", "K"),
        ("Q", "lower", "q"),
        ("А", "auto", "A"),  # Cyrillic A
        ("К", "upper", "K"),  # Cyrillic Ka
        ("Ｍ", "auto", "M"),  # fullwidth
        ("0", "upper", "O"),
        ("|", "upper", "I"),
        ("É", "auto", "E"),
        ("l", "upper", "I"),  # a bare bar in UPPERCASE mode is an I, not an L
        ("I", "lower", "l"),
        ("l", "auto", "l"),
        ("L", "upper", "L"),
        ("AB", "auto", ""),  # not a single letter
        ("7", "auto", ""),
        ("", "auto", ""),
    ],
)
def test_normalize_letter(raw, case, expected):
    assert normalize_letter(raw, case) == expected


def test_normalize_text_maps_cyrillic_and_collapses_spaces():
    assert normalize_text("АВС  hello\n wОrld!", "upper") == "ABC HELLO WORLD"


@pytest.mark.parametrize(
    "reply, expected",
    [("A", "A"), ("b.", "b"), ("**Q**", "Q"), ("The letter is K.", "K"), ("It's the letter 'R'", "R"),
     ("NONE", ""), ("none of these", ""), ("I think it's K", ""), ("", "")],
)
def test_parse_llm_reply(reply, expected):
    assert parse_letter(reply) == expected


def feed(stab, readings, conf=0.9):
    return [stab.update(r, conf).commit for r in readings]


def test_commits_after_hold_frames():
    s = Stabilizer(hold_frames=3, release_frames=2)
    assert feed(s, ["A", "A", "A"]) == [None, None, "A"]


def test_does_not_repeat_while_letter_is_held():
    s = Stabilizer(hold_frames=2, release_frames=2)
    assert feed(s, ["A"] * 8) == [None, "A"] + [None] * 6


def test_same_letter_again_after_it_leaves_the_view():
    s = Stabilizer(hold_frames=2, release_frames=2)
    assert feed(s, ["A", "A", "", "", "A", "A"]) == [None, "A", None, None, None, "A"]


def test_one_blank_frame_is_not_a_release():
    s = Stabilizer(hold_frames=2, release_frames=2)
    assert feed(s, ["A", "A", "", "A", "A"]) == [None, "A", None, None, None]


def test_new_letter_types_without_a_gap():
    s = Stabilizer(hold_frames=2, release_frames=2)
    assert feed(s, ["A", "A", "B", "B"]) == [None, "A", None, "B"]


def test_flicker_resets_the_streak():
    s = Stabilizer(hold_frames=3, release_frames=2)
    assert feed(s, ["A", "A", "R", "A", "A", "A"]) == [None, None, None, None, None, "A"]


def test_low_confidence_never_types():
    s = Stabilizer(hold_frames=2, release_frames=1, min_confidence=0.6)
    assert [s.update("A", 0.4).commit for _ in range(5)] == [None] * 5


def test_unsure_frames_do_not_retype_a_held_letter():
    """Regression: an 'I' held up read as I, I, (unsure), (unsure), I ... was typed twice."""
    s = Stabilizer(hold_frames=2, release_frames=2, min_confidence=0.55)
    frames = [("I", 0.75, True), ("I", 0.75, True), ("I", 0.42, True), ("", 0.0, True), ("I", 0.75, True),
              ("I", 0.75, True)]
    assert [s.update(t, c, present=p).commit for t, c, p in frames] == [None, "I", None, None, None, None]


def test_unsure_frame_does_not_break_the_streak():
    s = Stabilizer(hold_frames=3, release_frames=2)
    frames = [("K", 0.9), ("", 0.0), ("K", 0.9), ("K", 0.9)]
    assert [s.update(t, c, present=True).commit for t, c in frames] == [None, None, None, "K"]


def test_letter_leaving_the_view_rearms():
    s = Stabilizer(hold_frames=2, release_frames=2)
    frames = [("O", 0.9, True), ("O", 0.9, True), ("", 0.0, False), ("", 0.0, False), ("O", 0.9, True),
              ("O", 0.9, True)]
    assert [s.update(t, c, present=p).commit for t, c, p in frames] == [None, "O", None, None, None, "O"]


def test_progress_and_reset():
    s = Stabilizer(hold_frames=4, release_frames=2)
    assert s.update("C", 0.9).progress == pytest.approx(0.25)
    assert s.update("C", 0.9).progress == pytest.approx(0.5)
    s.reset()
    assert s.update("C", 0.9).progress == pytest.approx(0.25)
