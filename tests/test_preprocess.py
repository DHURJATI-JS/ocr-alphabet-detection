import numpy as np
import pytest

from app.preprocess import decode_image, prepare_letter, prepare_text
from tests.synthetic import available_fonts, blank, render

FONT = (available_fonts() or [None])[0]


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        decode_image(b"")


def test_blank_paper_has_no_glyph():
    for seed in range(5):
        assert not prepare_letter(decode_image(blank(seed=seed, noise=10))).found


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"noise": 10, "blur": 1.5, "gradient": 50, "rotate": 6},
        {"inverted": True, "noise": 8},
        {"clutter": True, "gradient": 55, "noise": 14, "text_px": 120},
    ],
    ids=["clean", "webcam", "inverted", "clutter"],
)
def test_glyph_is_isolated_dark_on_white(kw):
    p = prepare_letter(decode_image(render("M", font=FONT, **kw)))
    assert p.found
    h, w = p.binary.shape
    # Margin is white, glyph is black, whatever the original polarity.
    border = np.concatenate([p.binary[0], p.binary[-1], p.binary[:, 0], p.binary[:, -1]])
    assert border.min() == 255
    assert (p.binary == 0).mean() > 0.05
    assert p.gray[0].mean() > 200


def test_glyph_box_points_at_the_letter():
    # Letter drawn near the centre of a 640x400 frame.
    p = prepare_letter(decode_image(render("H", font=FONT, text_px=160, seed=3)))
    g = p.glyph
    cx, cy = g.x + g.w / 2, g.y + g.h / 2
    assert 0.35 < cx < 0.65 and 0.3 < cy < 0.7
    assert 0.1 < g.w < 0.4 and 0.2 < g.h < 0.6
    # to_source of the full prepared image covers the glyph box.
    full = p.to_source(0, 0, p.binary.shape[1], p.binary.shape[0])
    assert full.x <= g.x and full.y <= g.y
    assert full.x + full.w >= g.x + g.w - 1e-3


def test_dot_of_lowercase_i_is_merged():
    p = prepare_letter(decode_image(render("i", font=FONT, text_px=200)))
    assert p.found
    assert p.glyph.h / p.glyph.w > 2.5  # stem + dot, not just the stem's lower part


def test_text_mode_keeps_whole_region():
    p = prepare_text(decode_image(render("HELLO", font=FONT, text_px=90, size=(640, 240))))
    assert p.found
    assert p.scale >= 1.0
    assert p.binary.shape == p.gray.shape
