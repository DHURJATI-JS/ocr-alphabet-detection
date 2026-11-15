"""Generate webcam-like frames of letters for tests and accuracy checks."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/courbd.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def available_fonts() -> list[str]:
    return [p for p in _FONT_CANDIDATES if Path(p).is_file()]


def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render(
    text: str,
    *,
    font: str | None = None,
    size: tuple[int, int] = (640, 400),
    text_px: int = 180,
    seed: int = 0,
    rotate: float = 0.0,
    noise: float = 0.0,
    blur: float = 0.0,
    gradient: float = 0.0,
    inverted: bool = False,
    clutter: bool = False,
    jpeg_quality: int | None = 85,
) -> bytes:
    """Render `text` like a card held up to a webcam and return encoded image bytes."""
    rng = random.Random(seed)
    w, h = size
    bg, ink = (40, 235) if inverted else (225, 30)
    img = Image.new("L", (w, h), bg)
    draw = ImageDraw.Draw(img)
    f = _font(font, text_px)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
    tw, th = right - left, bottom - top
    jitter_x, jitter_y = rng.randint(-w // 12, w // 12), rng.randint(-h // 12, h // 12)
    draw.text(((w - tw) / 2 - left + jitter_x, (h - th) / 2 - top + jitter_y), text, font=f, fill=ink)
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC, fillcolor=bg)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    arr = np.asarray(img, dtype=np.float32)
    if gradient:  # uneven lighting: darker on one side
        ramp = np.linspace(-gradient, gradient, w, dtype=np.float32)[None, :]
        arr = arr + ramp
    if clutter:  # a dark object (hand, table edge) entering the frame
        arr[:, : w // 10] = arr[:, : w // 10] * 0.25
        arr[h - h // 12 :, :] = arr[h - h // 12 :, :] * 0.4
    if noise:
        arr = arr + np.random.default_rng(seed).normal(0, noise, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")

    buf = io.BytesIO()
    if jpeg_quality:
        out.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        out.save(buf, format="PNG")
    return buf.getvalue()


def blank(size: tuple[int, int] = (640, 400), noise: float = 6.0, seed: int = 0) -> bytes:
    arr = np.full((size[1], size[0]), 220, np.float32)
    arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()
