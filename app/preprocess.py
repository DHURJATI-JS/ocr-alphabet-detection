"""Prepare webcam frames for OCR.

The browser sends only the guide-box region of the webcam frame. From that we build:

* letter mode: find the single most prominent glyph, crop it tightly with a white margin and
  scale it to a fixed size. Engines read isolated, well-framed characters far more reliably
  than a letter floating in a busy frame (Apple Vision ignores it completely otherwise).
* text mode: the whole region, contrast-normalised and upscaled for small text.

Every prepared image is dark ink on a light background, whatever the original polarity.
`gray` is the contrast-normalised picture (for Apple / Windows OCR), `binary` is the
thresholded version (for Tesseract). Both share the same geometry, so a box found in
either maps back to the source frame with `to_source`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .faces import skin_like, skin_mask

FaceBoxes = list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) pixels, from faces.FaceGuard

LETTER_SIZE = 72  # longest side of the glyph after scaling, in px
LETTER_PAD = 0.35  # white margin around the glyph, as a fraction of its longest side
TEXT_TARGET = 1000  # longest side of a text-mode image after upscaling, in px
MAX_SOURCE_SIDE = 1600  # larger uploads are downscaled first


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    def as_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "w": round(self.w, 4), "h": round(self.h, 4)}


@dataclass
class Prepared:
    gray: np.ndarray  # uint8, dark text on light background
    binary: np.ndarray  # uint8 0/255, black text on white
    src_w: int
    src_h: int
    # prepared_px = (source_px - origin) * scale
    origin_x: float = 0.0
    origin_y: float = 0.0
    scale: float = 1.0
    found: bool = True  # letter mode: False when no glyph-like shape is in view
    glyph: Box | None = None  # letter mode: the glyph's box, normalised to the source

    def to_source(self, x: float, y: float, w: float, h: float) -> Box:
        """Map a box in prepared-image pixels to 0..1 coordinates of the source image."""
        sx = x / self.scale + self.origin_x
        sy = y / self.scale + self.origin_y
        return Box(sx / self.src_w, sy / self.src_h, (w / self.scale) / self.src_w, (h / self.scale) / self.src_h)


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/WebP bytes into a BGR image."""
    if not data:
        raise ValueError("empty image")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image (send JPEG, PNG or WebP bytes)")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SOURCE_SIDE:
        f = MAX_SOURCE_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return img


MIN_INK_CONTRAST = 45  # flattened gray levels between ink and paper; below this = nothing there


def flatten(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove uneven lighting and normalise polarity.

    Returns (flat, light_ink): `flat` has dark ink on an evenly white background.

    Polarity: paper covers most of the guide box, so the histogram median is paper and the
    ink is whichever tail reaches further from it. (Comparing morphological residuals does
    not work: the bright counter of an 'O' or 'G' is as large as its dark stroke.)

    The paper's illumination is then estimated with a morphological closing (which erases
    dark strokes thinner than the kernel) and divided out, so gradients and soft shadows vanish.
    """
    h, w = gray.shape
    f = min(1.0, 320.0 / max(h, w))
    small = cv2.resize(gray, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else gray
    lo, median, hi = np.percentile(cv2.GaussianBlur(small, (5, 5), 0), (1, 50, 99))
    light_ink = bool(hi - median > median - lo)
    if light_ink:
        gray, small = 255 - gray, 255 - small
    k = max(9, (min(small.shape) // 6) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    smooth = max(3, (k // 2) | 1)
    background = cv2.blur(background, (smooth, smooth))
    background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, background, scale=255), light_ink


def ink_mask(flat: np.ndarray) -> np.ndarray:
    """Binary mask (ink = 255) of a flattened image; empty when there is no real contrast."""
    blur = cv2.GaussianBlur(flat, (3, 3), 0)
    thresh, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = blur[mask > 0]
    paper = blur[mask == 0]
    if ink.size == 0 or paper.size == 0 or float(np.median(paper)) - float(np.median(ink)) < MIN_INK_CONTRAST:
        return np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# A printed letter sits on paper: the area around it is smooth and light. Measured on the
# flattened image (520 synthetic letters incl. noise, blur, clutter, white-on-black vs.
# shapes picked out of faces): letters' surroundings std <= 15 and <= 1.5% dark speckle;
# faces / hair / clothing 25-76 and up to 80%.
PAPER_MAX_STD = 22.0
# A letter being shown is held inside the guide box at a readable size. Small shapes or
# shapes at the box's edge are background (a window frame read as 'b' in a portrait).
MIN_GLYPH_HEIGHT = 0.12  # fraction of the guide box height
EDGE_MARGIN = 0.12  # fraction of the guide box the glyph's centre must stay away from the edge
PAPER_MAX_DARK = 0.06


def on_paper(flat: np.ndarray, mask: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Are the surroundings of this shape (all ink excluded) smooth like paper?"""
    pad = max(4, int(0.35 * max(w, h)))
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    ink = cv2.dilate(mask[y0:y1, x0:x1], np.ones((5, 5), np.uint8)) > 0
    ring = flat[y0:y1, x0:x1][~ink]
    if ring.size < 20:
        return False
    return float(ring.std()) <= PAPER_MAX_STD and float((ring < 200).mean()) <= PAPER_MAX_DARK


def _exclude_faces(mask: np.ndarray, faces: FaceBoxes | None) -> np.ndarray:
    """Remove ink inside face regions (eyes, nostrils and mouth are not letters)."""
    if faces:
        mask = mask.copy()
        for x0, y0, x1, y1 in faces:
            mask[y0:y1, x0:x1] = 0
    return mask


def _pick_glyph(mask: np.ndarray, skin: np.ndarray | None = None,
                flat: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    """Pick the most prominent glyph-like connected component; return its (x, y, w, h).
    With `skin`, shapes surrounded by skin are ignored; with `flat`, shapes whose
    surroundings are not smooth like paper are ignored."""
    h, w = mask.shape
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = h * w * 0.002
    min_height = h * MIN_GLYPH_HEIGHT
    best, best_score = None, 0.0
    candidates = []
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        candidates.append((x, y, cw, ch, area))
        touches_edge = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
        if touches_edge or area < min_area or ch < min_height:
            continue
        if cw > w * 0.95 or ch > h * 0.95:
            continue
        fill = area / float(cw * ch)
        aspect = ch / float(cw)
        if aspect < 0.15 or aspect > 8:
            continue  # lines and streaks
        if fill > 0.9 and aspect < 2.5:
            continue  # solid blobs (shadows, fingers); a solid tall bar is still an 'I'/'l'
        if skin is not None and skin_like(skin, mask, int(x), int(y), int(cw), int(ch)):
            continue  # a mouth / eye / nostril on a face, not a letter on paper
        if flat is not None and not on_paper(flat, mask, int(x), int(y), int(cw), int(ch)):
            continue  # textured surroundings: hair, clothing, furniture
        cx, cy = centroids[i]
        if not (EDGE_MARGIN * w <= cx <= (1 - EDGE_MARGIN) * w and EDGE_MARGIN * h <= cy <= (1 - EDGE_MARGIN) * h):
            continue  # hugging the guide box edge: background clutter, not the letter shown
        dist = np.hypot(cx / w - 0.5, cy / h - 0.5) / 0.7071  # 0 at centre, 1 at a corner
        score = area * (1.0 - 0.6 * dist)
        if score > best_score:
            best, best_score = (int(x), int(y), int(cw), int(ch)), score
    if best is None:
        return None

    # Merge detached parts of the same glyph (dot of 'i'/'j', strokes broken by glare).
    bx, by, bw, bh = best
    x0, y0, x1, y1 = bx, by, bx + bw, by + bh
    reach_x, reach_y = bw * 0.15, bh * 0.5
    for x, y, cw, ch, area in candidates:
        if (x, y, cw, ch) == best or area < min_area * 0.15:
            continue
        inside_x = x >= bx - reach_x and x + cw <= bx + bw + reach_x
        near_y = y + ch >= by - reach_y and y <= by + bh + reach_y
        if inside_x and near_y:
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + cw), max(y1, y + ch)
    return x0, y0, x1 - x0, y1 - y0


def prepare_letter(bgr: np.ndarray, faces: FaceBoxes | None = None, ignore_skin: bool = True) -> Prepared:
    """Isolate the single most prominent printed glyph. `faces` are masked out, and with
    ignore_skin shapes on skin-coloured surroundings are skipped."""
    src_h, src_w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat, _ = flatten(gray)
    mask = _exclude_faces(ink_mask(flat), faces)
    glyph = _pick_glyph(mask, skin_mask(bgr), flat) if ignore_skin else _pick_glyph(mask)
    if glyph is None:
        blank = np.full((8, 8), 255, np.uint8)
        return Prepared(gray=blank, binary=blank, src_w=src_w, src_h=src_h, found=False)

    x, y, w, h = glyph
    scale = LETTER_SIZE / float(max(w, h))
    pad = round(LETTER_SIZE * LETTER_PAD)
    out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))

    ink = cv2.resize(mask[y : y + h, x : x + w], (out_w, out_h), interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(ink, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.copyMakeBorder(255 - ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # Gray crop: stretch the glyph's local contrast so the background reads as near-white.
    crop = flat[y : y + h, x : x + w]
    lo, hi = np.percentile(crop, 2), np.percentile(crop, 98)
    crop = np.clip((crop.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    gray_out = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    prepared = Prepared(
        gray=gray_out,
        binary=binary,
        src_w=src_w,
        src_h=src_h,
        origin_x=x - pad / scale,
        origin_y=y - pad / scale,
        scale=scale,
        found=True,
    )
    prepared.glyph = Box(x / src_w, y / src_h, w / src_w, h / src_h)
    return prepared


def prepare_text(bgr: np.ndarray, faces: FaceBoxes | None = None, ignore_skin: bool = True) -> Prepared:
    """The whole region for word OCR, with faces and skin-coloured areas blanked."""
    src_h, src_w = bgr.shape[:2]
    scale = min(3.0, max(1.0, TEXT_TARGET / float(max(src_w, src_h))))
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat, _ = flatten(gray)
    mask = ink_mask(flat)
    blank = np.zeros_like(mask)
    for x0, y0, x1, y1 in faces or []:
        blank[int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)] = 255
    if ignore_skin:  # sizeable skin-coloured areas (face, hands): their creases are not text
        skin = cv2.morphologyEx(skin_mask(bgr), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
        big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 0.02 * skin.size]
        if big:
            region = np.isin(labels, big).astype(np.uint8) * 255
            blank |= cv2.dilate(cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)),
                                np.ones((15, 15), np.uint8))
    if ignore_skin:  # keep only shapes that sit on paper (checked before blanking faces)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        drop = np.zeros(count, bool)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            drop[i] = area < 12 or not on_paper(flat, mask, int(x), int(y), int(w), int(h))
        mask[drop[labels]] = 0
    if blank.any():
        mask[blank > 0] = 0
        flat = flat.copy()
        flat[blank > 0] = 255
    return Prepared(gray=flat, binary=255 - mask, src_w=src_w, src_h=src_h, scale=scale,
                    found=bool(cv2.countNonZero(mask)))


def thumbnail_png(img: np.ndarray, max_side: int = 240) -> bytes:
    h, w = img.shape[:2]
    f = min(1.0, max_side / float(max(h, w)))
    if f < 1.0:
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""
