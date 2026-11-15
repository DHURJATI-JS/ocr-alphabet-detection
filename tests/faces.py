"""Face-only frames (no hands, no printed letters) for "only hands / only letters" tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).parent / "data"

# (photo, x0, y0, x1, y1) as fractions: crops that contain a face but no hand.
CROPS = {
    "portrait head": ("portrait.jpg", 0.2, 0.0, 0.8, 0.45),
    "portrait face": ("portrait.jpg", 0.33, 0.03, 0.67, 0.32),
    "portrait mouth": ("portrait.jpg", 0.38, 0.18, 0.62, 0.31),
    "business head": ("business_person.jpg", 0.2, 0.0, 0.8, 0.32),
    "business face": ("business_person.jpg", 0.33, 0.04, 0.62, 0.3),
    "business mouth": ("business_person.jpg", 0.4, 0.2, 0.6, 0.3),
    "woman face": ("woman_hands.jpg", 0.3, 0.47, 0.72, 0.68),
}
VARIANTS = {
    "": lambda x: x,
    "flipped": lambda x: cv2.flip(x, 1),
    "dark": lambda x: (x * 0.6).astype(np.uint8),
    "bright": lambda x: np.clip(x * 1.35, 0, 255).astype(np.uint8),
}


def fit(img: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    """Letterbox into a webcam-shaped frame, as the page sends it."""
    h, w = img.shape[:2]
    s = min(width / w, height / h)
    r = cv2.resize(img, (int(w * s), int(h * s)))
    out = np.zeros((height, width, 3), np.uint8)
    y, x = (height - r.shape[0]) // 2, (width - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def face_frames() -> list[tuple[str, np.ndarray]]:
    """84 frames: 7 face crops (whole head down to a mouth close-up) x 4 lighting/flip
    variants x 3 rotations."""
    frames = []
    for name, (photo, x0, y0, x1, y1) in CROPS.items():
        img = cv2.imread(str(DATA / photo))
        h, w = img.shape[:2]
        crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        for variant, change in VARIANTS.items():
            for angle in (-15, 0, 15):
                im = change(crop)
                ch, cw = im.shape[:2]
                rot = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
                im = cv2.warpAffine(im, rot, (cw, ch), borderMode=cv2.BORDER_REFLECT)
                frames.append((f"{name} {variant} {angle}".strip(), fit(im)))
    return frames


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def centre_square(img: np.ndarray, fraction: float = 0.9) -> np.ndarray:
    """The printed-letter guide box: a centred square."""
    h, w = img.shape[:2]
    s = int(min(h, w) * fraction)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]
