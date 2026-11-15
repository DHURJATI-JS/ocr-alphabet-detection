"""Download the model files the app needs (cross-platform, idempotent).

    python scripts/download_models.py

* models/hand_landmarker.task  - MediaPipe hand landmarks (hand-sign mode), ~8 MB
* models/blaze_face_short_range.tflite - MediaPipe face detector, so OCR ignores faces, ~0.2 MB
* tessdata/eng.traineddata     - Tesseract English model (printed-letter OCR), ~23 MB.
  The *standard* model is used on purpose: it includes the legacy character classifier,
  which reads isolated single letters well (tessdata_fast / tessdata_best are LSTM-only).
"""

from __future__ import annotations

import ssl
import sys
import urllib.request
from pathlib import Path

try:  # python.org's macOS Python ships without root certificates; certifi has them
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    (ROOT / "models" / "hand_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
    (ROOT / "models" / "blaze_face_short_range.tflite",
     "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"),
    (ROOT / "tessdata" / "eng.traineddata",
     "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"),
]


def fetch(target: Path, url: str) -> None:
    if target.is_file() and target.stat().st_size > 100_000:
        print(f"already present: {target.relative_to(ROOT)}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(target)
    print(f"saved {target.relative_to(ROOT)}")


def main() -> int:
    for target, url in FILES:
        fetch(target, url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
