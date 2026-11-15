"""Measure per-engine letter accuracy and latency on synthetic webcam-like frames.

    python scripts/benchmark.py            # A-Z, all fonts, clean + degraded
    python scripts/benchmark.py --quick    # fewer samples
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DetectionOptions, Settings  # noqa: E402
from app.ocr.manager import OCRManager  # noqa: E402
from tests.synthetic import available_fonts, render  # noqa: E402

CONDITIONS = {
    "clean": dict(),
    "webcam": dict(noise=8, blur=1.2, gradient=35, rotate=4),
    "rough": dict(noise=14, blur=2.0, gradient=55, rotate=-8, clutter=True, text_px=120),
    "inverted": dict(noise=8, blur=1.0, inverted=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--case", default="upper", choices=["upper", "lower"])
    args = ap.parse_args()

    manager = OCRManager.from_settings(Settings.from_env())
    fonts = available_fonts() or [None]
    if args.quick:
        fonts = fonts[:3]
    letters = string.ascii_uppercase if args.case == "upper" else string.ascii_lowercase

    print("engines:", {k: (v or {}).get("detail") or (v or {}).get("reason") for k, v in manager.status().items()})
    for strategy in ("primary", "secondary", "fallback", "consensus"):
        opts = DetectionOptions(mode="letter", strategy=strategy, case=args.case)
        print(f"\n== strategy: {strategy}")
        for cond, kw in CONDITIONS.items():
            ok = total = missed = 0
            wrong: Counter = Counter()
            lat = []
            for fi, font in enumerate(fonts):
                for li, ch in enumerate(letters):
                    data = render(ch, font=font, seed=fi * 100 + li, **kw)
                    t = time.perf_counter()
                    res = manager.process(data, opts)
                    lat.append((time.perf_counter() - t) * 1000)
                    total += 1
                    if res.text == ch:
                        ok += 1
                    elif not res.text:
                        missed += 1
                    else:
                        wrong[f"{ch}->{res.text}"] += 1
            lat.sort()
            print(
                f"  {cond:9s} correct {ok:3d}/{total} ({ok / total:5.1%})  no-read {missed:3d}  "
                f"wrong {sum(wrong.values()):3d}  p50 {lat[len(lat) // 2]:5.1f}ms  p95 {lat[int(len(lat) * .95)]:5.1f}ms"
                + (f"  top errors: {', '.join(f'{k}×{v}' for k, v in wrong.most_common(5))}" if wrong else "")
            )
    manager.close()


if __name__ == "__main__":
    main()
