"""Measure a trained sign model on photos it has never seen, from other sources/people.

    python scripts/eval_signs.py --alphabet asl --parquet datasets/asl_ind/*.parquet
    python scripts/eval_signs.py --alphabet asl --folder  datasets/asl_phone   # <letter>/<photo>.jpg

Runs exactly the app's live pipeline per photo: downscale to 640 px, MediaPipe hands
(same thresholds), pose features, trained model with its "not a known sign" rejection.
"""

from __future__ import annotations

import argparse
import glob
import os
import string
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLOG_minloglevel", "2")

_state: dict = {}


def _init(alphabet: str) -> None:
    from app.config import Settings
    from app.hands.trained import TrainedAlphabet
    from app.hands.tracker import HandTracker

    _state["tracker"] = HandTracker(Settings().hand_model)
    _state["model"] = TrainedAlphabet.load(ROOT / "models" / f"{alphabet}_signs.npz")


def _predict(item):
    import cv2

    from app.hands.geometry import analyze, order_hands, prominence

    label, data = item
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return label, None, 0.0
    h, w = bgr.shape[:2]
    f = 640 / max(h, w)  # what the page sends
    bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    hands = sorted(_state["tracker"].detect(bgr), key=prominence, reverse=True)[:2]
    if not hands:
        return label, None, 0.0
    letter, p, _ = _state["model"].predict(order_hands([analyze(hd) for hd in hands]))
    return label, letter, p


def load_items(args) -> list[tuple[str, bytes]]:
    items = []
    if args.parquet:
        import pyarrow.parquet as pq

        for path in sorted(p for pattern in args.parquet for p in glob.glob(pattern)):
            table = pq.read_table(path)
            names = table.schema.field("label").type
            label_names = getattr(names, "names", None)
            if label_names is None:
                import json

                meta = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}"))
                label_names = meta["info"]["features"]["label"]["names"]
            for img, lab in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
                items.append((str(label_names[lab]).upper(), img["bytes"]))
    if args.folder:
        for path in sorted(Path(args.folder).glob("*/*")):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png") and len(path.parent.name) == 1:
                items.append((path.parent.name.upper(), path.read_bytes()))
    items = [it for it in items if it[0] in string.ascii_uppercase]
    if args.per_letter:
        rng = np.random.default_rng(0)
        by_letter: dict[str, list] = {}
        for it in items:
            by_letter.setdefault(it[0], []).append(it)
        items = [it for group in by_letter.values()
                 for it in [group[i] for i in rng.permutation(len(group))[: args.per_letter]]]
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphabet", choices=["asl", "isl"], required=True)
    ap.add_argument("--parquet", nargs="*")
    ap.add_argument("--folder")
    ap.add_argument("--per-letter", type=int, default=0, help="sample at most N photos per letter")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    from app.hands.trained import TrainedAlphabet

    known = set(TrainedAlphabet.load(ROOT / "models" / f"{args.alphabet}_signs.npz").letters)
    items = [it for it in load_items(args) if it[0] in known]
    print(f"{len(items)} photos, letters: {''.join(sorted({it[0] for it in items}))}")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.alphabet,)) as pool:
        results = list(pool.map(_predict, items, chunksize=8))

    no_hand = sum(1 for _, guess, _ in results if guess is None)
    scored = [(truth, guess) for truth, guess, _ in results if guess is not None]
    correct = sum(1 for t, g in scored if g == t)
    rejected = sum(1 for t, g in scored if g == "")
    wrong = len(scored) - correct - rejected
    n = max(len(scored), 1)
    per = {}
    for t, g in scored:
        per.setdefault(t, [0, 0])[0 if g == t else 1] += 1
    conf = Counter(f"{t}->{g}" for t, g in scored if g and g != t)
    print(f"hand found in {len(scored)}/{len(results)} photos")
    print(f"  correct  {correct:5d} ({correct / n:.1%})")
    print(f"  wrong    {wrong:5d} ({wrong / n:.1%})   <- would type a wrong letter")
    print(f"  rejected {rejected:5d} ({rejected / n:.1%})   <- types nothing (unsure)")
    print(f"  accuracy of what it types: {correct / max(correct + wrong, 1):.1%}")
    worst = sorted(per.items(), key=lambda kv: kv[1][0] / sum(kv[1]))[:10]
    print("  worst letters: " + ", ".join(f"{k} {v[0] / sum(v):.0%}" for k, v in worst))
    print("  top confusions: " + ", ".join(f"{k}x{v}" for k, v in conf.most_common(10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
