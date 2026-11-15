"""Train the hand-sign alphabet recognisers from real photos of several signers.

    python scripts/train_signs.py --alphabet isl             # train + save models/isl_signs.npz
    python scripts/train_signs.py --alphabet asl --cross     # also report cross-source accuracy

1. Download each source dataset from Hugging Face (cached under ./datasets).
2. Run the app's own hand tracker (same model and thresholds as live use) on every photo
   and cache the raw hand landmarks.
3. Build training poses: every photo, its mirror image (left-handed signers), and random
   variations that imitate other people and cameras - hand tilt, finger proportions,
   joint jitter. With only a few signers per dataset, this is what lets the model
   generalise to someone new.
4. --cross: train on all sources but one and test on the one left out. Those numbers say
   how well it works for people it has never seen; same-source held-out numbers do not
   (they are near-copies of training photos).
5. Train on all sources and write models/<alphabet>_signs.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLOG_minloglevel", "2")

SOURCES = {
    "isl": {
        "Hemg/Indian_sign_language_dataset": ["data/train-00000-of-00001-a1731e778755d263.parquet"],
        "akritRihal/Indian_Sign_Language_dataset": ["data/train-00000-of-00001.parquet",
                                                    "data/test-00000-of-00001.parquet"],
    },
    "asl": {
        "Marxulia/asl_sign_languages_alphabets_v03": ["data/train-00000-of-00001.parquet"],
        "suneetpaul/sign_language_image_dataset": [f"data/train-0000{i}-of-00004.parquet" for i in range(4)],
    },
}
LETTERS = set(string.ascii_uppercase)
NONE = "_"  # the "not a sign" class
# Relaxed, non-signing hands from many people (HaGRID "no_gesture", CC BY-SA 4.0). Without
# them the classifier has no notion of "not a letter" and labels a resting hand as one.
NEGATIVES = ("cj-mills/hagrid-classification-512p-no-gesture-150k",
             "data/train-00002-of-00008-08a4580cc458cb78.parquet", "6")  # label 6 = no_gesture
NEGATIVE_PHOTOS = 6000
AUGMENTED_COPIES = 3
MAX_PHOTOS_PER_LETTER = 700  # per source; keeps big single-signer sets from drowning the others

_tracker = None


def download(url: str, target: Path) -> None:
    """HTTPS download that also works with python.org's macOS Python, which ships without
    root certificates (uses certifi's bundle, a mediapipe dependency)."""
    import shutil
    import ssl

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120, context=context) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)
    tmp.replace(target)


# --------------------------------------------------------------------- landmarks
def _init_worker() -> None:
    global _tracker
    from app.config import Settings
    from app.hands.tracker import HandTracker

    _tracker = HandTracker(Settings().hand_model)


def _landmarks(args):
    """Worker: photo -> list of (label, score, image(21,3), world(21,3), aspect) per hand."""
    import cv2

    from app.hands.geometry import prominence

    index, data = args
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return index, []
    h, w = bgr.shape[:2]
    f = 640 / max(h, w)  # same size the page sends (small photos are upscaled)
    bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    hands = _tracker.detect(bgr)
    if not hands:  # tightly cropped photos: the palm detector needs some context
        pad = int(0.3 * max(bgr.shape[:2]))
        corner = [int(c) for c in bgr[0, 0]]
        hands = _tracker.detect(cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=corner))
    hands = sorted(hands, key=prominence, reverse=True)[:2]
    return index, [(hd.label, hd.score, hd.image.astype(np.float32), hd.world.astype(np.float32), hd.aspect)
                   for hd in hands]


def _read_parquets(paths: list[Path]) -> tuple[list[bytes], list[str]]:
    import pyarrow.parquet as pq

    images, labels = [], []
    for path in paths:
        table = pq.read_table(path)
        names = getattr(table.schema.field("label").type, "names", None)
        if names is None:
            meta = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}"))
            names = meta.get("info", {}).get("features", {}).get("label", {}).get("names")
        images += [img["bytes"] for img in table.column("image").to_pylist()]
        # Without label names in the file (names only on the dataset card) keep the numbers.
        labels += [str(names[i] if names else i).upper() for i in table.column("label").to_pylist()]
    return images, labels


def source_landmarks(repo: str, files: list[str], cache: Path, workers: int,
                     only_label: str | None = None, rename: str | None = None, limit: int = 0) -> dict:
    raw_slug = repo.replace("/", "__")
    slug = raw_slug + (f"__{only_label}" if only_label else "")
    out_file = cache / "landmarks" / f"{slug}.npz"
    if out_file.is_file():
        return dict(np.load(out_file, allow_pickle=False))
    paths = []
    for f in files:
        target = cache / "raw" / raw_slug / Path(f).name
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{f}"
            print(f"  downloading {url}")
            download(url, target)
        paths.append(target)
    images, labels = _read_parquets(paths)
    if only_label:
        keep = [i for i, lab in enumerate(labels) if lab == only_label.upper()]
        if limit:
            keep = sorted(np.random.default_rng(0).permutation(keep)[:limit].tolist())
        labels = [rename or lab for lab in labels]
    else:
        keep = [i for i, lab in enumerate(labels) if lab in LETTERS]
    print(f"  {repo}: {len(keep)} letter photos; finding hands with {workers} workers")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for done, (index, hands) in enumerate(pool.map(_landmarks, [(i, images[i]) for i in keep], chunksize=32), 1):
            if hands:
                rows.append((index, labels[index], hands))
            if done % 5000 == 0:
                print(f"    {done}/{len(keep)} ({time.time() - t0:.0f}s)")
    print(f"    hands found in {len(rows)}/{len(keep)} photos")
    n = len(rows)
    data = {
        "photo": np.array([r[0] for r in rows], np.int32),
        "letter": np.array([r[1] for r in rows]),
        "count": np.array([len(r[2]) for r in rows], np.int8),
        "label": np.full((n, 2), "", dtype="<U5"),
        "image": np.zeros((n, 2, 21, 3), np.float32),
        "world": np.zeros((n, 2, 21, 3), np.float32),
        "aspect": np.ones((n, 2), np.float32),
    }
    for i, (_, _, hands) in enumerate(rows):
        for j, (label, _score, image, world, aspect) in enumerate(hands):
            data["label"][i, j], data["image"][i, j], data["world"][i, j], data["aspect"][i, j] = label, image, world, aspect
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **data)
    return data


# ------------------------------------------------------------------- augmentation
FINGER_CHAINS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


def _hands_of(data: dict, i: int):
    from app.hands.geometry import Hand

    return [Hand(label=str(data["label"][i, j]), score=1.0, image=data["image"][i, j].astype(np.float64),
                 world=data["world"][i, j].astype(np.float64), aspect=float(data["aspect"][i, j]))
            for j in range(int(data["count"][i]))]


def mirror(hand):
    from app.hands.geometry import Hand

    image, world = hand.image.copy(), hand.world.copy()
    image[:, 0] = 1.0 - image[:, 0]
    world[:, 0] = -world[:, 0]
    return Hand({"Left": "Right", "Right": "Left"}.get(hand.label, hand.label), hand.score, image, world, hand.aspect)


def vary(hands, rng):
    """Same sign, as if by another person / camera: tilt, finger proportions, jitter."""
    from app.hands.geometry import Hand

    roll = np.radians(rng.uniform(-20, 20))
    rot = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    aspect = hands[0].aspect
    scale_xy = np.array([aspect, 1.0])
    centre = np.concatenate([hd.image[:, :2] * scale_xy for hd in hands]).mean(axis=0)
    out = []
    for hd in hands:
        world = hd.world.copy()
        image = hd.image.copy()
        xy = image[:, :2] * scale_xy
        for chain in FINGER_CHAINS:  # longer / shorter fingers
            s = 1.0 + rng.normal(0, 0.07)
            base = chain[0]
            world[list(chain[1:])] = world[base] + (world[list(chain[1:])] - world[base]) * s
            xy[list(chain[1:])] = xy[base] + (xy[list(chain[1:])] - xy[base]) * s
        world += rng.normal(0, 0.0025, world.shape)  # ~3% of a palm
        xy += rng.normal(0, 0.003, xy.shape)
        world[:, :2] = world[:, :2] @ rot.T  # tilt about the camera axis
        xy = (xy - centre) @ rot.T + centre
        image[:, :2] = xy / scale_xy
        out.append(Hand(hd.label, hd.score, image, world, hd.aspect))
    return out


def build(data: dict, rows: np.ndarray, copies: int, seed: int) -> dict:
    """Pose features for the given photos (+ mirrored + `copies` random variations)."""
    from app.hands.geometry import analyze, frame_features, order_hands

    rng = np.random.default_rng(seed)
    X = {1: [], 2: []}
    y = {1: [], 2: []}
    for i in rows:
        base = _hands_of(data, i)
        versions = [base, [mirror(h) for h in base]]
        for _ in range(copies):
            versions.append(vary(versions[int(rng.integers(0, 2))], rng))
        for hs in versions:
            n, vec = frame_features(order_hands([analyze(h) for h in hs]))
            X[n].append(vec)
            y[n].append(data["letter"][i])
    return {n: (np.array(X[n], np.float32), np.array(y[n])) for n in (1, 2) if X[n]}


def build_negatives(data: dict, rows: np.ndarray, copies: int, seed: int) -> dict:
    """'Not a sign' poses: single relaxed hands, and pairs of them side by side."""
    from app.hands.geometry import Hand, analyze, frame_features, order_hands

    rng = np.random.default_rng(seed)
    singles = build(data, rows, copies, seed)
    X2, y2 = [], []
    for _ in range(len(rows)):
        a, b = (_hands_of(data, int(i))[0] for i in rng.choice(rows, 2, replace=False))
        b = mirror(b)  # one left and one right hand
        pair = []
        for hand, x in ((a, rng.uniform(0.2, 0.4)), (b, rng.uniform(0.6, 0.8))):
            image = hand.image.copy()
            centre = image[:, :2].mean(axis=0)
            scale = rng.uniform(0.6, 1.2) * 0.25 / max(np.ptp(image[:, 1]), 1e-3)
            image[:, :2] = (image[:, :2] - centre) * scale + np.array([x, rng.uniform(0.45, 0.8)])
            pair.append(Hand(hand.label, hand.score, image, hand.world, hand.aspect))
        n, vec = frame_features(order_hands([analyze(h) for h in pair]))
        X2.append(vec)
        y2.append(NONE)
    singles[2] = (np.array(X2, np.float32), np.array(y2))
    return singles


def cap_per_letter(data: dict, limit: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for letter in np.unique(data["letter"]):
        idx = np.flatnonzero(data["letter"] == letter)
        keep += rng.permutation(idx)[:limit].tolist()
    return np.array(sorted(keep))


def merge(parts: list[dict]) -> dict:
    out: dict = {}
    for part in parts:
        for n, (X, y) in part.items():
            if n in out:
                out[n] = (np.vstack([out[n][0], X]), np.concatenate([out[n][1], y]))
            else:
                out[n] = (X, y)
    return out


# ------------------------------------------------------------------- evaluation
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)


def score(model, data: dict) -> dict:
    """Run the app's prediction on every photo of a source (no augmentation)."""
    from app.hands.geometry import analyze, order_hands

    outcomes = []  # (truth, letter, probability)
    for i in range(len(data["letter"])):
        truth = str(data["letter"][i])
        if truth not in model.letters:
            continue
        letter, p, _ = model.predict(order_hands([analyze(h) for h in _hands_of(data, i)]))
        outcomes.append((truth, letter, p))
    n = max(len(outcomes), 1)
    table = []
    for t in THRESHOLDS:  # what would be typed if only readings with p >= t count
        typed = [(tr, le) for tr, le, p in outcomes if le and p >= t]
        right = sum(1 for tr, le in typed if tr == le)
        table.append((t, right / n, (len(typed) - right) / n, 1 - len(typed) / n))
    confusions: dict[str, int] = {}
    for tr, le, _ in outcomes:
        if le and le != tr:
            confusions[f"{tr}->{le}"] = confusions.get(f"{tr}->{le}", 0) + 1
    top = ", ".join(f"{k}x{v}" for k, v in sorted(confusions.items(), key=lambda kv: -kv[1])[:6])
    per: dict[str, list[int]] = {}
    for tr, le, _ in outcomes:
        per.setdefault(tr, [0, 0])[0 if le == tr else 1] += 1
    worst = ", ".join(f"{k} {v[0] / sum(v):.0%}" for k, v in sorted(per.items(), key=lambda kv: kv[1][0] / sum(kv[1]))[:8])
    return {"n": n, "correct": table[0][1], "wrong": table[0][2], "rejected": table[0][3], "top": top,
            "worst": worst, "table": table}


def print_score(r: dict) -> None:
    print(f"    correct {r['correct']:.1%}   wrong letter {r['wrong']:.1%}   nothing {r['rejected']:.1%}")
    print(f"    worst letters: {r['worst']}")
    print(f"    top confusions: {r['top']}")
    print("    if only readings with confidence >= t are typed:")
    print("        t     typed right   typed WRONG   not typed")
    for t, right, wrong, none in r["table"]:
        print(f"      {t:4.2f}      {right:6.1%}        {wrong:6.1%}       {none:6.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphabet", choices=sorted(SOURCES), required=True)
    ap.add_argument("--cache", default=str(ROOT / "datasets"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--copies", type=int, default=AUGMENTED_COPIES, help="random variations per photo")
    ap.add_argument("--cross", action="store_true", help="report leave-one-source-out accuracy")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    from app.hands.geometry import analyze, order_hands
    from app.hands.trained import fit

    cache = Path(args.cache)
    sources = {repo: source_landmarks(repo, files, cache, args.workers) for repo, files in SOURCES[args.alphabet].items()}
    neg_repo, neg_file, neg_label = NEGATIVES
    negatives = source_landmarks(neg_repo, [neg_file], cache, args.workers, only_label=neg_label, rename=NONE,
                                 limit=NEGATIVE_PHOTOS)
    neg_rows = np.random.default_rng(1).permutation(len(negatives["letter"]))
    neg_test, neg_train = neg_rows[:800], neg_rows[800:]
    print("building training poses ...")
    built = {repo: build(d, cap_per_letter(d, MAX_PHOTOS_PER_LETTER), args.copies, seed=k)
             for k, (repo, d) in enumerate(sources.items())}
    built["negatives"] = build_negatives(negatives, neg_train, 1, seed=99)

    report = {}
    if args.cross:
        for held_out, d in sources.items():
            model = fit(merge([b for repo, b in built.items() if repo != held_out]))
            r = score(model, d)
            report[held_out] = r
            print(f"\n  trained without {held_out}, tested on it ({r['n']} photos, people never seen):")
            print_score(r)
    if args.no_save:
        return 0
    model = fit(merge(list(built.values())))
    typed = sum(1 for i in neg_test if model.predict(order_hands([analyze(h) for h in _hands_of(negatives, int(i))]))[0])
    print(f"\nrelaxed, non-signing hands (held out, {len(neg_test)} photos): {typed / len(neg_test):.1%} would type a letter")
    out = ROOT / "models" / f"{args.alphabet}_signs.npz"
    model.save(out, meta={"alphabet": args.alphabet, "sources": list(sources), "augmented_copies": args.copies,
                          "cross_source": {k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in report.items()}})
    print(f"\nsaved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB), letters: {''.join(model.letters)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
