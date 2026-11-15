"""A small forward-kinematics hand that produces MediaPipe-style landmarks for ASL letters.

Used to test the geometry and the letter templates end to end (all orientations, noise,
left and right hands) without needing photos of every sign. Real-hand behaviour is checked
separately against photos in tests/data/.
"""

from __future__ import annotations

import math

import numpy as np

from app.hands.geometry import FINGERS, Hand

# Palm-frame coordinates (a: towards the thumb side, u: towards the fingers, n: palm side).
MCP = {"index": (0.30, 0.95, 0.0), "middle": (0.0, 1.0, 0.0), "ring": (-0.25, 0.93, 0.0), "pinky": (-0.48, 0.82, 0.0)}
SEGMENTS = {"index": (0.45, 0.27, 0.22), "middle": (0.50, 0.30, 0.23), "ring": (0.47, 0.29, 0.22),
            "pinky": (0.37, 0.21, 0.20)}
THUMB_CMC = np.array([0.30, 0.25, 0.10])

EXT = (0, 3, 5)  # (mcp, pip, dip) flexion, degrees
CURL = (80, 100, 60)
BENT = (35, 55, 25)
HOOK = (10, 95, 55)


def finger_points(name: str, abduction: float, flex: tuple[float, float, float]) -> np.ndarray:
    a, u, n = np.eye(3)
    ab = math.radians(abduction)
    d0 = math.sin(ab) * a + math.cos(ab) * u
    pts = [np.array(MCP[name], dtype=float)]
    theta = 0.0
    for angle, length in zip(flex, SEGMENTS[name]):
        theta += math.radians(angle)
        pts.append(pts[-1] + length * (math.cos(theta) * d0 + math.sin(theta) * n))
    return np.array(pts)


def thumb_points(tip: np.ndarray) -> np.ndarray:
    tip = np.asarray(tip, dtype=float)
    span = tip - THUMB_CMC
    bulge = np.array([0.06, -0.04, -0.03])
    mcp = THUMB_CMC + 0.42 * span + bulge
    ip = THUMB_CMC + 0.74 * span + bulge * 0.6
    return np.array([THUMB_CMC, mcp, ip, tip])


# letter -> (fingers {name: (abduction, flex)}, thumb target or ("touch", finger) / ("near", fingers))
def _fist(**override):
    base = {"index": (0, CURL), "middle": (0, CURL), "ring": (0, CURL), "pinky": (0, CURL)}
    base.update(override)
    return base


LETTERS = {
    "A": (_fist(), (0.45, 0.80, 0.10)),
    "B": ({"index": (2, EXT), "middle": (0, EXT), "ring": (-2, EXT), "pinky": (-4, EXT)}, (0.0, 0.55, 0.15)),
    "C": ({"index": (0, BENT), "middle": (0, BENT), "ring": (0, BENT), "pinky": (0, BENT)}, (0.40, 0.65, 0.60)),
    "D": (_fist(index=(0, EXT)), ("touch", "middle")),
    "E": (_fist(index=(0, (60, 100, 70)), middle=(0, (60, 100, 70)), ring=(0, (60, 100, 70)),
                pinky=(0, (60, 100, 70))), ("near", ("index", "middle"))),
    "F": ({"index": (0, (50, 80, 50)), "middle": (0, EXT), "ring": (-4, EXT), "pinky": (-8, EXT)},
          ("touch", "index")),
    "G": (_fist(index=(0, EXT)), (0.50, 0.95, 0.15)),
    "H": (_fist(index=(2, EXT), middle=(-2, EXT)), (-0.10, 0.80, 0.50)),
    "I": (_fist(pinky=(-5, EXT)), (-0.10, 0.80, 0.55)),
    "K": (_fist(index=(10, EXT), middle=(-10, EXT)), (0.05, 1.30, 0.12)),
    "L": (_fist(index=(0, EXT)), (1.00, 0.45, 0.05)),
    "M": (_fist(), (-0.37, 0.90, 0.30)),
    "N": (_fist(), (-0.12, 0.95, 0.30)),
    "O": ({"index": (0, (40, 60, 30)), "middle": (0, (40, 60, 30)), "ring": (0, (40, 60, 30)),
           "pinky": (0, (40, 60, 30))}, ("near", ("index", "middle"))),
    "P": (_fist(index=(10, EXT), middle=(-10, EXT)), (0.05, 1.30, 0.12)),
    "Q": (_fist(index=(0, EXT)), (0.50, 0.95, 0.15)),
    "R": (_fist(index=(-12, EXT), middle=(12, EXT)), (-0.20, 0.85, 0.55)),
    "S": (_fist(), (0.0, 0.80, 0.70)),
    "T": (_fist(), (0.15, 1.00, 0.35)),
    "U": (_fist(index=(2, EXT), middle=(-2, EXT)), (-0.25, 0.85, 0.55)),
    "V": (_fist(index=(10, EXT), middle=(-10, EXT)), (-0.25, 0.85, 0.55)),
    "W": (_fist(index=(12, EXT), middle=(0, EXT), ring=(-12, EXT)), (-0.35, 0.80, 0.50)),
    "X": (_fist(index=(0, HOOK)), (-0.05, 0.85, 0.55)),
    "Y": (_fist(pinky=(-15, EXT)), (1.00, 0.45, 0.05)),
}
ORIENTATION = {"G": "side", "H": "side", "P": "down", "Q": "down"}
# Open hand, fingers spread (the passive hand of two-handed alphabets).
OPEN_SPREAD = ({"index": (12, EXT), "middle": (3, EXT), "ring": (-6, EXT), "pinky": (-15, EXT)}, (0.95, 0.55, 0.10))
RELAXED_OPEN = ({"index": (6, (25, 25, 15)), "middle": (0, (20, 25, 15)), "ring": (-6, (20, 25, 15)),
                 "pinky": (-12, (15, 25, 15))}, (0.75, 0.75, 0.25))


def palm_pose(spec) -> np.ndarray:
    fingers, thumb = spec
    pts = np.zeros((21, 3))
    for name, ids in FINGERS.items():
        abd, flex = fingers[name]
        pts[list(ids)] = finger_points(name, abd, flex)
    if isinstance(thumb[0], str):
        kind, target = thumb
        names = (target,) if isinstance(target, str) else target
        tips = np.mean([pts[FINGERS[n][3]] for n in names], axis=0)
        tip = tips + (np.array([0.0, -0.05, 0.02]) if kind == "touch" else np.array([0.0, -0.12, -0.08]))
    else:
        tip = np.array(thumb)
    pts[1:5] = thumb_points(tip)
    return pts


def _rotation(kind: str) -> np.ndarray:
    a_cam = np.array([1.0, 0.0, 0.0])
    if kind == "up":
        u_cam = np.array([0.0, -1.0, 0.0])
    elif kind == "side":
        a_cam, u_cam = np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    elif kind == "down":
        u_cam = np.array([0.0, 0.85, -0.53])
        u_cam /= np.linalg.norm(u_cam)
    else:
        raise ValueError(kind)
    n_cam = np.cross(a_cam, u_cam)
    return np.stack([a_cam, u_cam, n_cam], axis=1)  # columns: palm axes in camera coords


def _jitter(rng: np.random.Generator, degrees: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(-degrees, degrees))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def make_hand(spec, orientation: str = "up", seed: int = 0, noise: float = 0.0, tilt: float = 0.0,
              left: bool = False, center=(0.5, 0.55), palm_img: float = 0.18) -> Hand:
    """Landmarks of a pose as MediaPipe would report them for an un-mirrored webcam frame."""
    rng = np.random.default_rng(seed)
    pts = palm_pose(spec)
    if noise:
        pts = pts + rng.normal(0, noise, pts.shape)
    rot = _rotation(orientation) @ np.eye(3)
    if tilt:
        rot = _jitter(rng, tilt) @ rot
    cam = pts @ rot.T  # palm frame -> camera axes (x right, y down)
    label = "Right"
    if left:  # a left hand is the mirror image
        cam = cam * np.array([-1.0, 1.0, 1.0])
        label = "Left"
    world = cam * 0.085
    image = np.empty_like(cam)
    image[:, 0] = center[0] + cam[:, 0] * palm_img
    image[:, 1] = center[1] + cam[:, 1] * palm_img
    image[:, 2] = cam[:, 2] * palm_img
    return Hand(label=label, score=0.98, image=image, world=world, aspect=1.0)


def letter_hand(letter: str, **kw) -> Hand:
    return make_hand(LETTERS[letter], ORIENTATION.get(letter, "up"), **kw)


def touching(active_spec, orientation, passive: Hand, landmark: int, seed: int = 0, noise: float = 0.0,
             palm_img: float = 0.18) -> Hand:
    """A right hand placed so its index fingertip touches `landmark` of the passive hand."""
    probe = make_hand(active_spec, orientation, seed=seed, noise=noise, center=(0.0, 0.0), palm_img=palm_img)
    target = passive.image[landmark, :2]
    center = target - probe.image[8, :2]
    return make_hand(active_spec, orientation, seed=seed, noise=noise, center=tuple(center), palm_img=palm_img)


def two_hand_vowel(tip: int, seed: int = 0, noise: float = 0.0, tilt: float = 0.0, vary_size: bool = False) -> list[Hand]:
    """Two-handed-alphabet vowel: the right index touches one fingertip (4/8/12/16/20 =
    thumb, index, middle, ring, little) of the open left hand. tilt / vary_size make the
    hands' relative position unreliable while the contact point stays the same."""
    rng = np.random.default_rng(seed)
    size = (lambda: float(rng.uniform(0.14, 0.22))) if vary_size else (lambda: 0.18)
    passive = make_hand(OPEN_SPREAD, "up", seed=seed, noise=noise, tilt=tilt, left=True, center=(0.36, 0.62),
                        palm_img=size())
    active = touching(LETTERS["D"], "side", passive, tip, seed=seed + 100, noise=noise, palm_img=size())
    return [passive, active]
