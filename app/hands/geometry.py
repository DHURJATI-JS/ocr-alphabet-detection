"""Hand geometry from MediaPipe's 21 landmarks: a palm-aligned frame and finger states.

Landmarks: 0 wrist; thumb 1-4 (CMC, MCP, IP, tip); index 5-8; middle 9-12; ring 13-16;
pinky 17-20 (MCP, PIP, DIP, tip).

Palm frame (all coordinates in units of the palm size = |wrist -> middle MCP|):
  a  across the palm, from the pinky side towards the index/thumb side
  u  up the palm, wrist -> middle-finger knuckle
  n  out of the palm on the palm side (where curled fingers go)
Expressed this way a sign looks the same wherever the hand is in the frame and however far
from the camera it is. Absolute orientation (pointing up vs sideways, which separates e.g.
ASL U from H) is kept separately from the image coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
TIPS = (4, 8, 12, 16, 20)
# Skeleton edges, for drawing.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Which way is the palm when no finger is bent enough to tell? Depends on the handedness
# label MediaPipe gives (it assumes a mirrored selfie image; webcam frames are not mirrored).
# Calibrated on real photos: with n = a x u, curled fingertips lie on +n for hands labelled
# "Right" and on -n for "Left" (see tests/test_hands.py::test_palm_side_matches_label).
LABEL_PALM_SIGN = {"Right": 1.0, "Left": -1.0}


@dataclass
class Hand:
    """One detected hand."""

    label: str  # MediaPipe handedness label ("Left"/"Right")
    score: float
    image: np.ndarray  # (21, 3) normalised image coords: x, y in 0..1 (y down), z relative depth
    world: np.ndarray  # (21, 3) metres, camera-aligned axes (x right, y down)
    aspect: float = 1.0  # image width / height, to measure image directions without distortion


@dataclass
class Finger:
    mcp: float  # flexion at the knuckle, degrees (towards the palm positive)
    pip: float  # flexion at the middle joint
    dip: float  # flexion at the last joint

    @property
    def total(self) -> float:
        return self.mcp + self.pip + self.dip


@dataclass
class HandPose:
    hand: Hand
    p: np.ndarray  # (21, 3) palm-frame coordinates (a, u, n)
    fingers: dict[str, Finger]
    up_img: np.ndarray  # unit 2D image direction wrist -> middle knuckle (x right, y down)
    index_img: np.ndarray  # unit 2D image direction index knuckle -> index tip
    palm_img: float  # palm size in image units (fraction of image height)
    palm_sign_from: str  # "fingers" or "label": how the palm side was decided
    basis: np.ndarray | None = None  # rows a, u, n in camera axes: the hand's 3D orientation

    def dist(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.p[i] - self.p[j]))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    c = float(np.dot(_unit(v1), _unit(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def palm_frame(world: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (palm-frame coords (21,3), basis rows [a, u, n] in camera coords, sign source)."""
    w = world - world[WRIST]
    size = float(np.linalg.norm(w[9])) or 1.0
    u = _unit(w[9])
    across = w[5] - w[17]
    a = _unit(across - np.dot(across, u) * u)
    n = np.cross(a, u)
    basis = np.stack([a, u, n])
    p = (w @ basis.T) / size

    # Palm side: bent fingers move their tips towards the palm. Use that when fingers are
    # bent enough to tell; otherwise fall back to the handedness label.
    evidence = 0.0
    for ids in FINGERS.values():
        mcp, tip = p[ids[0]], p[ids[3]]
        straightness = (tip[1] - mcp[1]) / 0.9  # ~1 when straight up the palm
        evidence += (tip[2] - mcp[2]) * max(0.0, 1.0 - straightness)
    if abs(evidence) > 0.15:
        sign, source = (1.0 if evidence > 0 else -1.0), "fingers"
    else:
        sign, source = LABEL_PALM_SIGN.get(label, 1.0), "label"
    p[:, 2] *= sign
    basis[2] *= sign
    return p, basis, source


def finger_state(p: np.ndarray, ids: tuple[int, int, int, int]) -> Finger:
    m, j, d, t = p[list(ids)]
    v1, v2, v3 = j - m, d - j, t - d
    mcp = math.degrees(math.atan2(v1[2], v1[1]))  # 0 = straight up the palm, 90 = folded over
    return Finger(mcp=mcp, pip=_angle(v1, v2), dip=_angle(v2, v3))


def analyze(hand: Hand) -> HandPose:
    p, basis, source = palm_frame(hand.world, hand.label)
    fingers = {name: finger_state(p, ids) for name, ids in FINGERS.items()}
    xy = hand.image[:, :2] * np.array([hand.aspect, 1.0])
    return HandPose(
        hand=hand,
        p=p,
        fingers=fingers,
        up_img=_unit(xy[9] - xy[0]),
        index_img=_unit(xy[8] - xy[5]),
        palm_img=float(np.linalg.norm(xy[9] - xy[0])),
        palm_sign_from=source,
        basis=basis,
    )


# --------------------------------------------------------------------------- features
ORIENTATION_WEIGHT = 1.5


def pose_features(pose: HandPose) -> np.ndarray:
    """Feature vector of one hand: its shape in palm coordinates, thumb contacts, and its
    orientation - both in the image and in 3D (which way the palm faces). Signs such as
    ASL U / H or G / Q have the same shape and differ only in orientation; with real,
    tilted hands the palm's facing direction is what tells them apart."""
    shape = pose.p[1:].ravel()  # 20 landmarks x (a, u, n), wrist is the origin
    tips = pose.p[list(TIPS)]
    contacts = np.array([np.linalg.norm(tips[0] - tips[i]) for i in range(1, 5)])  # thumb to fingertips
    orientation = np.concatenate([pose.up_img, pose.index_img]) * ORIENTATION_WEIGHT
    facing = (pose.basis.ravel() if pose.basis is not None else np.zeros(9)) * ORIENTATION_WEIGHT
    return np.concatenate([shape, contacts * 2.0, orientation, facing]).astype(np.float32)


CONTACT_RANGE = 0.6  # palm widths: fingertips closer than about this count as touching
CONTACT_WEIGHT = 2.0


def two_hand_features(a: HandPose, b: HandPose) -> np.ndarray:
    """How two hands relate: wrist offset, plus closeness of every fingertip of one hand to
    every fingertip of the other. Two-handed alphabets hinge on such contacts, e.g. in ISL
    the vowels are the index finger touching a different fingertip of the other hand."""
    scale = max((a.palm_img + b.palm_img) / 2, 1e-3)
    to_xy = np.array([a.hand.aspect, 1.0])
    offset = (b.hand.image[0, :2] - a.hand.image[0, :2]) * to_xy / scale
    tips_a = a.hand.image[list(TIPS), :2] * to_xy
    tips_b = b.hand.image[list(TIPS), :2] * to_xy
    dist = np.linalg.norm(tips_a[:, None, :] - tips_b[None, :, :], axis=-1) / scale  # 5 x 5
    contact = np.exp(-((dist / CONTACT_RANGE) ** 2))  # 1 = touching, ~0 = apart
    return np.concatenate([np.clip(offset, -6, 6), contact.ravel() * CONTACT_WEIGHT]).astype(np.float32)


def order_hands(poses: list[HandPose]) -> list[HandPose]:
    """Left-to-right in the image: stable for two-handed signs, where hands keep their side."""
    return sorted(poses, key=lambda ps: ps.hand.image[0, 0])


def frame_features(poses: list[HandPose]) -> tuple[int, np.ndarray]:
    """(number of hands, features) of a frame, for matching taught signs. With two hands
    both shapes and how they relate are included, so two-handed alphabets can be taught."""
    poses = order_hands(poses)
    parts = [pose_features(ps) for ps in poses]
    if len(poses) == 2:
        parts.append(two_hand_features(*poses))
    return len(poses), np.concatenate(parts)


def prominence(hand: Hand) -> float:
    """Bigger (closer) and more confidently detected hands first."""
    xy = hand.image[:, :2] * np.array([hand.aspect, 1.0])
    return float(np.linalg.norm(xy[9] - xy[0])) * hand.score


def body_side(label: str) -> str:
    """The signer's actual hand. MediaPipe labels assume a mirrored selfie image; webcam
    frames sent by the page are not mirrored, so its labels are swapped."""
    return {"Left": "right", "Right": "left"}.get(label, "")
