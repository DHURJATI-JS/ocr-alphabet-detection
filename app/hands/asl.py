"""Built-in recogniser for the static ASL (American Sign Language) fingerspelling letters.

Every letter is a template of fuzzy conditions on the hand pose ("index straight", "thumb
across the palm", "pointing sideways" ...). Each condition scores 0..1, a letter's score is
the geometric mean of its conditions, and the best letter wins if it clearly beats the
runner-up. J and Z involve movement and are handled in motion.py.

Thresholds come from MediaPipe landmarks of real hands: curled fingers bend 140-240 degrees
in total over their three joints, straight ones under ~40, a relaxed open hand ~65-95.

Rules can only go so far: A/E/M/N/S/T differ only in where a mostly hidden thumb sits.
For a reliable result on a particular person and camera, teach the signs (library.py).
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import HandPose

ACCEPT_SCORE = 0.55


def up(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear in between."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def down(x: float, lo: float, hi: float) -> float:
    return 1.0 - up(x, lo, hi)


def band(x: float, lo: float, hi: float, soft: float) -> float:
    return min(up(x, lo - soft, lo), down(x, hi, hi + soft))


class Features:
    """Fuzzy facts about one hand pose, computed once and shared by all templates."""

    def __init__(self, pose: HandPose) -> None:
        p, f = pose.p, pose.fingers
        self.pose = pose
        self.ext = {k: down(v.total, 45, 80) * down(v.pip, 40, 60) for k, v in f.items()}
        self.curl = {k: up(v.total, 110, 150) for k, v in f.items()}
        self.bent = {k: band(v.total, 65, 125, 25) for k, v in f.items()}
        self.hook = {k: down(v.mcp, 35, 55) * up(v.pip + v.dip, 90, 130) for k, v in f.items()}
        self.folded = {k: max(self.curl[k], self.bent[k] * 0.8) for k in f}  # not straight
        self.cupped = min(band(v.total, 95, 150, 15) * down(v.pip, 95, 115) for v in f.values())

        thumb, idx_mcp = p[4], p[5]
        self.thumb_tip = thumb
        self.t_out = float(thumb[0] - idx_mcp[0])  # >0: beyond the index side of the palm
        thumb_dir = p[4] - p[2]
        self.thumb_angle = _deg_between(thumb_dir[:2], np.array([0.0, 1.0]))  # vs up-the-palm, in palm plane
        self.thumb_out = up(self.t_out, 0.25, 0.5) * up(self.thumb_angle, 30, 50)
        self.thumb_side = band(self.t_out, -0.05, 0.45, 0.2) * down(self.thumb_angle, 45, 70)
        self.thumb_across = down(self.t_out, -0.05, 0.1)
        # Where along the knuckle line the thumb tip sits (for T / N / M).
        a = float(thumb[0])
        mcp_a = {k: float(p[ids][0]) for k, ids in (("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17))}
        self.slot_t = band(a, mcp_a["middle"], mcp_a["index"], 0.08)
        self.slot_n = band(a, mcp_a["ring"], mcp_a["middle"], 0.08)
        self.slot_m = band(a, mcp_a["pinky"], mcp_a["ring"], 0.08)
        # Thumb in front of the folded fingers (S) or tucked between / under them (T, N, M).
        knuckle_n = float(np.mean([p[6][2], p[10][2], p[14][2]]))
        self.thumb_front = up(float(thumb[2]) - knuckle_n, -0.05, 0.12)
        self.thumb_under = down(float(thumb[2]) - knuckle_n, -0.05, 0.12)

        self.touch = {name: down(pose.dist(4, tip), 0.25, 0.5)
                      for name, tip in (("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20))}
        self.thumb_mid_joint = down(min(pose.dist(4, 10), pose.dist(4, 11), pose.dist(4, 9)), 0.3, 0.5)
        spread = _deg_between((p[8] - p[5])[:2], (p[12] - p[9])[:2])
        self.together = down(spread, 9, 16)
        self.apart = up(spread, 12, 20)
        self.apart3 = up(_deg_between((p[8] - p[5])[:2], (p[16] - p[13])[:2]), 14, 24)  # index vs ring (W)
        # How far the fingertips stand off their knuckles: ~0.4 tucked into a fist, ~0.75 for
        # the round O / C shapes, ~0.95 straight.
        reach = float(np.mean([pose.dist(mcp, tip) for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))]))
        self.tips_tucked = down(reach, 0.5, 0.65)
        self.tips_out = up(reach, 0.55, 0.7)
        self.no_hook = 1.0 - self.hook["index"]  # a fist's index bends at the knuckle; an X's doesn't
        self.crossed = up(float(p[12][0] - p[8][0]), 0.0, 0.08)  # index tip past the middle tip
        self.uncrossed = 1.0 - self.crossed

        self.hand_up = up(-float(pose.up_img[1]), 0.45, 0.75)
        self.index_up = up(-float(pose.index_img[1]), 0.45, 0.75)
        self.index_side = up(abs(float(pose.index_img[0])), 0.6, 0.85)
        self.index_down = up(float(pose.index_img[1]), 0.35, 0.65)
        self.hand_down = up(float(pose.up_img[1]), 0.2, 0.5)

    def all(self, state: dict, *fingers: str) -> float:
        return min(state[k] for k in fingers)


def _deg_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))))


# Each template returns the list of condition scores for its letter.
def _templates(x: Features) -> dict[str, list[float]]:
    fist = min(x.all(x.curl, "index", "middle", "ring", "pinky"), x.no_hook, x.tips_tucked)
    four_up = x.all(x.ext, "index", "middle", "ring", "pinky")
    two_up = min(x.ext["index"], x.ext["middle"])
    rest2 = min(x.curl["ring"], x.curl["pinky"])
    rest3 = min(x.folded["middle"], x.folded["ring"], x.folded["pinky"])
    curl3 = min(x.curl["middle"], x.curl["ring"], x.curl["pinky"])
    return {
        "A": [fist, x.thumb_side, 1 - x.thumb_out * 0.8],
        "B": [four_up, x.together, x.thumb_across, x.hand_up],
        # A relaxed open hand is also slightly curved (65-95 deg); a C is cupped (95-150 deg)
        # with the fingertips curving out of the palm towards the thumb.
        "C": [x.cupped, band(x.pose.dist(4, 8), 0.35, 1.0, 0.2), 1 - x.thumb_across * 0.7,
              up(float(x.pose.p[8][2]), 0.25, 0.45)],
        "D": [x.ext["index"], rest3, max(x.touch["middle"], x.touch["ring"]), x.index_up],
        "E": [x.all(x.folded, "index", "middle", "ring", "pinky"), x.thumb_across, x.thumb_under,
              max(x.touch["index"], x.touch["middle"]), 1 - x.slot_m * 0.5],
        "F": [x.touch["index"], x.folded["index"], x.all(x.ext, "middle", "ring", "pinky")],
        "G": [x.ext["index"], curl3, x.index_side, max(x.thumb_side, x.thumb_out)],
        "H": [two_up, x.together, rest2, x.index_side, x.uncrossed],
        "I": [x.ext["pinky"], x.all(x.curl, "index", "middle", "ring"), 1 - x.thumb_out],
        "K": [two_up, x.apart, rest2, x.thumb_mid_joint, x.hand_up, x.uncrossed],
        "L": [x.ext["index"], curl3, x.thumb_out, x.index_up],
        "M": [fist, x.slot_m, x.thumb_under, x.thumb_across],
        "N": [fist, x.slot_n, x.thumb_under, x.thumb_across],
        "O": [x.all(x.folded, "index", "middle", "ring", "pinky"),
              (x.touch["index"] + x.touch["middle"] + x.touch["ring"]) / 3, x.tips_out],
        "P": [two_up, rest2, x.thumb_mid_joint, max(x.index_down, x.hand_down)],
        "Q": [x.ext["index"], curl3, x.index_down, max(x.thumb_side, x.thumb_out, x.touch["index"] * 0.5)],
        "R": [two_up, x.crossed, rest2],
        "S": [fist, x.thumb_across, x.thumb_front],
        "T": [fist, x.slot_t, x.thumb_under],
        "U": [two_up, x.together, rest2, x.thumb_across, x.hand_up, x.uncrossed, 1 - x.thumb_mid_joint * 0.6],
        "V": [two_up, x.apart, rest2, x.thumb_across, x.hand_up, x.uncrossed, 1 - x.thumb_mid_joint * 0.6],
        "W": [x.all(x.ext, "index", "middle", "ring"), x.curl["pinky"], x.apart3],
        "X": [x.hook["index"], curl3, x.index_up * 0.5 + 0.5],
        "Y": [x.ext["pinky"], x.thumb_out, x.all(x.curl, "index", "middle", "ring")],
    }


def _geo_mean(scores: list[float]) -> float:
    return math.exp(sum(math.log(max(s, 0.02)) for s in scores) / len(scores))


def score_letters(pose: HandPose) -> list[tuple[str, float]]:
    """All letters with their template scores, best first."""
    x = Features(pose)
    scored = [(letter, _geo_mean(terms)) for letter, terms in _templates(x).items()]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def classify(pose: HandPose) -> tuple[str, float, list[tuple[str, float]]]:
    """(letter or '', confidence 0..1, top candidates)."""
    ranked = score_letters(pose)
    (best, s1), (_, s2) = ranked[0], ranked[1]
    margin_factor = min(1.0, 0.5 + (s1 - s2) * 4.0)
    confidence = s1 * margin_factor
    if s1 < ACCEPT_SCORE:
        return "", confidence, ranked[:3]
    return best, confidence, ranked[:3]
