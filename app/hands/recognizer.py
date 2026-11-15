"""Hand-sign letters: MediaPipe hands -> a recogniser trained on real photos of the chosen
alphabet (ISL / ASL), with the user's own taught signs taking priority."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import DetectionOptions
from ..letters import apply_case
from ..preprocess import decode_image
from ..results import EngineReading, FrameResult
from . import asl
from .geometry import HandPose, analyze, body_side, frame_features, order_hands, pose_features, prominence
from .gate import HandGate
from .library import SignLibrary
from .motion import MotionTracker
from .tracker import HandTracker
from .trained import TrainedAlphabet

ALPHABET_LABELS = {"isl": "ISL", "asl": "ASL"}

MAX_SIDE = 640  # hands are large in the frame; more pixels only cost time
AGREEMENT_BOOST = 0.15
TAUGHT_MIN = 0.45  # taught match needed to override the trained recogniser


def _sides(hands) -> list[str]:
    """The signer's left/right for each hand, or '' when MediaPipe's handedness is
    contradictory (both hands given the same label, which happens with unusual poses):
    no tag is better than a wrong one."""
    sides = [body_side(hd.label) for hd in hands]
    if len(sides) == 2 and sides[0] == sides[1]:
        return ["", ""]
    return sides


class SignRecognizer:
    def __init__(self, tracker: HandTracker, library: SignLibrary,
                 alphabets: dict[str, TrainedAlphabet] | None = None) -> None:
        self.tracker = tracker
        self.library = library
        self.alphabets = alphabets or {}

    @classmethod
    def load_alphabets(cls, models_dir) -> dict[str, TrainedAlphabet]:
        from pathlib import Path

        out = {}
        for name in ALPHABET_LABELS:
            path = Path(models_dir) / f"{name}_signs.npz"
            if path.is_file():
                out[name] = TrainedAlphabet.load(path)
        return out

    @property
    def available(self) -> bool:
        return self.tracker.available

    def status(self) -> dict:
        st = self.tracker.status()
        st["taught"] = self.library.counts()
        st["taught_hands"] = self.library.hands_by_letter()
        st["alphabets"] = {name: {"letters": "".join(m.letters), "sources": m.meta.get("sources", [])}
                           for name, m in self.alphabets.items()}
        if self.library.load_error:
            st["library_error"] = self.library.load_error
        return st

    def process(self, image_bytes: bytes, opts: DetectionOptions, motion: MotionTracker | None = None,
                gate: HandGate | None = None) -> tuple[FrameResult, tuple[int, np.ndarray] | None]:
        """Recognise one frame. Returns the result and the frame's feature vector (for teaching).

        gate: for a live stream, only use hands that were also seen in the previous frame.
        """
        t0 = time.perf_counter()
        bgr = decode_image(image_bytes)
        h, w = bgr.shape[:2]
        if max(h, w) > MAX_SIDE:
            f = MAX_SIDE / float(max(h, w))
            bgr = cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        hands = self.tracker.detect(bgr) if self.tracker.available else []
        t_detect = (time.perf_counter() - t0) * 1000
        if gate is not None:
            hands = gate.filter(hands)
        # The tracker always looks for two hands; in one-hand mode keep the most prominent.
        hands = sorted(hands, key=prominence, reverse=True)[: opts.hands]
        poses = order_hands([analyze(hd) for hd in hands])
        hands = [ps.hand for ps in poses]

        readings: list[EngineReading] = []
        model_pick: tuple[str, float] = ("", 0.0)
        model_engine = opts.alphabet
        if poses and opts.alphabet in self.alphabets:
            model_pick, reading = self._trained(poses, opts)
            readings.append(reading)
        elif poses and opts.alphabet == "asl":  # no trained ASL model on disk: built-in rules
            model_pick, reading = self._rules(poses, opts)
            model_engine = "asl-rules"
            readings.append(reading)
        taught_pick = ("", 0.0)
        features = frame_features(poses) if poses else None
        if poses and len(self.library):
            taught_pick, reading = self._taught(poses, opts)
            readings.append(reading)

        text, conf, engine, agreement = self._fuse(model_pick, model_engine, taught_pick)

        result = FrameResult(
            mode="sign", strategy=opts.alphabet, text=text, confidence=round(conf, 3), engine=engine,
            agreement=agreement, found=bool(poses), glyph=None, readings=readings,
            timings={"hands_ms": round(t_detect, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
            hands=[{"label": hd.label, "side": side, "score": round(hd.score, 3),
                    "points": [[round(float(x), 4), round(float(y), 4)] for x, y in hd.image[:, :2]]}
                   for hd, side in zip(hands, _sides(hands))],
        )
        if motion is not None and opts.alphabet == "asl":  # ASL J / Z are drawn in the air
            main_pose = max(poses, key=lambda ps: ps.palm_img) if poses else None
            event = motion.update(main_pose, text)
            if event:
                result.motion, result.motion_base = apply_case(event[0], opts.case), event[1]
        return result, features

    # ------------------------------------------------------------------ sources
    def _rules(self, poses: list[HandPose], opts: DetectionOptions):
        t0 = time.perf_counter()
        best = ("", 0.0, None, [])
        for pose in poses:  # ASL is one-handed: take whichever hand signs most clearly
            letter, conf, top = asl.classify(pose)
            if conf > best[1] or best[2] is None:
                best = (letter, conf, pose, top)
        letter, conf, pose, top = best
        letter = apply_case(letter, opts.case)
        reading = EngineReading(
            engine="asl-rules", label="ASL (rules)", text=letter,
            raw=" · ".join(f"{lt} {round(s * 100)}" for lt, s in top), confidence=round(conf if letter else 0.0, 3),
            confidence_estimated=False, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return (letter, conf if letter else 0.0), reading

    def _trained(self, poses: list[HandPose], opts: DetectionOptions):
        t0 = time.perf_counter()
        letter, conf, top = self.alphabets[opts.alphabet].predict(poses)
        letter = apply_case(letter, opts.case)
        reading = EngineReading(
            engine=opts.alphabet, label=f"{ALPHABET_LABELS[opts.alphabet]} (trained)", text=letter,
            raw=" · ".join(f"{lt} {round(s * 100)}" for lt, s in top) + ("" if letter else " (not a known sign)"),
            confidence=round(conf if letter else 0.0, 3), confidence_estimated=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return (letter, conf if letter else 0.0), reading

    def _taught(self, poses: list[HandPose], opts: DetectionOptions):
        t0 = time.perf_counter()
        candidates = []
        n, vec = frame_features(poses)
        if n in self.library.hand_counts():
            candidates.append(self.library.classify(n, vec))
        if len(poses) == 2 and 1 in self.library.hand_counts():  # one-handed signs, other hand in view
            candidates += [self.library.classify(1, pose_features(p)) for p in poses]
        letter, conf, top = max(candidates, key=lambda c: (bool(c[0]), c[1]), default=("", 0.0, []))
        letter = apply_case(letter, opts.case)
        reading = EngineReading(
            engine="taught", label="Your signs", text=letter,
            raw=" · ".join(f"{lt} {round(s * 100)}" for lt, s in top) or "no match",
            confidence=round(conf if letter else 0.0, 3), confidence_estimated=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return (letter, conf if letter else 0.0), reading

    @staticmethod
    def _fuse(model: tuple[str, float], model_engine: str, taught: tuple[str, float]):
        (m_text, m_conf), (t_text, t_conf) = model, taught
        if m_text and t_text and m_text == t_text:
            return m_text, min(1.0, max(m_conf, t_conf) + AGREEMENT_BOOST), f"{model_engine}+taught", True
        if t_text and (t_conf >= TAUGHT_MIN or not m_text):
            return t_text, t_conf, "taught", (False if m_text else None)
        if m_text:
            return m_text, m_conf, model_engine, (False if t_text else None)
        return "", 0.0, None, None
