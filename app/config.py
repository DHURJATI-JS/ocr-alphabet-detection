"""Process-wide settings (from environment) and per-client detection options."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional: read settings from ./.env (see .env.example)
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

Mode = Literal["sign", "letter", "text"]  # sign = hand-sign alphabet; letter/text = OCR
Alphabet = Literal["isl", "asl", "custom"]
Strategy = Literal["fallback", "consensus", "primary", "secondary"]
Case = Literal["upper", "lower", "auto"]
SecondaryChoice = Literal["auto", "apple", "windows", "none"]


@dataclass(frozen=True)
class Settings:
    """Server settings. Override with environment variables (see README)."""

    host: str = "127.0.0.1"
    port: int = 8000
    tessdata_dir: Path = PROJECT_ROOT / "tessdata"
    tesseract_lang: str = "eng"
    tesseract_cmd: str | None = None  # only used by the pytesseract (CLI) backend
    secondary: SecondaryChoice = "auto"
    windows_language: str = "en-US"
    workers: int = 4
    hand_model: Path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    hand_presence: float = 0.7  # landmark model's "this is a hand" score; see hands/tracker.py
    face_model: Path = PROJECT_ROOT / "models" / "blaze_face_short_range.tflite"
    signs_path: Path = PROJECT_ROOT / "data" / "signs.json"  # signs taught in the UI
    # Optional LLM arbiter: any OpenAI-compatible endpoint with a vision-capable model
    # (LAN gateway, Ollama http://localhost:11434/v1, LM Studio http://localhost:1234/v1).
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gemma"
    llm_timeout: float = 15.0

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        secondary = env.get("OCR_SECONDARY", "auto").lower()
        if secondary not in ("auto", "apple", "windows", "none"):
            raise ValueError(f"OCR_SECONDARY must be auto|apple|windows|none, got {secondary!r}")
        return cls(
            host=env.get("HOST", cls.host),
            port=int(env.get("PORT", cls.port)),
            tessdata_dir=Path(env.get("TESSDATA_DIR", str(cls.tessdata_dir))).expanduser(),
            tesseract_lang=env.get("TESSERACT_LANG", cls.tesseract_lang),
            tesseract_cmd=env.get("TESSERACT_CMD") or None,
            secondary=secondary,  # type: ignore[arg-type]
            windows_language=env.get("WINDOWS_OCR_LANGUAGE", cls.windows_language),
            workers=max(1, int(env.get("OCR_WORKERS", cls.workers))),
            hand_model=Path(env.get("HAND_MODEL", str(cls.hand_model))).expanduser(),
            hand_presence=float(env.get("HAND_PRESENCE", cls.hand_presence)),
            face_model=Path(env.get("FACE_MODEL", str(cls.face_model))).expanduser(),
            signs_path=Path(env.get("SIGNS_PATH", str(cls.signs_path))).expanduser(),
            llm_base_url=(env.get("LLM_BASE_URL") or "").strip().rstrip("/") or None,
            llm_api_key=env.get("LLM_API_KEY") or None,
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            llm_timeout=float(env.get("LLM_TIMEOUT", cls.llm_timeout)),
        )


class DetectionOptions(BaseModel):
    """Options a browser client can change at runtime over the socket."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    mode: Mode = "sign"
    # Sign mode: which fingerspelling alphabet the trained recogniser reads - "isl" (Indian,
    # mostly two-handed) or "asl" (American) - or "custom": only signs taught in the page.
    # Taught signs take priority in every alphabet.
    alphabet: Alphabet = "isl"
    # Sign mode: how many hands to track. 2 = two-handed signs (e.g. ISL) and both hands
    # shown; 1 = only the most prominent hand (the other is ignored).
    hands: int = Field(2, ge=1, le=2)
    # "consensus": Tesseract and the platform OCR (Apple Vision / Windows OCR) both read every
    # frame. Degrades to Tesseract alone when the platform OCR is not available.
    strategy: Strategy = "consensus"
    case: Case = "upper"
    # Engine confidence (0-1) a reading needs before it counts towards typing (OCR modes).
    min_confidence: float = Field(0.55, ge=0.0, le=1.0)
    # The same for hand signs. Measured on photos of people the model never saw: 0.7 types
    # 79.7% right / 8.6% wrong versus 81.2% / 12.5% at 0.55 - a wrong letter costs a deaf
    # user more than a letter they have to sign again.
    sign_min_confidence: float = Field(0.7, ge=0.0, le=1.0)

    @property
    def typing_confidence(self) -> float:
        return self.sign_min_confidence if self.mode == "sign" else self.min_confidence
    # Consecutive identical frames required before a reading is typed.
    hold_frames: int = Field(3, ge=1, le=30)
    # Blank frames required before the same letter can be typed again.
    release_frames: int = Field(2, ge=1, le=30)
    # In "fallback" strategy, the secondary engine runs when the primary is below this.
    fallback_below: float = Field(0.70, ge=0.0, le=1.0)
    # OCR modes: ignore faces (eyes / nostrils / mouth look like letters) and shapes on skin.
    ignore_faces: bool = True
    # Ask the LLM (if configured) about glyphs the OCR engines disagree on or can't read.
    llm_arbiter: bool = True
    # Return a thumbnail of what the OCR engines actually saw.
    debug_image: bool = False

    def merged(self, changes: dict) -> "DetectionOptions":
        return DetectionOptions.model_validate({**self.model_dump(), **(changes or {})})
