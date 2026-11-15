"""Result types shared by the OCR and hand-sign recognisers (what the browser receives)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EngineReading:
    engine: str
    label: str
    text: str  # normalised letters ('' when nothing usable)
    raw: str  # what the engine actually returned
    confidence: float
    confidence_estimated: bool
    latency_ms: float
    boxes: list[dict] = field(default_factory=list)  # 0..1 coords of the submitted image
    error: str | None = None
    note: str | None = None  # e.g. "asking…" / "cached" for the LLM


@dataclass
class FrameResult:
    mode: str
    strategy: str
    text: str  # best reading ('' = nothing confident in view)
    confidence: float
    engine: str | None  # which engine(s) produced `text`
    agreement: bool | None  # when two sources were compared: did they agree?
    found: bool  # something (glyph / ink / hand) was in view
    glyph: dict | None  # letter mode: glyph box, 0..1 coords of the submitted image
    readings: list[EngineReading]
    timings: dict
    debug_image: str | None = None  # data: URL of what the engines saw
    hands: list[dict] | None = None  # sign mode: [{label, points: [[x, y], ...]}] in 0..1 coords
    motion: str | None = None  # sign mode: a letter drawn in the air (J / Z) on this frame
    motion_base: str | None = None  # the static handshape the motion started from

    def as_dict(self) -> dict:
        return asdict(self)
