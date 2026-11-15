"""Optional LLM arbiter for hard glyphs, via any OpenAI-compatible vision chat endpoint.

An LLM call takes hundreds of milliseconds to seconds, far too slow for every webcam frame,
so it is only consulted "if needed": when the OCR engines disagree, read nothing, or are
below the confidence threshold. Even then the live loop never waits for it:

1. The first doubtful frame of a glyph submits a background request and returns at once.
2. The answer is cached under a perceptual hash of the glyph crop.
3. Later frames of the same held glyph hit the cache, so the arbiter's verdict feeds the
   typing stabiliser at full frame rate. One held letter costs one LLM call.
"""

from __future__ import annotations

import base64
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

PROMPT = (
    "This image is a single character cropped from a webcam frame. "
    "Which letter of the English alphabet (A-Z) is it? "
    "Reply with only that one letter, keeping its case. "
    "If it is not a letter, reply NONE."
)
HASH_SIDE = 16
HASH_TOLERANCE = 24  # differing bits out of 256 still counted as the same glyph
CACHE_SIZE = 64
CACHE_TTL = 60.0  # seconds
FAILURES_BEFORE_BACKOFF = 3
BACKOFF_SECONDS = 30.0


@dataclass
class LLMVerdict:
    letter: str  # '' when the model said NONE or answered something unusable
    raw: str
    latency_ms: float
    cached: bool = False


def glyph_hash(binary: np.ndarray) -> np.ndarray:
    """256-bit shape hash of a (tightly cropped) glyph, stable across webcam frames."""
    small = cv2.resize(binary, (HASH_SIDE, HASH_SIDE), interpolation=cv2.INTER_AREA)
    return (small < 128).ravel()


def parse_letter(content: str) -> str:
    """Extract the single letter from a model reply; '' for NONE / unusable replies."""
    text = (content or "").strip().strip("`*\"'. ")
    if not text or text.upper().startswith("NONE"):
        return ""
    if re.fullmatch(r"[A-Za-z]", text):
        return text
    m = re.search(r"\bletter\s+(?:is\s+)?[\"'“‘(]?([A-Za-z])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    capitals = re.findall(r"\b([A-Z])\b", text)
    return capitals[0] if len(capitals) == 1 else ""


def _png_data_url(gray: np.ndarray, side: int = 224) -> str:
    h, w = gray.shape[:2]
    f = side / float(max(h, w))
    img = cv2.resize(gray, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_CUBIC)
    pad = side // 8
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class LLMArbiter:
    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 15.0) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.label = f"LLM ({model})"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        self._lock = threading.Lock()
        self._cache: list[tuple[np.ndarray, LLMVerdict, float]] = []
        self._pending: np.ndarray | None = None  # hash of the glyph being asked about
        self._failures = 0
        self._backoff_until = 0.0
        self.last_error = ""
        self.calls = 0

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        backing_off = time.monotonic() < self._backoff_until
        return {
            "name": self.name,
            "label": self.label,
            "available": not backing_off,
            "reason": f"paused after errors: {self.last_error}" if backing_off else "",
            "detail": f"{self.model} @ {self.base_url}",
            "calls": self.calls,
            "last_error": self.last_error,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()

    # --------------------------------------------------------------- lookup
    def lookup(self, binary: np.ndarray, gray: np.ndarray, wait: bool = False) -> LLMVerdict | None:
        """Cached verdict for this glyph, or None while a background request is running.

        wait=True blocks for the answer (used by the single-image REST endpoint).
        """
        key = glyph_hash(binary)
        with self._lock:
            hit = self._find(key)
            if hit is not None:
                return LLMVerdict(hit.letter, hit.raw, hit.latency_ms, cached=True)
            if time.monotonic() < self._backoff_until:
                return None
            if self._pending is not None and not wait:
                return None  # one request at a time; this frame goes without
            self._pending = key
        future = self._pool.submit(self._ask_and_store, key, gray)
        return future.result() if wait else None

    def _find(self, key: np.ndarray) -> LLMVerdict | None:
        now = time.monotonic()
        self._cache = [(k, v, t) for k, v, t in self._cache if now - t < CACHE_TTL]
        for k, v, _ in self._cache:
            if np.count_nonzero(k != key) <= HASH_TOLERANCE:
                return v
        return None

    def _ask_and_store(self, key: np.ndarray, gray: np.ndarray) -> LLMVerdict | None:
        try:
            verdict = self.ask(gray)
        except Exception as exc:  # noqa: BLE001 - network/model errors must not break OCR
            with self._lock:
                self._pending = None
                self._failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                if self._failures >= FAILURES_BEFORE_BACKOFF:
                    self._backoff_until = time.monotonic() + BACKOFF_SECONDS
                    self._failures = 0
            return None
        with self._lock:
            self._pending = None
            self._failures = 0
            self.last_error = ""
            self._cache.append((key, verdict, time.monotonic()))
            del self._cache[:-CACHE_SIZE]
        return verdict

    def ask(self, gray: np.ndarray) -> LLMVerdict:
        """Blocking request to the model."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": _png_data_url(gray)}},
                    ],
                }
            ],
        }
        t0 = time.perf_counter()
        self.calls += 1
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return LLMVerdict(parse_letter(content), content.strip()[:40], round((time.perf_counter() - t0) * 1000, 1))
