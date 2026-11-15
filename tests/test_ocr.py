import json
import string
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import DetectionOptions


def Opts(**kw):
    """OCR options (the app's default mode is hand signs)."""
    return DetectionOptions(**{"mode": "letter", **kw})
from app.ocr.base import EngineOutput, OCREngine
from app.ocr.llm import LLMArbiter
from app.ocr.manager import OCRManager, self_test
from tests.synthetic import available_fonts, blank, render

FONTS = available_fonts() or [None]
WEBCAM = {"noise": 8, "blur": 1.2, "gradient": 35, "rotate": 4}
# Every reading counts as unsure (Tesseract alone never reports a full 100%), so the LLM
# arbiter is always consulted. With both engines, an agreed reading is boosted to 100%.
UNSURE = Opts(min_confidence=1.0, strategy="primary")


def accuracy(manager, opts, fonts, **kw):
    ok = total = 0
    for fi, font in enumerate(fonts):
        for li, ch in enumerate(string.ascii_uppercase):
            res = manager.process(render(ch, font=font, seed=fi * 100 + li, **kw), opts)
            ok += res.text == ch
            total += 1
    return ok / total


def test_tesseract_reads_webcam_letters(manager):
    opts = Opts(strategy="primary")
    assert accuracy(manager, opts, FONTS[:3], **WEBCAM) >= 0.95


def test_fallback_strategy_reads_webcam_letters(manager):
    opts = Opts(strategy="fallback")
    assert accuracy(manager, opts, FONTS[:3], **WEBCAM) >= 0.97


def test_tesseract_is_deterministic(manager):
    """The legacy classifier's adaptive learning is off: same input, same output."""
    data = [render(ch, font=FONTS[0], **WEBCAM) for ch in "OQRIZ"]
    opts = Opts(strategy="primary")
    first = [(r.text, r.confidence) for r in (manager.process(d, opts) for d in data)]
    for _ in range(2):
        assert [(r.text, r.confidence) for r in (manager.process(d, opts) for d in data)] == first


def test_blank_frame_reads_nothing_without_running_engines(manager):
    res = manager.process(blank(), Opts())
    assert res.text == "" and not res.found and res.readings == []


def test_case_option(manager):
    data = render("G", font=FONTS[0])
    assert manager.process(data, Opts(case="lower", strategy="primary")).text == "g"
    assert manager.process(data, Opts(case="upper", strategy="primary")).text == "G"


def test_text_mode_reads_a_word(manager):
    data = render("HELLO", font=FONTS[0], text_px=90, size=(640, 240), noise=5)
    res = manager.process(data, Opts(mode="text", strategy="primary"))
    assert res.text == "HELLO"
    assert res.readings[0].boxes  # word boxes for the overlay


def test_consensus_runs_both_engines(manager):
    if not (manager.secondary and manager.secondary.available):
        pytest.skip("no secondary engine on this platform")
    res = manager.process(render("K", font=FONTS[0]), Opts(strategy="consensus"))
    assert [r.engine for r in res.readings] == ["tesseract", manager.secondary.name]
    assert res.text == "K" and res.agreement is True and "+" in res.engine


def test_secondary_engine_alone(manager):
    if not (manager.secondary and manager.secondary.available):
        pytest.skip("no secondary engine on this platform")
    res = manager.process(render("E", font=FONTS[0]), Opts(strategy="secondary"))
    assert res.text == "E" and res.engine == manager.secondary.name


def test_default_runs_tesseract_and_platform_ocr_every_frame(manager):
    if not manager.secondary_available:
        pytest.skip("no platform OCR on this machine")
    for ch in "KAX":
        res = manager.process(render(ch, font=FONTS[0]), Opts())
        assert res.strategy == "consensus"
        assert [r.engine for r in res.readings] == ["tesseract", manager.secondary.name]
        assert res.text == ch


class StubEngine(OCREngine):
    name, label = "stub", "Stub OCR"

    def __init__(self, available=True, fail=False):
        super().__init__()
        self.available, self.fail, self.calls = available, fail, 0
        self.reason = "" if available else "not on this OS"

    def recognize(self, image, mode):
        self.calls += 1
        if self.fail:
            raise OSError("OCR language pack missing")
        return EngineOutput(text="Z", confidence=0.99)


def test_unavailable_platform_ocr_falls_back_to_tesseract_alone(manager):
    stub = StubEngine(available=False)
    m = OCRManager(manager.primary, stub)
    for strategy in ("consensus", "fallback", "secondary", "primary"):
        res = m.process(render("K", font=FONTS[0]), Opts(strategy=strategy))
        assert res.strategy == "primary"
        assert [r.engine for r in res.readings] == ["tesseract"] and res.text == "K"
    assert stub.calls == 0
    m.close()


def test_available_platform_ocr_is_used(manager):
    stub = StubEngine(available=True)
    m = OCRManager(manager.primary, stub)
    res = m.process(render("K", font=FONTS[0]), Opts(strategy="consensus"))
    assert [r.engine for r in res.readings] == ["tesseract", "stub"] and stub.calls == 1
    m.close()


def test_self_test_marks_broken_engine_unavailable():
    broken, healthy = StubEngine(fail=True), StubEngine()
    self_test(broken)
    self_test(healthy)
    assert not broken.available and "self-test" in broken.reason and "language pack" in broken.reason
    assert healthy.available


def test_debug_image(manager):
    res = manager.process(render("B", font=FONTS[0]), Opts(debug_image=True))
    assert res.debug_image.startswith("data:image/png;base64,")


# ------------------------------------------------------------------ LLM arbiter
class FakeLLM:
    """Minimal OpenAI-compatible /v1/chat/completions server."""

    def __init__(self, reply="Z", delay=0.0, status=200):
        self.reply, self.delay, self.status = reply, delay, status
        self.requests = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                time.sleep(fake.delay)
                out = json.dumps({"choices": [{"message": {"role": "assistant", "content": fake.reply}}]}).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_llm():
    servers = []

    def make(**kw):
        servers.append(FakeLLM(**kw))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def with_llm(manager, url):
    return OCRManager(manager.primary, manager.secondary, LLMArbiter(url, "gemma", "secret-key", timeout=5))


def test_llm_not_consulted_when_engines_are_confident(manager, fake_llm):
    llm = fake_llm(reply="X")
    m = with_llm(manager, llm.url)
    res = m.process(render("K", font=FONTS[0]), Opts(), wait_for_llm=True)
    assert res.text == "K" and llm.requests == []
    m.llm.close()


def test_llm_arbitrates_unsure_glyph(manager, fake_llm):
    llm = fake_llm(reply="Z")
    m = with_llm(manager, llm.url)
    opts = UNSURE  # force "unsure" so the arbiter is needed
    res = m.process(render("Z", font=FONTS[0]), opts, wait_for_llm=True)
    assert len(llm.requests) == 1
    req = llm.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer secret-key"
    content = req["body"]["messages"][0]["content"]
    assert req["body"]["model"] == "gemma"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert res.text == "Z" and "llm" in res.engine
    assert res.readings[-1].engine == "llm"
    m.llm.close()


def test_llm_runs_in_background_and_caches_per_glyph(manager, fake_llm):
    llm = fake_llm(reply="Q", delay=0.3)
    m = with_llm(manager, llm.url)
    opts = UNSURE
    frames = [render("Q", font=FONTS[0], seed=s, noise=6) for s in range(4)]

    t0 = time.perf_counter()
    first = m.process(frames[0], opts)  # live path: must not wait for the LLM
    assert time.perf_counter() - t0 < 0.25
    assert first.readings[-1].note == "asking…"

    time.sleep(0.6)
    later = [m.process(f, opts) for f in frames[1:]]
    assert len(llm.requests) == 1  # the same held glyph is asked about once
    assert all(r.readings[-1].note == "cached" and r.text == "Q" for r in later)
    m.llm.close()


def test_llm_errors_are_reported_and_back_off(manager, fake_llm):
    llm = fake_llm(reply="nope", status=500)
    m = with_llm(manager, llm.url)
    opts = UNSURE
    for i in range(5):
        res = m.process(render("W", font=FONTS[0], seed=i), opts, wait_for_llm=True)
        assert res.text in ("", "W")  # OCR result still returned
    assert len(llm.requests) == 3  # paused after 3 consecutive failures
    status = m.llm.status()
    assert not status["available"] and "500" in status["last_error"]
    m.llm.close()


class SlowEngine(OCREngine):
    """Stands in for an OS OCR service that stalls."""

    name, label = "slow", "Stalling OCR"

    def __init__(self, delay):
        super().__init__()
        self.available, self.delay, self.calls = True, delay, 0

    def recognize(self, image, mode):
        import time as _t

        self.calls += 1
        _t.sleep(self.delay)
        return EngineOutput(text="Z", confidence=0.99)


def test_a_stalling_engine_cannot_freeze_frames(manager):
    import time as _t

    slow = SlowEngine(delay=2.0)
    m = OCRManager(manager.primary, slow)
    m.engine_timeout = 0.3
    data = render("K", font=FONTS[0])
    t0 = _t.perf_counter()
    first = m.process(data, Opts(strategy="consensus"))
    assert _t.perf_counter() - t0 < 1.0  # did not wait for the 2 s engine
    assert first.text == "K" and "no answer" in first.readings[1].error
    second = m.process(data, Opts(strategy="consensus"))
    assert second.readings[1].note and "benched" in second.readings[1].note
    third = m.process(data, Opts(strategy="consensus"))  # benched: Tesseract alone, instantly
    assert third.strategy == "primary" and [r.engine for r in third.readings] == ["tesseract"]
    m.close()
