import asyncio
from pathlib import Path
import socket
import threading
import time

import pytest
import socketio
import uvicorn
from fastapi.testclient import TestClient

from app import main
from tests.synthetic import available_fonts, blank, render

FONT = (available_fonts() or [None])[0]


@pytest.fixture(scope="module")
def client():
    if not main.manager.primary.available:
        pytest.skip(f"Tesseract unavailable: {main.manager.primary.reason}")
    with TestClient(main.app) as c:
        yield c


def test_index_renders_engines_and_config(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Alphabet OCR" in r.text
    assert "Tesseract" in r.text
    assert 'id="app-config"' in r.text and '"hold_frames": 3' in r.text
    assert "/static/app.js" in r.text and "/static/vendor/socket.io.min.js" in r.text


def test_default_strategy_uses_platform_ocr_only_if_available(client):
    expected = "consensus" if main.manager.secondary_available else "primary"
    assert main.default_options().strategy == expected
    assert f'"strategy": "{expected}"' in client.get("/").text


def test_launcher_skips_busy_port():
    from app.__main__ import port_in_use

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert port_in_use("127.0.0.1", port)
    assert not port_in_use("127.0.0.1", _free_port())


def test_static_assets(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/vendor/socket.io.min.js").status_code == 200


def test_engines_endpoint(client):
    data = client.get("/api/engines").json()
    assert data["primary"]["name"] == "tesseract" and data["primary"]["available"]


def test_ocr_endpoint_reads_a_letter(client):
    r = client.post("/api/ocr?mode=letter&strategy=primary", content=render("T", font=FONT))
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "T" and body["engine"] == "tesseract" and body["glyph"]


def test_ocr_endpoint_errors(client):
    assert client.post("/api/ocr", content=b"junk").status_code == 400
    assert client.post("/api/ocr?mode=nope", content=render("T", font=FONT)).status_code == 422


# ----------------------------------------------------------- Socket.IO, real server
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    if not main.manager.primary.available:
        pytest.skip("Tesseract unavailable")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def run(coro):
    return asyncio.run(coro)


async def _connect(url):
    sio = socketio.AsyncClient()
    status = asyncio.get_running_loop().create_future()
    sio.on("status", lambda data: status.done() or status.set_result(data))
    await sio.connect(url, transports=["websocket"])
    return sio, await asyncio.wait_for(status, 5)


def test_socket_types_held_letters(server_url):
    async def scenario():
        sio, status = await _connect(server_url)
        assert status["engines"]["primary"]["available"]
        ack = await sio.call("settings", {"mode": "letter", "hold_frames": 2, "release_frames": 1, "strategy": "primary"})
        assert ack["options"]["hold_frames"] == 2

        commits = []
        frames = [render("A", font=FONT, seed=i, noise=6) for i in range(3)]
        frames += [blank()]
        frames += [render("B", font=FONT, seed=i, noise=6) for i in range(3)]
        for i, image in enumerate(frames):
            res = await sio.call("frame", {"id": i, "image": image}, timeout=10)
            assert res["id"] == i
            if res["commit"]:
                commits.append(res["commit"])
        await sio.disconnect()
        return commits

    assert run(scenario()) == ["A", "B"]


def test_socket_rejects_bad_input(server_url):
    async def scenario():
        sio, _ = await _connect(server_url)
        bad_setting = await sio.call("settings", {"hold_frames": 999})
        not_binary = await sio.call("frame", {"id": 1, "image": "hello"})
        garbage = await sio.call("frame", {"id": 2, "image": b"\x00\x01"})
        await sio.disconnect()
        return bad_setting, not_binary, garbage

    bad_setting, not_binary, garbage = run(scenario())
    assert "hold_frames" in bad_setting["error"] and bad_setting["options"]["hold_frames"] == 3
    assert "binary" in not_binary["error"]
    assert "decode" in garbage["error"]


def test_socket_text_mode_and_reset(server_url):
    async def scenario():
        sio, _ = await _connect(server_url)
        await sio.call("settings", {"mode": "text", "hold_frames": 2, "strategy": "primary"})
        word = render("CAT", font=FONT, text_px=100, size=(640, 260))
        results = [await sio.call("frame", {"id": i, "image": word}, timeout=10) for i in range(3)]
        await sio.call("reset")
        again = [await sio.call("frame", {"id": 10 + i, "image": word}, timeout=10) for i in range(2)]
        await sio.disconnect()
        return results, again

    results, again = run(scenario())
    assert [r["commit"] for r in results] == [None, "CAT", None]
    assert [r["commit"] for r in again] == [None, "CAT"]  # reset allows typing it again


# ------------------------------------------------------------------ hand signs
DATA = Path(__file__).parent / "data"


def test_sign_mode_is_default_and_types_a_signed_letter(server_url):
    if not main.signs.available:
        pytest.skip("hand tracking unavailable")
    victory = (DATA / "victory.jpg").read_bytes()

    async def scenario():
        sio, status = await _connect(server_url)
        assert status["options"]["mode"] == "sign" and status["engines"]["hands"]["available"]
        await sio.call("settings", {"hold_frames": 2, "alphabet": "asl"})
        results = [await sio.call("frame", {"id": i, "image": victory}, timeout=10) for i in range(4)]
        await sio.disconnect()
        return results

    results = run(scenario())
    # Frame 0: the hand is not used yet - it must still be there in the next frame (HandGate).
    assert results[0]["hands"] == [] and not results[0]["found"]
    assert [r["commit"] for r in results] == [None, None, "V", None]
    assert results[1]["mode"] == "sign" and len(results[1]["hands"]) == 1
    assert results[1]["readings"][0]["engine"] in ("asl", "asl-rules")


def test_teaching_a_sign_over_the_socket(server_url):
    if not main.signs.available:
        pytest.skip("hand tracking unavailable")
    fist = (DATA / "thumbs_down.jpg").read_bytes()

    async def scenario():
        sio, _ = await _connect(server_url)
        await sio.call("frame", {"id": -1, "image": fist}, timeout=10)  # hand seen once: confirmed from now on
        bad = await sio.call("teach_start", {"letter": "7"})
        started = await sio.call("teach_start", {"letter": "k", "samples": 5})
        rec = [await sio.call("frame", {"id": i, "image": fist}, timeout=10) for i in range(5)]
        await sio.call("settings", {"alphabet": "custom", "hold_frames": 2})
        typed = [await sio.call("frame", {"id": 10 + i, "image": fist}, timeout=10) for i in range(2)]
        deleted = await sio.call("signs_delete", {"letter": "K"})
        await sio.disconnect()
        return bad, started, rec, typed, deleted

    bad, started, rec, typed, deleted = run(scenario())
    assert "A-Z" in bad["error"]
    assert started["recording"] == {"letter": "K", "total": 5, "done": 0}
    assert [r["recording"]["done"] for r in rec] == [1, 2, 3, 4, 5]
    assert all(r["commit"] is None for r in rec)  # nothing is typed while teaching
    assert rec[-1]["recording"]["finished"] and rec[-1]["taught"] == {"counts": {"K": 5}, "hands": {"K": [1]}}
    assert main.settings.signs_path.is_file()  # saved
    assert [r["commit"] for r in typed] == [None, "K"] and typed[1]["engine"] == "taught"
    assert deleted["counts"] == {}


def test_hands_setting_defaults_to_two_and_can_be_one(server_url):
    if not main.signs.available:
        pytest.skip("hand tracking unavailable")
    two = (DATA / "woman_hands.jpg").read_bytes()

    async def scenario():
        sio, status = await _connect(server_url)
        default = status["options"]["hands"]
        await sio.call("frame", {"id": 0, "image": two}, timeout=10)  # first sighting
        both = await sio.call("frame", {"id": 1, "image": two}, timeout=10)
        ack = await sio.call("settings", {"hands": "1"})  # select elements send strings
        one = await sio.call("frame", {"id": 2, "image": two}, timeout=10)
        bad = await sio.call("settings", {"hands": 3})
        await sio.disconnect()
        return default, both, ack, one, bad

    default, both, ack, one, bad = run(scenario())
    assert default == 2 and len(both["hands"]) == 2
    assert {h["side"] for h in both["hands"]} == {"left", "right"}
    assert ack["options"]["hands"] == 1 and len(one["hands"]) == 1
    assert "hands" in bad["error"]


def _photo_with_one_hand_removed(name):
    """The same photo with its upper hand painted out: the other hand stays exactly where
    it was, as when one hand drops out of view for a moment."""
    import cv2
    import numpy as np

    img = cv2.imread(str(DATA / name))
    h, w = img.shape[:2]
    hands = main.signs.tracker.detect(img)
    upper = min(hands, key=lambda hd: hd.image[:, 1].mean())
    xs, ys = upper.image[:, 0] * w, upper.image[:, 1] * h
    pad = 0.4 * max(np.ptp(xs), np.ptp(ys))
    x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, w))
    y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, h))
    img[y0:y1, x0:x1] = img[0:1, 0:1]  # background colour from the corner
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_teaching_locks_to_two_hands(server_url):
    if not main.signs.available:
        pytest.skip("hand tracking unavailable")
    two = (DATA / "woman_hands.jpg").read_bytes()
    one = _photo_with_one_hand_removed("woman_hands.jpg")

    async def scenario():
        sio, _ = await _connect(server_url)
        await sio.call("frame", {"id": -1, "image": two}, timeout=10)  # first sighting
        await sio.call("teach_start", {"letter": "B", "samples": 5})
        frames = [two, one, two, two, two, two, two]
        rec = [await sio.call("frame", {"id": i, "image": img}, timeout=10) for i, img in enumerate(frames)]
        await sio.call("signs_clear")
        await sio.disconnect()
        return rec

    rec = run(scenario())
    info = [r["recording"] for r in rec]
    # The first frame locks the sign to two hands. While one hand is missing (frame 1) and
    # until it is confirmed again (frame 2), frames are skipped instead of recorded.
    assert [i["visible"] for i in info] == [2, 1, 1, 2, 2, 2, 2]
    assert [i["done"] for i in info] == [1, 1, 1, 2, 3, 4, 5]
    assert info[1]["hands"] == 2 and info[-1]["skipped"] == 2
    assert rec[-1]["taught"]["hands"] == {"B": [2]}
