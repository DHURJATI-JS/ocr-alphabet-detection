// Live webcam -> Socket.IO -> recogniser -> typed letters.
//
// Sign mode sends the whole frame (hands move around); OCR modes send only the guide box.
// Exactly one frame is in flight at a time: the next one is captured when the server's ack
// arrives, so a slow pass lowers the frame rate instead of building up a queue.
(() => {
  "use strict";

  const CONFIG = JSON.parse(document.getElementById("app-config").textContent);
  const $ = (id) => document.getElementById(id);
  const HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
  ];

  const els = {
    conn: $("conn"), chips: $("engine-chips"),
    stage: $("stage"), video: $("video"), overlay: $("overlay"), recBanner: $("rec-banner"),
    placeholder: $("placeholder"), placeholderText: $("placeholder-text"),
    startBtn: $("start-btn"), startInline: $("start-btn-inline"),
    cameraSelect: $("camera-select"), mirror: $("mirror"), roi: $("roi"),
    fps: $("fps"), fpsOut: $("fps-out"),
    statFps: $("stat-fps"), statRtt: $("stat-rtt"), statOcr: $("stat-ocr"), statMsg: $("stat-msg"),
    statHands: $("stat-hands"),
    bigLetter: $("big-letter"), ring: $("ring"), bestEngine: $("best-engine"), bestConf: $("best-conf"),
    confFill: $("confbar-fill"), hint: $("hint"), readings: $("readings"), debugImg: $("debug-img"),
    output: $("output"), autotype: $("autotype"), history: $("history"),
    letterGrid: $("letter-grid"), teachRecord: $("teach-record"), teachDelete: $("teach-delete"),
    teachClear: $("teach-clear"), teachSamples: $("teach-samples"), teachTotal: $("teach-total"),
    teachProgress: document.querySelector(".teach-progress"), teachFill: $("teach-progress-fill"),
  };

  const RING_LEN = 2 * Math.PI * 54;
  els.ring.style.strokeDasharray = `${RING_LEN}`;
  els.ring.style.strokeDashoffset = `${RING_LEN}`;

  const state = {
    options: { ...CONFIG.defaults },
    stream: null,
    running: false,
    inFlight: false,
    frameId: 0,
    lastResult: null,
    lastRoi: null, // {x, y, w, h} in video pixels for the frame that produced lastResult
    acks: [],
    teachLetter: "A",
    recording: null, // {letter, total, done} while teaching
    taught: {}, // letter -> example count
    taughtHands: {}, // letter -> [1] / [2] / [1, 2]: hands it was taught with
  };

  const grab = document.createElement("canvas"); // offscreen: the part of the frame we send
  const grabCtx = grab.getContext("2d");
  const ctx = els.overlay.getContext("2d");
  const isSign = () => state.options.mode === "sign";

  // ------------------------------------------------------------------ socket
  const socket = io({ transports: ["websocket", "polling"] });

  socket.on("connect", () => {
    setConn(true);
    // After a reconnect the server has a fresh session: push our settings again.
    socket.emit("settings", state.options, (res) => res && res.options && applyOptions(res.options));
  });
  socket.on("disconnect", () => { setConn(false); state.inFlight = false; setRecording(null); });
  socket.on("connect_error", () => setConn(false));
  socket.on("status", (msg) => { renderChips(msg.engines); renderTaught(msg.taught); });
  socket.on("taught", (msg) => renderTaught(msg));

  function setConn(on) {
    els.conn.textContent = on ? "connected" : "offline";
    els.conn.className = `pill ${on ? "pill-on" : "pill-off"}`;
  }

  function renderChips(engines) {
    els.chips.innerHTML = "";
    for (const [role, info] of Object.entries(engines || {})) {
      if (!info) continue;
      const chip = document.createElement("span");
      chip.className = `chip ${info.available ? "chip-ok" : "chip-off"}`;
      chip.title = info.available ? info.detail : info.reason;
      chip.textContent = info.label;
      const small = document.createElement("small");
      small.textContent = role;
      chip.appendChild(small);
      els.chips.appendChild(chip);
    }
  }

  // ---------------------------------------------------------------- settings
  const optInputs = document.querySelectorAll("[data-opt]");
  const modeRadios = document.querySelectorAll('input[name="mode-radio"]');

  function applyOptions(opts) {
    const modeChanged = opts.mode && opts.mode !== state.options.mode;
    state.options = { ...state.options, ...opts };
    for (const input of optInputs) {
      const key = input.dataset.opt;
      if (!(key in state.options)) continue;
      if (input.type === "checkbox") input.checked = !!state.options[key] && !input.disabled;
      else input.value = state.options[key];
      const out = document.querySelector(`output[data-for="${key}"]`);
      if (out) out.textContent = formatOpt(key, state.options[key]);
    }
    modeRadios.forEach((r) => { r.checked = r.value === state.options.mode; });
    document.body.classList.toggle("mode-sign", isSign());
    document.body.classList.toggle("mode-ocr", !isSign());
    els.debugImg.hidden = isSign() || !state.options.debug_image;
    els.hint.textContent = isSign()
      ? "Hold a sign steady until the ring fills. J and Z: draw them in the air."
      : state.options.mode === "letter"
        ? "Hold a printed letter inside the guide box until the ring fills."
        : "Hold a printed word inside the guide box until the ring fills.";
    if (modeChanged) { state.lastResult = null; drawOverlay(); }
  }

  function formatOpt(key, value) {
    if (key === "min_confidence" || key === "sign_min_confidence") return `${Math.round(value * 100)}%`;
    return String(value);
  }

  function sendSetting(changes) {
    socket.emit("settings", changes, (res) => {
      if (res && res.error) flash(`Setting rejected: ${res.error}`);
      if (res && res.options) applyOptions(res.options);
    });
  }

  for (const input of optInputs) {
    const event = input.type === "range" ? "input" : "change";
    input.addEventListener(event, () => {
      const key = input.dataset.opt;
      let value = input.type === "checkbox" ? input.checked : input.value;
      if (input.type === "range") value = Number(value);
      const out = document.querySelector(`output[data-for="${key}"]`);
      if (out) out.textContent = formatOpt(key, value);
      sendSetting({ [key]: value });
    });
  }
  modeRadios.forEach((r) => r.addEventListener("change", () => {
    if (!r.checked) return;
    if (r.value === "sign" && !els.mirror.checked) { els.mirror.checked = true; mirrorChanged(); }
    applyOptions({ mode: r.value }); // instant UI feedback; server confirms
    sendSetting({ mode: r.value });
  }));

  els.fps.addEventListener("input", () => { els.fpsOut.textContent = els.fps.value; });
  els.fpsOut.textContent = els.fps.value;
  els.mirror.checked = isSign(); // signing feels natural in a mirror; printed text does not
  applyOptions(state.options);

  // ------------------------------------------------------------------ camera
  els.startBtn.addEventListener("click", toggleCamera);
  els.startInline.addEventListener("click", toggleCamera);
  els.cameraSelect.addEventListener("change", () => state.running && startCamera(els.cameraSelect.value));
  els.mirror.addEventListener("change", mirrorChanged);
  function mirrorChanged() {
    els.video.classList.toggle("mirrored", els.mirror.checked);
    drawOverlay();
  }
  mirrorChanged();

  async function toggleCamera() {
    if (state.running) stopCamera();
    else await startCamera(els.cameraSelect.value || undefined);
  }

  async function startCamera(deviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showPlaceholder("Camera access needs a secure page: open the localhost address (not a LAN IP) or use HTTPS.");
      return;
    }
    stopCamera();
    const constraints = (id) => {
      const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
      if (id) video.deviceId = { exact: id };
      return { video, audio: false };
    };
    try {
      try {
        state.stream = await navigator.mediaDevices.getUserMedia(constraints(deviceId));
      } catch (err) {
        // The chosen camera may have been unplugged: fall back to the default one.
        if (!deviceId || !["OverconstrainedError", "NotFoundError"].includes(err.name)) throw err;
        state.stream = await navigator.mediaDevices.getUserMedia(constraints());
      }
    } catch (err) {
      showPlaceholder(err.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it in the browser's site settings and try again."
        : `Could not open the camera: ${err.message || err.name}`);
      return;
    }
    els.video.srcObject = state.stream;
    await els.video.play().catch(() => {});
    await listCameras();
    state.running = true;
    els.placeholder.hidden = true;
    els.startBtn.textContent = "Stop camera";
    if (els.video.readyState >= 1) onVideoSize();
    loop();
  }

  function stopCamera() {
    state.running = false;
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    els.video.srcObject = null;
    els.startBtn.textContent = "Start camera";
    if (guided) stopGuided();
    if (state.recording) stopTeaching();
    state.lastResult = null;
    drawOverlay();
    showPlaceholder("Camera is off");
  }

  function showPlaceholder(text) {
    els.placeholderText.textContent = text;
    els.placeholder.hidden = false;
  }

  async function listCameras() {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const current = state.stream && state.stream.getVideoTracks()[0]?.getSettings().deviceId;
    els.cameraSelect.innerHTML = "";
    devices.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${i + 1}`;
      if (d.deviceId === current) opt.selected = true;
      els.cameraSelect.appendChild(opt);
    });
    els.cameraSelect.disabled = devices.length < 2;
  }

  els.video.addEventListener("loadedmetadata", onVideoSize);
  window.addEventListener("resize", sizeOverlay);

  function onVideoSize() {
    const { videoWidth: w, videoHeight: h } = els.video;
    if (w && h) els.stage.style.aspectRatio = `${w} / ${h}`;
    sizeOverlay();
  }

  function sizeOverlay() {
    const rect = els.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    els.overlay.width = Math.round(rect.width * dpr);
    els.overlay.height = Math.round(rect.height * dpr);
    drawOverlay();
  }

  // ------------------------------------------------------------- frame loop
  function captureRegion() {
    // Region of the video (in video pixels) that is sent: whole frame for hand signs,
    // a square guide box for a printed letter, a wide one for printed words.
    const vw = els.video.videoWidth, vh = els.video.videoHeight;
    if (isSign()) return { x: 0, y: 0, w: vw, h: vh };
    const f = Number(els.roi.value) / 100;
    let w, h;
    if (state.options.mode === "letter") {
      w = h = Math.round(f * Math.min(vw, vh));
    } else {
      w = Math.round(f * vw);
      h = Math.round(Math.min(vh * f, w * 0.45));
    }
    return { x: Math.round((vw - w) / 2), y: Math.round((vh - h) / 2), w, h };
  }

  function loop() {
    if (!state.running) return;
    const interval = 1000 / Number(els.fps.value);
    const started = performance.now();
    sendFrame().finally(() => {
      const wait = Math.max(0, interval - (performance.now() - started));
      setTimeout(loop, wait);
    });
  }

  async function sendFrame() {
    if (state.inFlight || !socket.connected || document.hidden) return;
    if (els.video.readyState < 2 || !els.video.videoWidth) return;
    const roi = captureRegion();
    const scale = Math.min(1, 640 / Math.max(roi.w, roi.h)); // the server never needs more
    grab.width = Math.round(roi.w * scale);
    grab.height = Math.round(roi.h * scale);
    grabCtx.drawImage(els.video, roi.x, roi.y, roi.w, roi.h, 0, 0, grab.width, grab.height);
    const blob = await new Promise((resolve) => grab.toBlob(resolve, "image/jpeg", 0.85));
    if (!blob) return;
    const image = await blob.arrayBuffer();

    state.inFlight = true;
    const id = ++state.frameId;
    const sentAt = performance.now();
    await new Promise((resolve) => {
      socket.timeout(10000).emit("frame", { id, image }, (err, res) => {
        state.inFlight = false;
        if (err) { flash("Server did not answer in time"); resolve(); return; }
        handleResult(res, roi, performance.now() - sentAt);
        resolve();
      });
    });
  }

  // ---------------------------------------------------------------- results
  function handleResult(res, roi, rtt) {
    if (!res) return;
    if (res.error) { flash(res.error); return; }
    if (res.dropped) return;
    state.lastResult = res;
    state.lastRoi = roi;

    const now = performance.now();
    state.acks.push(now);
    while (state.acks.length && now - state.acks[0] > 2000) state.acks.shift();
    els.statFps.textContent = (state.acks.length / 2).toFixed(1);
    els.statRtt.textContent = `${Math.round(rtt)} ms`;
    els.statOcr.textContent = `${Math.round(res.timings.total_ms)} ms`;
    els.statHands.textContent = String((res.hands || []).length);

    if (res.recording) updateRecording(res);

    const shown = res.recording ? "" : (res.candidate || res.text || "");
    els.bigLetter.textContent = shown || "–";
    els.bigLetter.classList.toggle("long", shown.length > 2);
    els.ring.style.strokeDashoffset = `${RING_LEN * (1 - (res.recording ? 0 : (res.progress || 0)))}`;
    els.bestEngine.textContent = res.text ? labelFor(res.engine)
      : (res.found ? "unsure" : (isSign() ? "no hand in view" : "nothing in view"));
    els.bestConf.textContent = res.text ? `${Math.round(res.confidence * 100)}%` : "–";
    els.confFill.style.width = `${Math.round((res.confidence || 0) * 100)}%`;
    const needed = isSign() ? state.options.sign_min_confidence : state.options.min_confidence;
    els.confFill.classList.toggle("low", res.confidence < needed);

    renderReadings(res.readings, res.found);
    if (!isSign() && state.options.debug_image && res.debug_image) els.debugImg.src = res.debug_image;
    if (res.commit) {
      typeCommit(res.commit, res.replace || 0);
      pause.since = null;
    }
    if (isSign() && !res.recording) checkPause(res.found);
    drawOverlay();
  }

  // ------------------------------------------------- hands-free word breaks
  // Lowering the hands for PAUSE_MS after a letter ends the word: a space is typed (and the
  // word optionally spoken), so a signer never has to reach for the keyboard.
  const PAUSE_MS = 2000;
  const pause = { since: null };
  function checkPause(found) {
    const out = els.output.value;
    if (found || !out || out.endsWith(" ")) { pause.since = null; return; }
    if (!$("auto-space").checked) return;
    const now = performance.now();
    if (pause.since === null) { pause.since = now; return; }
    if (now - pause.since < PAUSE_MS) return;
    pause.since = null;
    const word = out.trim().split(/\s+/).pop();
    els.output.value = out + " ";
    addHistory("␣");
    if ($("speak-words").checked && word) speak(word);
  }

  function speak(text) {
    if (!("speechSynthesis" in window)) { flash("This browser cannot speak"); return; }
    const clean = text.trim();
    if (!clean) { flash("Nothing to speak yet"); return; }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(clean.toLowerCase());
    u.lang = state.options.alphabet === "isl" ? "en-IN" : "en-US";
    u.rate = 0.95;
    window.speechSynthesis.speak(u);
  }
  $("btn-speak").addEventListener("click", () => speak(els.output.value));

  function labelFor(engine) {
    const names = {
      tesseract: "Tesseract", apple: "Apple Vision", windows: "Windows OCR", llm: "LLM",
      asl: "ASL model", isl: "ISL model", "asl-rules": "ASL rules", taught: "your taught signs",
    };
    return (engine || "").split("+").map((e) => names[e] || e).join(" + ");
  }

  function renderReadings(readings, found) {
    els.readings.innerHTML = "";
    if (!readings || !readings.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 4;
      td.className = "muted";
      td.textContent = isSign()
        ? (found ? "Hand found, nothing to compare yet" : "Show your hand to the camera")
        : (found ? "No engine ran" : "No letter-like shape in the guide box");
      tr.appendChild(td);
      els.readings.appendChild(tr);
      return;
    }
    for (const r of readings) {
      const tr = document.createElement("tr");
      let read = r.error ? "error" : (r.note && !r.text ? r.note : (r.text || "–"));
      if (r.note === "cached") read += " (cached)";
      const cells = [
        r.label,
        read,
        r.text ? `${Math.round(r.confidence * 100)}%${r.confidence_estimated ? "*" : ""}` : "–",
        r.latency_ms != null ? String(Math.round(r.latency_ms)) : "–",
      ];
      cells.forEach((c, i) => {
        const td = document.createElement("td");
        td.textContent = c;
        if (i === 1 && r.text) td.className = "read";
        if (i === 1 && r.raw) td.title = `candidates: ${r.raw}`;
        if (r.error) td.title = r.error;
        tr.appendChild(td);
      });
      els.readings.appendChild(tr);
    }
  }

  // ----------------------------------------------------------------- typing
  function typeCommit(text, replace) {
    if (!els.autotype.checked) return;
    const out = els.output;
    if (replace) { // J / Z: the hand first held I / the index shape, which was already typed
      out.value = out.value.slice(0, -replace);
      els.history.firstChild?.remove();
    }
    let insert = text;
    if (state.options.mode === "text" && out.value && !out.value.endsWith(" ")) insert = ` ${text}`;
    out.value += insert;
    out.setSelectionRange(out.value.length, out.value.length);
    out.scrollLeft = out.scrollWidth;
    out.dispatchEvent(new Event("input", { bubbles: true }));
    out.classList.remove("typed");
    void out.offsetWidth; // restart the CSS animation
    out.classList.add("typed");
    addHistory(text);
  }

  function addHistory(text) {
    const chip = document.createElement("span");
    chip.className = "hist";
    chip.textContent = text;
    els.history.prepend(chip);
    while (els.history.children.length > 24) els.history.lastChild.remove();
  }

  $("btn-space").addEventListener("click", () => { els.output.value += " "; els.output.focus(); });
  $("btn-back").addEventListener("click", () => { els.output.value = els.output.value.slice(0, -1); els.output.focus(); });
  $("btn-clear").addEventListener("click", () => {
    els.output.value = "";
    els.history.innerHTML = "";
    socket.emit("reset"); // also lets the letter still in view be typed again
  });
  $("btn-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(els.output.value);
      flash("Copied");
    } catch {
      els.output.select();
      flash("Press Ctrl/Cmd+C to copy");
    }
  });

  let flashTimer = null;
  function flash(msg) {
    els.statMsg.textContent = msg;
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => { els.statMsg.textContent = ""; }, 3500);
  }

  // ------------------------------------------------------------ teach signs
  function renderTaught(taught) {
    // taught: {counts: {letter: n}, hands: {letter: [1|2, ...]}}; undefined = just re-render
    if (taught) {
      state.taught = taught.counts || {};
      state.taughtHands = taught.hands || {};
    }
    for (const btn of els.letterGrid.querySelectorAll(".letter-btn")) {
      const n = state.taught[btn.dataset.letter] || 0;
      const two = (state.taughtHands[btn.dataset.letter] || []).includes(2);
      btn.querySelector("small").textContent = n ? String(n) : "";
      btn.classList.toggle("taught", n > 0);
      btn.classList.toggle("two-hand", two);
      btn.title = n ? `${n} examples, ${two ? "two-handed" : "one-handed"} sign` : "not taught";
      btn.classList.toggle("selected", btn.dataset.letter === state.teachLetter);
    }
    const letters = Object.keys(state.taught).length;
    els.teachTotal.textContent = letters ? `${letters} letter${letters > 1 ? "s" : ""} taught` : "none taught yet";
    els.teachRecord.textContent = state.recording ? "Stop" : `Record ${state.teachLetter}`;
    els.teachDelete.textContent = `Delete ${state.teachLetter}`;
    els.teachDelete.disabled = !state.taught[state.teachLetter];
  }

  els.letterGrid.addEventListener("click", (e) => {
    const btn = e.target.closest(".letter-btn");
    if (!btn || state.recording) return;
    state.teachLetter = btn.dataset.letter;
    renderTaught();
  });

  els.teachRecord.addEventListener("click", () => {
    if (guided) { stopGuided("Stopped"); return; }
    if (state.recording) { stopTeaching(); return; }
    if (!state.running) { flash("Start the camera first"); return; }
    socket.emit("teach_start", { letter: state.teachLetter, samples: Number(els.teachSamples.value) }, (res) => {
      if (res && res.error) { flash(res.error); return; }
      setRecording(res.recording);
    });
  });

  function stopTeaching() {
    socket.emit("teach_stop", null, (res) => res && renderTaught(res));
    setRecording(null);
  }

  function setRecording(rec) {
    state.recording = rec;
    els.teachProgress.hidden = !rec;
    els.recBanner.hidden = !rec;
    if (rec) {
      els.teachFill.style.width = `${Math.round((rec.done / rec.total) * 100)}%`;
      const kind = rec.hands === 2 ? "two-handed " : rec.hands === 1 ? "one-handed " : "";
      let tip = "hold the sign, move a little";
      if (rec.hands && rec.visible !== undefined && rec.visible !== rec.hands) {
        tip = rec.visible === 0 ? "show your hand" + (rec.hands === 2 ? "s" : "")
          : rec.hands === 2 ? "show both hands (paused)" : "only one hand please (paused)";
      }
      els.recBanner.textContent = `● Recording ${kind}“${rec.letter}”  ${rec.done}/${rec.total} — ${tip}`;
    }
    renderTaught();
  }

  function updateRecording(res) {
    const rec = res.recording;
    if (rec.finished) {
      setRecording(null);
      if (res.taught) renderTaught(res.taught);
      flash(`Learned ${rec.hands === 2 ? "two-handed " : ""}“${rec.letter}” from ${rec.total} frames`);
      if (guided) nextGuided(+1);
    } else {
      setRecording(rec);
    }
  }

  // ------------------------------------------ guided: teach every letter in turn
  // A countdown before each letter gives time to form the sign; recording then starts by
  // itself and moves on when done. Esc or the button stops it.
  const GUIDED_SAMPLES = 60; // ~6 s per letter; 60 frames clearly beat 30 in calibration tests
  const GUIDED_COUNTDOWN = 3;
  let guided = null; // {letters, i, timer}
  const guidedBtn = $("teach-guided");

  function startGuided() {
    if (!state.running) { flash("Start the camera first"); return; }
    const letters = [...els.letterGrid.querySelectorAll(".letter-btn")].map((b) => b.dataset.letter);
    const from = letters.indexOf(state.teachLetter);
    guided = { letters, i: Math.max(0, from), timer: null };
    guidedBtn.textContent = "Stop teaching";
    nextGuided(0);
  }

  function stopGuided(message) {
    if (!guided) return;
    clearInterval(guided.timer);
    guided = null;
    guidedBtn.textContent = "Teach all letters…";
    if (state.recording) stopTeaching();
    els.recBanner.hidden = true;
    if (message) flash(message);
  }

  function nextGuided(step) {
    if (!guided) return;
    guided.i += step;
    if (guided.i >= guided.letters.length) { stopGuided("All letters taught"); return; }
    const letter = guided.letters[guided.i];
    state.teachLetter = letter;
    renderTaught();
    let left = GUIDED_COUNTDOWN;
    const show = () => {
      els.recBanner.hidden = false;
      els.recBanner.textContent = `Get ready: sign “${letter}” — recording in ${left}…  (${guided.i + 1}/${guided.letters.length})`;
    };
    show();
    clearInterval(guided.timer);
    guided.timer = setInterval(() => {
      left -= 1;
      if (left > 0) { show(); return; }
      clearInterval(guided.timer);
      socket.emit("teach_start", { letter, samples: GUIDED_SAMPLES }, (res) => {
        if (res && res.error) { stopGuided(res.error); return; }
        setRecording(res.recording);
      });
    }, 1000);
  }

  guidedBtn.addEventListener("click", () => (guided ? stopGuided("Stopped") : startGuided()));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && guided) stopGuided("Stopped"); });

  els.teachDelete.addEventListener("click", () => {
    socket.emit("signs_delete", { letter: state.teachLetter }, (res) => res && renderTaught(res));
  });
  els.teachClear.addEventListener("click", () => {
    if (!Object.keys(state.taught).length) return;
    if (!window.confirm("Delete all taught signs?")) return;
    socket.emit("signs_clear", null, (res) => res && renderTaught(res));
  });

  // ---------------------------------------------------------------- overlay
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function drawOverlay() {
    const W = els.overlay.width, H = els.overlay.height;
    ctx.clearRect(0, 0, W, H);
    if (!state.running || !els.video.videoWidth) return;
    const vw = els.video.videoWidth, vh = els.video.videoHeight;
    const sx = W / vw, sy = H / vh;
    const dpr = window.devicePixelRatio || 1;
    const mirror = els.mirror.checked;
    // video px -> overlay px. The video element is flipped with CSS when mirrored; the
    // overlay is not (so labels stay readable), hence the manual x flip.
    const toScreen = (x, y, w = 0, h = 0) => ({ x: (mirror ? vw - x - w : x) * sx, y: y * sy, w: w * sx, h: h * sy });
    const res = state.lastResult;
    const accent = cssVar("--accent") || "#4f8cff";
    const good = cssVar("--good") || "#22c55e";
    ctx.save();
    ctx.font = `600 ${Math.round(16 * dpr)}px system-ui, sans-serif`;

    if (isSign()) {
      if (res && res.hands && state.lastRoi) drawHands(res, toScreen, dpr, res.text ? good : accent);
      ctx.restore();
      return;
    }

    const roi = captureRegion();
    const g = toScreen(roi.x, roi.y, roi.w, roi.h);
    ctx.fillStyle = "rgba(0, 0, 0, 0.38)";
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.rect(g.x, g.y, g.w, g.h);
    ctx.fill("evenodd");
    ctx.strokeStyle = res && res.progress >= 1 ? good : accent;
    ctx.lineWidth = 3 * dpr;
    drawCorners(g, Math.min(g.w, g.h) * 0.12);

    if (res && state.lastRoi && res.mode !== "sign") {
      const r = state.lastRoi;
      const boxes = [];
      if (res.mode === "letter" && res.glyph) boxes.push({ ...res.glyph, text: res.candidate || res.text });
      else {
        const src = (res.readings || []).find((rd) => rd.text && res.engine && res.engine.startsWith(rd.engine));
        (src ? src.boxes : []).forEach((b) => boxes.push(b));
      }
      for (const b of boxes) {
        const s = toScreen(r.x + b.x * r.w, r.y + b.y * r.h, b.w * r.w, b.h * r.h);
        ctx.strokeStyle = res.text ? good : "rgba(255,255,255,0.7)";
        ctx.lineWidth = 2 * dpr;
        ctx.strokeRect(s.x, s.y, s.w, s.h);
        if (b.text) label(b.text, s.x, s.y, res.text ? good : "rgba(0,0,0,0.6)", dpr);
      }
    }
    ctx.restore();
  }

  const HAND_COLORS = ["#22c55e", "#f59e0b"]; // first / second hand (left to right in the image)

  function drawHands(res, toScreen, dpr, color) {
    const r = state.lastRoi;
    const two = res.hands.length > 1;
    res.hands.forEach((hand, i) => {
      const pts = hand.points.map(([x, y]) => toScreen(r.x + x * r.w, r.y + y * r.h));
      ctx.strokeStyle = two ? HAND_COLORS[i % 2] : color;
      ctx.lineWidth = 3 * dpr;
      ctx.lineCap = "round";
      ctx.beginPath();
      for (const [a, b] of HAND_EDGES) {
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
      }
      ctx.stroke();
      ctx.fillStyle = "#fff";
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.5 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
      if (two && hand.side) { // which of the signer's hands this is, under the wrist
        const tag = hand.side === "right" ? "R" : "L";
        label(tag, pts[0].x - 8 * dpr, pts[0].y + 30 * dpr, two ? HAND_COLORS[i % 2] : color, dpr, 13);
      }
    });
    const shown = res.recording ? "" : (res.candidate || res.text);
    if (shown && res.hands.length) {
      const all = res.hands.flatMap((h) => h.points.map(([x, y]) => toScreen(r.x + x * r.w, r.y + y * r.h)));
      const left = Math.min(...all.map((p) => p.x));
      const top = Math.min(...all.map((p) => p.y));
      label(shown, left, top - 6 * dpr, color, dpr, 30);
    }
  }

  function label(text, x, y, bg, dpr, size = 16) {
    ctx.font = `700 ${Math.round(size * dpr)}px system-ui, sans-serif`;
    const pad = 6 * dpr;
    const tw = ctx.measureText(text).width + pad * 2;
    const th = (size + 10) * dpr;
    const top = Math.max(0, y - th);
    ctx.fillStyle = bg;
    ctx.fillRect(Math.max(0, x), top, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, Math.max(0, x) + pad, top + th * 0.74);
  }

  function drawCorners(r, len) {
    ctx.beginPath();
    for (const [cx, cy, dx, dy] of [
      [r.x, r.y, 1, 1], [r.x + r.w, r.y, -1, 1], [r.x, r.y + r.h, 1, -1], [r.x + r.w, r.y + r.h, -1, -1],
    ]) {
      ctx.moveTo(cx + dx * len, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * len);
    }
    ctx.stroke();
  }

  els.roi.addEventListener("input", drawOverlay);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) drawOverlay(); });
  renderTaught({ counts: {}, hands: {} });
})();
