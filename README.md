# Alphabet OCR

> [!WARNING]
> **This app is still in development, and its accuracy is uncertain.**
>
> Recognition of hand signs and printed letters has not been reliably measured and can vary
> a lot between people, hands, cameras, lighting and backgrounds. The accuracy figures
> below come from small, limited tests and are not a guarantee of real-world performance.
> Expect wrong letters, missed letters, bugs and breaking changes.
>
> Do not rely on it for important, medical, legal, emergency or other critical
> communication. It is not a replacement for a qualified sign-language interpreter.

Live webcam fingerspelling-to-text, built as an assistive tool for deaf and speech-impaired
people: sign letters with your hands (ISL or ASL) and they are typed, and can be spoken
aloud. Printed letters can be read too (OCR).

- **Server:** FastAPI + Jinja2 page + Socket.IO (python-socketio, ASGI)
- **Hand signs (default mode):** MediaPipe hand landmarks (up to two hands) feed a
  recogniser **trained on real photos** of the chosen alphabet: ISL (Indian) or ASL
  (American). Your own **taught signs** take priority, which is how to make it accurate for
  you.
- **Communication:** 🔊 Speak reads the typed text aloud. Lowering your hands for 2 s
  types a space, so no keyboard is needed.
- **Primary OCR (printed letters):** Tesseract 5, embedded in Python via `tesserocr` (no system install)
- **Platform OCR, only if available:** Apple Vision on macOS, Windows.Media.Ocr on
  Windows. When present it reads every frame alongside Tesseract. When missing or broken,
  it is skipped at startup and Tesseract works alone.
- **Optional:** an LLM arbiter (any OpenAI-compatible vision endpoint) for letters the OCR
  engines are unsure about

## Quick start

macOS / Linux:

```bash
./setup.sh      # venv, dependencies, downloads the hand model (8 MB) and Tesseract model (23 MB)
./run.sh        # prints the URL to open
```

Windows:

```bat
setup.bat
run.bat
```

Open the printed URL (browsers allow camera access only on `localhost` or HTTPS) and click
**Start camera**. Pick what to recognise with the buttons above the video:

- **✋ Hand signs:** sign a letter and hold it steady until the ring fills. The page
  shows the hand skeleton MediaPipe sees.
- **🔤 Printed letter / 📝 Printed words:** hold paper inside the guide box (OCR). Only
  letters on paper are read: faces, mouths, skin and background clutter are ignored (see
  below).

## Hand signs

### Trained recognisers

`models/isl_signs.npz` and `models/asl_signs.npz` are small neural networks over hand-pose
features. Those features are joint positions in palm coordinates, the hand's 3D
orientation, and how the two hands touch. They are trained with `scripts/train_signs.py`
on these public datasets, every photo mirrored for left-handed signers, plus random
variations (tilt, finger proportions, joint jitter):

| Alphabet | Training photos | "Not a sign" examples |
|---|---|---|
| ISL | Hemg/Indian_sign_language_dataset, akritRihal/Indian_Sign_Language_dataset | HaGRID `no_gesture` (relaxed hands of many people, CC BY-SA 4.0) |
| ASL | Marxulia/asl_sign_languages_alphabets_v03, suneetpaul/sign_language_image_dataset | same |

**Measured accuracy** (limited tests; real-world accuracy is uncertain, see the notice at
the top). These are on photos of people the model never saw, not on
near-copies of its training photos. Same-dataset held-out scores (92-99%) are misleading,
because those photos come from the same signer and session.

| Test | Typed right | Typed wrong | Nothing typed |
|---|---|---|---|
| ASL, 130 phone photos of other people (at the 70% typing threshold) | 79.7% | 8.6% | 11.7% |
| ASL, same photos, *previous hand-written rules* | 30.1% | 42.9% | 27.0% |
| Relaxed, non-signing hands (800 photos, HaGRID) | | 3.5% (ASL) / 3.4% (ISL) | rest |

**ISL caveat.** The public ISL photos come from very few signers. Two-handed letters whose
hands touch or overlap are also seen differently by the hand tracker depending on the
exact overlap. A model trained on one ISL dataset reads the other dataset at only 14-23%.
**Do not rely on the generic ISL model for a new person: run the guided calibration
below.** In simulations, 60 recorded frames per letter gave a signer 85-92% correct with
0-2% wrong letters.

Hand signs are typed only at ≥ 70% confidence (**Settings → Min confidence**). On unseen
photos this cut wrong letters from 12.5% to 8.6% for 1.5% fewer correct ones. A letter
must also be held for 3 frames.

**J and Z** (ASL) are drawn in the air. The app tracks the little finger (J) or index (Z)
over the last 1.6 s and replaces the static shape typed at the start of the motion.

Retrain or extend: `python scripts/train_signs.py --alphabet isl --cross` downloads the
datasets (about 3 GB, cached in `./datasets`), trains, and reports leave-one-source-out
accuracy. `python scripts/eval_signs.py --alphabet asl --folder <letter>/<photo>.jpg`
measures a model on your own labelled photos. Training needs `scikit-learn` and `pyarrow`
(in requirements-dev.txt); the app itself does not.

### One or two hands

**Settings → Hands** is **Two hands** by default: both hands are tracked and drawn (each in
its own colour, tagged L/R when MediaPipe is sure which is which). Two-handed signs can be
taught. The features include how the hands touch: every fingertip of one hand against every
fingertip of the other. That is what separates e.g. ISL vowels, where the index finger
touches a different fingertip of the other hand; with hands at varying distances and tilts
it lifts accuracy from 71-88% to 99%. **One hand** uses only the most prominent hand and
ignores the other.

### Only hands count

Nothing else is ever treated as a hand: not a face or mouth, not anything else in the room.

- MediaPipe's landmark model scores how much a detection really is a hand. The app requires
  0.7 (`HAND_PRESENCE`). On 84 face-only frames, including mouth close-ups, rotations and
  over/under-exposure, forcing the palm detector to fire produces a few "hands". All of
  them score below 0.5, while every real hand tested scores above it, including a fist
  held in front of a face.
- A hand must also be seen in two consecutive frames, at about the same place, before it is
  drawn or recognised. One-frame flickers never reach the letters.

### Teach your signs (strongly recommended)

The most reliable way to make it accurate for a particular person and camera:

- **Teach all letters…** walks through A–Z. For each letter it says "Get ready: sign A…
  3, 2, 1", records about 6 s (60 frames), and moves on. Esc stops it. Move your hand a
  little while recording (closer, further, slightly turned).
- Or pick one letter, press **Record**, and hold the sign.

Taught signs take priority over the trained model whenever they match well. A pose that
resembles nothing taught is not forced onto a letter. A sign recorded with both hands is
stored as two-handed (marked 2✋), and frames where a hand drops out are skipped.
Examples are saved in `data/signs.json`. **Only my taught signs** in Settings ignores the
trained model entirely (for a custom or regional alphabet).

The launcher (`python -m app`) uses port 8000. If 8000 is taken (for example by a Docker
container), it uses the next free port and prints it. To choose a port: `./run.sh --port 9000`.

### Is the platform OCR active?

The startup log and the chips at the top of the page show each engine's state:

```
INFO:     OCR primary   Tesseract     tesseract 5.5.4 (embedded via tesserocr)
INFO:     OCR secondary Apple Vision  Vision VNRecognizeTextRequest (...)
```

An engine counts as available when its OS feature exists, its Python bindings import, and
it passes a self-test at startup. If any of these fails, the log line says
`UNAVAILABLE - <reason>`. The engine-dependent options in Settings are then greyed out and
every frame uses Tesseract.

## How it works

Hand signs:

```
browser                                   server
-------                                   ------
webcam -> whole frame -> JPEG  --frame-->  MediaPipe hand landmarks (21 points per hand)
                                              -> palm-relative 3D pose
                                              -> built-in ASL templates   +   your taught signs (nearest neighbour)
                                              -> J / Z motion tracker
input box <- type <- ack {letter, skeleton,  stabiliser: hold N frames -> commit once
                         progress, commit}
```

Printed letters (OCR):

```
browser                                   server
-------                                   ------
webcam -> crop guide box -> JPEG  --frame-->  preprocess (OpenCV)
                                              |- flatten lighting, fix polarity
                                              |- find the glyph, crop it tightly
                                              v
                                          Tesseract (legacy + LSTM cross-check)
                                            + Apple Vision / Windows OCR, if available
                                              | engines disagree or unsure?  (optional)
                                              v
                                          LLM arbiter, in the background, cached per glyph
                                              v
input box <- type <- ack {letter, boxes,  stabiliser: hold N frames -> commit once
                         progress, commit}
```

- **One frame in flight.** The browser sends the next frame only after the previous one is
  answered, so slow OCR lowers the frame rate instead of building a backlog.
- **Stabiliser.** A letter is typed after it has been read in `hold_frames` consecutive
  frames. It is typed once per appearance. To type it again, take it out of view and bring
  it back. Frames with low confidence neither count towards the hold nor re-arm the letter.
- **Printed letter vs. printed words.** Letter mode isolates one glyph (square guide
  box). Word mode reads words across a wide guide box and types them separated by spaces.

### Only letters on paper

Eyes, nostrils and mouths are dark shapes on a lighter background, which is exactly what
OCR looks for. Without a guard, OCR "read" a letter on 79 of 84 face-only frames and would
have typed 23 of them. With **Settings → Only read letters on paper** (on by default):
**0 of 84**, with printed-letter accuracy unchanged (99.6-100%). The guard has four parts:

1. Faces are found with MediaPipe's face detector and masked out. Detections without
   skin-coloured pixels are dropped, because the face detector also "sees faces" in printed
   G, S and Q.
2. Shapes surrounded by skin (close-ups the face detector misses) are skipped.
3. Paper check: a letter's surroundings must be smooth and light. On letters they measure
   a spread of 15 or less; on faces, hair and clothes, 25-76.
4. The letter must be a readable size (at least 12% of the guide box) and not sit at the
   box's edge, where background objects such as a window frame end up.

Every OCR engine call is also time-limited (3 s). An engine that stalls twice in a row is
benched for 60 s while Tesseract carries on, so the live view never freezes.

### Engine strategies (Settings → Engines)

| Strategy | Behaviour |
|---|---|
| `consensus` (default) | Tesseract and the platform OCR on every frame; agreement boosts confidence |
| `fallback` | Tesseract first; the platform OCR only when Tesseract is unsure |
| `primary` | Tesseract only |
| `secondary` | Apple Vision / Windows OCR only |

Without platform OCR, every strategy runs as `primary`. The result's `strategy` field
reports what actually ran.

### Measured accuracy

`python scripts/benchmark.py` renders A–Z in 9 fonts under 4 conditions and runs them
through the full pipeline. The conditions are: clean; webcam-like (noise, blur, lighting
gradient, tilt); rough (heavier noise and blur, clutter at the frame edge); and inverted
(white on black). Results on an Apple Silicon Mac:

*These letters are computer-rendered, not real camera footage, so real-world accuracy is
uncertain and is likely lower.*

| Strategy | clean | webcam | rough | inverted | p50 latency |
|---|---|---|---|---|---|
| Tesseract only | 99.6% | 100% | 98.7% | 99.6% | ~9 ms |
| Apple Vision only | 94.4% | 94.4% | 95.3% | 94.4% | ~7 ms |
| fallback | 99.6% | 100% | 99.6% | 99.6% | ~10 ms |
| **consensus** (default) | **99.6%** | **100%** | **99.6%** | **99.6%** | ~11 ms |

What got Tesseract there (details in `app/ocr/tesseract.py`):

- The **legacy** classifier reads isolated glyphs far better than the LSTM in single-char
  mode (97% vs 85%). This needs the *standard* `tessdata` model, not `tessdata_fast` or
  `tessdata_best`.
- The LSTM gets a second opinion by reading the glyph **tiled three times** as a text line.
- The legacy **adaptive classifier is disabled**. Otherwise it learns from every frame, so
  results drift with the history of the stream.

Apple Vision's `accurate` level ignores single characters, so letter mode uses `fast`.

## LLM arbiter (optional)

It is used only when the engines disagree, read nothing, or fall below the confidence
threshold. It never blocks the live loop: the first doubtful frame starts a background
request, and later frames of the same held glyph reuse the cached answer. One held letter
costs one call. After 3 consecutive errors it pauses for 30 s.

Enable it by copying `.env.example` to `.env` and setting:

```ini
LLM_BASE_URL=http://192.168.0.226:8000/v1   # your gateway, or Ollama / LM Studio
LLM_API_KEY=change-me-gateway-key
LLM_MODEL=gemma                             # must accept images (e.g. Gemma 3 4B+)
```

## Configuration

All optional, via environment or `.env`: `HOST`, `PORT`, `OCR_SECONDARY`
(`auto|apple|windows|none`), `TESSDATA_DIR`, `TESSERACT_LANG`, `TESSERACT_CMD` (to use a
system tesseract through pytesseract instead), `WINDOWS_OCR_LANGUAGE`, `OCR_WORKERS`,
`HAND_MODEL`, `HAND_PRESENCE` (default 0.7), `FACE_MODEL`, `SIGNS_PATH` (where taught signs
are saved), `LLM_*`.

## API

| | |
|---|---|
| `GET /` | Live page |
| `GET /api/engines` | Engine availability |
| `GET /api/signs` | Taught examples per letter |
| `POST /api/ocr?mode=sign` | Recognise one image (raw JPEG/PNG body); `mode=sign\|letter\|text` |
| Socket.IO `frame` | `{id, image: <bytes>}` → ack with result + `commit` |
| Socket.IO `settings` | Partial options → ack with effective options |
| Socket.IO `reset` | Clear typing state |
| Socket.IO `teach_start` / `teach_stop` | `{letter, samples}`: record the next frames as that sign |
| Socket.IO `signs_delete` / `signs_clear` | Forget one letter / all taught signs |

```bash
curl --data-binary @hand.jpg   "http://localhost:8000/api/ocr?mode=sign"
curl --data-binary @letter.png "http://localhost:8000/api/ocr?mode=letter"
```

## Development

```bash
.venv/bin/python -m pytest          # 143 tests: hand signs, trained models, only-hands/only-letters, OCR, LLM (mock), HTTP, Socket.IO
.venv/bin/python scripts/benchmark.py
```

## Layout

```
app/
  main.py            FastAPI routes + Socket.IO events
  config.py          settings (env) and per-client options
  preprocess.py      lighting/polarity normalisation, glyph isolation
  letters.py         homoglyph / look-alike normalisation to A-Z
  stabilizer.py      per-frame readings -> deliberate keystrokes
  results.py         result types sent to the browser
  faces.py           face detector + skin check that keep faces out of OCR
  hands/             tracker.py (MediaPipe), gate.py (hands must persist), geometry.py (palm frame, features),
                     trained.py (ISL/ASL models), library.py (taught signs), motion.py (J/Z),
                     asl.py (old rule-based ASL, used only if no ASL model), recognizer.py
  ocr/               tesseract.py, apple.py, windows.py, llm.py, manager.py
  templates/index.html, static/app.js, static/style.css
scripts/             download_models.py, train_signs.py + eval_signs.py (sign models), benchmark.py (OCR)
models/              hand + face detectors (downloaded), isl_signs.npz / asl_signs.npz (trained)
tests/               handmodel.py (synthetic hand), faces.py (face-only frames), data/ (sample photos)
data/signs.json      your taught signs (created when you teach one)
```

## License

MIT, see [LICENSE](LICENSE).

This software is provided "as is", without warranty of any kind. It is a work in progress
with uncertain accuracy. The author and contributors are not liable for any damages or
consequences arising from its use, including wrong, missed or misread letters.

Third-party parts keep their own licences: the Socket.IO client in `app/static/vendor/`
(MIT), the MediaPipe and Tesseract models fetched by `setup.sh` (Apache 2.0), the sample
photos in `tests/data/` (Apache 2.0, see `SOURCES.txt`), and the public datasets used to
train the sign models (see the table under *Trained recognisers*).
