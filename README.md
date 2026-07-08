# Real-Time Multimodal Student Engagement & Distress Monitor

 Each week adds one piece; nothing is trained from
scratch — everything is a pre-trained model wired together by a fusion
module that **is** the original research contribution.

The system watches a student via webcam and listens via mic in parallel,
turning five raw signals into one engagement score (0–100) and a distress
flag, then uses an NVIDIA-hosted LLM to turn that into a plain-language
observation about the student's overall state.

## Architecture — 4 layers

The pipeline runs input → output through four layers. Every folder in the
repo maps to a layer.

```
Layer 1 — Vision        3 parallel CV streams on every webcam frame
Layer 2 — Speech        async transcription + speech emotion on mic audio
Layer 3 — Fusion        weighted combine of 5 signals → score + distress flag
Layer 4 — Reasoning     NVIDIA LLM observation →  FastAPI endpoints  →  Docker
```

The **five signals** that feed the fusion module:

| # | Signal | Source | Layer |
|---|---|---|---|
| 1 | Head pose (yaw/pitch/roll) → attentive/distracted/away | MediaPipe Face Mesh | Vision |
| 2 | Facial emotion (7 classes) | `trpakov/vit-face-expression` | Vision |
| 3 | Eye state — blink rate + gaze direction | MediaPipe eye landmarks | Vision |
| 4 | Speech emotion (frustrated/calm/distressed/…) | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Speech |
| 5 | Confusion keywords ("huh?", "I don't understand") | `openai/whisper-small` transcript | Speech |

## Setup (do this first)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll need:
- A working webcam
- A working microphone
- (Week 7) An NVIDIA API key — set as `NVIDIA_API_KEY` in a `.env` file
  (get one at https://build.nvidia.com; a fallback key is baked in so it
  also runs without setup)

## Folder structure

Each folder is one layer; the week tags show when that folder gets built.

```
engagement_monitor/
├── requirements.txt
├── vision/         # Layer 1 — facial emotion (W2), head pose + eye/gaze (W5)
├── speech/         # Layer 2 — speech emotion (W3), transcription (W4)
├── fusion/         # Layer 3 — combines all 5 signals into one score (W6)  ← original contribution
├── reasoning/      # Layer 4 — NVIDIA LLM observation generation (W7)
├── api/            # Layer 4 — FastAPI endpoints, Docker, GCP Cloud Run (W8)
├── notebooks/      # scratch work + ablation study (W6)
└── data/           # local test audio/video — never commit real student data
```

## Week 1 — Basic understanding & setup (you are here)

Run the two sanity-check scripts. Both should run with zero errors before
you touch any model:

```bash
python vision/test_webcam.py      # press 'q' to quit
python speech/test_mic.py          # records 5s, prints volume bars
```

If `test_webcam.py` shows "NO FACE" even when you're in frame, check
lighting and camera angle before moving on — every later layer depends
on this working.

## Full roadmap

| Week | Layer | Focus | Model(s) |
|---|---|---|---|
| 1 | — | Basic understanding & setup | — |
| 2 | Vision | Facial emotion | `trpakov/vit-face-expression` (ViT, FER2013, 7 classes) |
| 3 | Speech | Speech emotion | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| 4 | Speech | Transcription | `openai/whisper-small` (or `whisper-base` for speed) |
| 5 | Vision | Head pose / eye movement | MediaPipe Face Mesh (468 landmarks, no download) |
| 6 | Fusion | Fusion + ablation study | custom `fusion.py` — your original contribution |
| 7 | Reasoning | AI integration | NVIDIA API (`observations.py`) — observation generation |
| 8 | Reasoning | Deployment | FastAPI + Docker + GCP Cloud Run |
| 9 | — | Thesis write-up + conference paper draft | — |

By the end of Week 5 all five signals exist independently. Week 6 is where
they become one number — that fusion step, its weighting schema, and the
ablation study around it are the research contribution.

## Running the whole project (Week 8)

The four weekly scripts each grab the camera/mic on their own, so they
can't all run at once. The integration layer fixes that: **one** camera
loop and **one** mic loop fan their data out to all the analyzers, feed a
single fusion instance, and publish to a web dashboard.

```bash
# 1. (once) copy the env template and add your NVIDIA key (optional)
cp .env.example .env            # Windows: copy .env.example .env

# 2. run it
python run.py                   # live — real webcam + mic
python run.py --sim             # simulation — no hardware, synthetic signals
```

Then open **http://127.0.0.1:8000**. The dashboard shows the live
engagement gauge, all 5 signals, a score-over-time chart, the annotated
camera feed, and the NVIDIA model's observations on the student's state.

> Try `--sim` first: it drives the **real** fusion + alert pipeline with
> scripted signals, so you can see the entire UI working before depending
> on camera/mic/model downloads.

### Integration files (the glue, all importable)

| File | Role |
|---|---|
| `session_store.py` | thread-safe shared state — **stores** every signal, score, alert, frame |
| `orchestrator.py` | runs camera + mic loops, **passes** data into fusion + NVIDIA observation |
| `server.py` | FastAPI app — endpoints + serves the dashboard |
| `static/dashboard.html` | the live UI — **displays** everything |
| `run.py` | single launch command |

Each weekly script (`facial_emotion.py`, `head_pose_eye.py`,
`speech_emotion.py`, `transcription.py`) now also exposes an `Analyzer`
class the orchestrator imports — so the standalone demos **and** the
integrated app share the exact same logic.

API endpoints: `GET /api/state` (live JSON), `GET /video` (MJPEG feed),
`GET /api/report` (session summary).

## Fusion weights (design decision — document your reasoning in the thesis)

```
head_pose          = 0.25
facial_emotion     = 0.30
eye_state          = 0.20
speech_emotion     = 0.15
confusion_keywords = 0.10
```
Rolling 10-second window average to smooth noise. These weights are
configurable and are the variable swept in the Week 6 ablation study
(vision-only vs speech-only vs full fusion).
