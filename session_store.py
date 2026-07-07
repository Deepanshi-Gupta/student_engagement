"""
Week 8 — Session Store (shared state: store + pass information)
-------------------------------------------------------------------
One thread-safe object that the orchestrator WRITES to and the FastAPI
server READS from. This is the seam that lets the heavy ML loops (camera,
mic, fusion, Gemini) run in background threads while the web layer just
reads the latest snapshot — neither side blocks the other.

Nothing here imports cv2/torch/whisper, so it's cheap to import and easy
to reason about. It holds:

  * the latest reading from each of the 5 signals
  * the latest fused engagement result (score + distress flag)
  * a rolling history of scores for the live chart
  * the most recent Gemini alert / nudge / summary
  * the latest annotated camera frame (JPEG bytes) for the video feed
  * session metadata (mode, uptime, which devices came up)
"""

import time
import threading
from collections import deque


# How many (timestamp, score) points to keep for the live chart.
# ~1 sample/sec, so 600 ≈ the last 10 minutes.
HISTORY_MAXLEN = 600


class SessionStore:
    def __init__(self, mode: str = "live"):
        self._lock = threading.Lock()
        self._start_time = time.time()

        self.mode = mode                 # "live" | "sim"
        self.running = False
        self.camera_ok = None            # True/False once the camera loop has tried
        self.mic_ok = None               # True/False once the audio loop has tried

        # Latest per-signal readings (None until a stream reports).
        self.signals = {
            "head_label": None,          # ATTENTIVE | DISTRACTED | AWAY
            "eye_label": None,           # ALERT | DROWSY
            "facial_emotion": None,      # happy | sad | ...
            "facial_conf": 0.0,
            "speech_emotion": None,      # happy | angry | ...
            "speech_conf": 0.0,
            "confusion": None,           # bool — confusion keyword detected
            "transcript": "",            # last transcribed line
            "yaw": None, "pitch": None, "ear": None,
            "face_visible": False,
        }

        # Latest fused result, shape matches EngagementFusion.current().
        self.fusion = {"engagement_score": None, "distress_flag": False, "samples": 0}

        # Rolling score history for the chart: list of {"t": seconds, "score": float}.
        self._history = deque(maxlen=HISTORY_MAXLEN)

        # Rolling speech-to-text log: list of {"t": seconds, "text": str}.
        self._transcript_log = deque(maxlen=50)

        # Generative-AI output: a single rolling "observation" from the model.
        self.observation = {
            "text": None,
            "at": None,                  # session-seconds when produced
            "error": None,               # last model error, surfaced to the UI
        }

        self._frame_jpeg = None          # bytes — latest annotated frame

    # --- time --------------------------------------------------------
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    # --- writers (called by the orchestrator threads) ----------------
    def update_signals(self, **kwargs):
        with self._lock:
            self.signals.update(kwargs)

    def add_transcript(self, text: str):
        """Append one transcribed line to the rolling speech-to-text log
        (skips blanks and consecutive duplicates) and set it as the latest."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            if self._transcript_log and self._transcript_log[-1]["text"] == text:
                return
            self._transcript_log.append({"t": round(self.elapsed_seconds(), 1), "text": text})
            self.signals["transcript"] = text

    def set_fusion(self, result: dict):
        with self._lock:
            self.fusion = dict(result)
            score = result.get("engagement_score")
            if score is not None:
                self._history.append({"t": round(self.elapsed_seconds(), 1), "score": score})

    def set_observation(self, *, text=None, error=None):
        with self._lock:
            if text is not None:
                self.observation["text"] = text
                self.observation["at"] = round(self.elapsed_seconds(), 1)
            self.observation["error"] = error  # set or clear

    def set_frame(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame_jpeg = jpeg_bytes

    def set_device_status(self, *, camera_ok=None, mic_ok=None, running=None):
        with self._lock:
            if camera_ok is not None:
                self.camera_ok = camera_ok
            if mic_ok is not None:
                self.mic_ok = mic_ok
            if running is not None:
                self.running = running

    # --- readers (called by the FastAPI layer) -----------------------
    def get_frame(self):
        with self._lock:
            return self._frame_jpeg

    def state_dict(self) -> dict:
        """Everything the dashboard needs for one poll."""
        with self._lock:
            return {
                "mode": self.mode,
                "running": self.running,
                "uptime_seconds": round(self.elapsed_seconds(), 1),
                "camera_ok": self.camera_ok,
                "mic_ok": self.mic_ok,
                "signals": dict(self.signals),
                "fusion": dict(self.fusion),
                "observation": dict(self.observation),
                "history": list(self._history)[-120:],  # last ~2 min for the chart
                "transcript_log": list(self._transcript_log)[-15:],  # recent spoken lines
            }

    def report_dict(self) -> dict:
        """Session summary stats for /api/report."""
        with self._lock:
            scores = [p["score"] for p in self._history]
            n = len(scores)
            avg = round(sum(scores) / n, 1) if n else None
            low = sum(1 for s in scores if s < 50)
            return {
                "mode": self.mode,
                "duration_seconds": round(self.elapsed_seconds(), 1),
                "samples": n,
                "average_engagement": avg,
                "min_engagement": round(min(scores), 1) if n else None,
                "max_engagement": round(max(scores), 1) if n else None,
                "low_engagement_readings": low,
                "distress_currently": self.fusion.get("distress_flag", False),
                "last_observation": self.observation.get("text"),
            }
