"""
Week 8 — Orchestrator (the integration layer that connects everything)
-------------------------------------------------------------------------
This is where the four standalone weekly scripts stop being separate demos
and become ONE running system. The problem it solves: only one process can
own the webcam, and only one can own the mic — so the per-week scripts
(facial_emotion.py, head_pose_eye.py, speech_emotion.py, transcription.py)
can't all run at once. Here we open each device exactly once and fan the
data out:

    [ camera thread ]  one frame  ->  FacialEmotionAnalyzer (Week 2)
                                  ->  HeadPoseEyeAnalyzer   (Week 5)

    [ audio thread ]   one chunk  ->  SpeechEmotionAnalyzer (Week 3)
                                  ->  TranscriptionAnalyzer (Week 4)

    [ fusion thread ]  every 1s   ->  EngagementFusion      (Week 6)
                                  ->  SessionStore           (Week 8)

    [ observe thread ] every 30s  ->  ObservationEngine (NVIDIA, Week 7)
                                  ->  SessionStore (a plain-language note)

The FastAPI server (server.py) never touches a model or a device — it only
reads the SessionStore this orchestrator writes to.

Modes:
  live  — real camera + mic (default). Degrades gracefully: if the camera
          or mic can't open, that stream is simply absent and fusion uses
          whatever signals it has.
  sim   — no hardware, no heavy models. Feeds synthetic signals through the
          REAL fusion module so the whole UI/alert pipeline can be demoed
          (and tested) on any machine.
"""

import os
import time
import random
import threading

from fusion import EngagementFusion, SignalSnapshot
from observations import ObservationEngine
from session_store import SessionStore

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 5
FUSION_TICK_SECONDS = 1.0
OBSERVE_EVERY_SECONDS = 30      # how often to ask the model for a fresh observation


class EngagementMonitor:
    def __init__(self, store: SessionStore, mode: str = "live",
                 topic: str = None, student_name: str = "the student"):
        self.store = store
        self.mode = mode
        self.topic = topic or os.getenv("MONITOR_TOPIC", "the current lesson")
        self.student_name = student_name

        self.fusion = EngagementFusion(window_seconds=10, samples_per_second=1.0)
        self.engine = ObservationEngine()
        self._ai_enabled = bool(self.engine.api_key)

        self._stop = threading.Event()
        self._threads = []

        # Shared latest-frame for the camera thread -> not needed across
        # threads since vision thread both produces and stores frames.

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self.store.set_device_status(running=True)
        if self.mode == "sim":
            self._spawn(self._sim_loop)
        else:
            self._preload_models()   # warm imports in THIS thread (see method docstring)
            self._spawn(self._vision_loop)
            self._spawn(self._audio_loop)
            self._spawn(self._fusion_loop)
        if self._ai_enabled:
            self._spawn(self._observation_loop)   # own thread: a slow LLM call must not block fusion
        else:
            self.store.set_observation(error="Observations disabled — no NVIDIA_API_KEY set. "
                                             "Scores and video still work.")

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self.store.set_device_status(running=False)

    def _spawn(self, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self._threads.append(t)

    def _preload_models(self):
        """Import the heavy analyzer modules once, here in the main thread.
        transformers' lazy-import machinery is NOT thread-safe on first
        import — if the vision and audio threads both run
        `from transformers import pipeline` at the same moment, one of them
        crashes with "cannot import name 'pipeline'". Importing the modules
        once up front populates sys.modules so the threads just hit the
        cache. (No model weights download here — that happens lazily when
        each Analyzer is constructed inside its thread.)"""
        try:
            import facial_emotion, head_pose_eye, speech_emotion, transcription  # noqa: F401
        except Exception as e:
            self.store.set_observation(error=f"Model preload failed: {e}")

    # ------------------------------------------------------------------
    # Live: vision thread — one camera, two analyzers (Weeks 2 + 5)
    # ------------------------------------------------------------------
    def _vision_loop(self):
        try:
            import cv2
            from facial_emotion import FacialEmotionAnalyzer
            from head_pose_eye import HeadPoseEyeAnalyzer
        except Exception as e:
            self.store.set_device_status(camera_ok=False)
            self.store.set_observation(error=f"Vision import failed: {e}")
            return

        # On Windows the default MSMF backend is flaky/slow to open; DirectShow
        # is far more reliable. Fall back to the default backend otherwise.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.store.set_device_status(camera_ok=False)
            self.store.set_observation(error="Could not open webcam (index 0). "
                                             "Is another app using the camera?")
            return
        self.store.set_device_status(camera_ok=True)

        emotion = FacialEmotionAnalyzer()
        pose = HeadPoseEyeAnalyzer()

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                emo = emotion.analyze(frame)
                hp = pose.analyze(frame)

                self.store.update_signals(
                    head_label=hp["head_label"],
                    eye_label=hp["eye_label"],
                    facial_emotion=emo["label"],
                    facial_conf=emo["conf"],
                    yaw=hp["yaw"], pitch=hp["pitch"], ear=hp["ear"],
                    face_visible=hp["face"],
                )
                self._encode_frame(cv2, frame, emo, hp)
        finally:
            cap.release()
            pose.close()

    def _encode_frame(self, cv2, frame, emo, hp):
        """Draw a compact overlay and push the JPEG to the store for /video."""
        score = self.store.fusion.get("engagement_score")
        if emo.get("box"):
            x1, y1, x2, y2 = emo["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (29, 158, 117), 2)
        lines = [
            f"HEAD: {hp['head_label']}   EYES: {hp['eye_label'] or '-'}",
            f"FACE: {(emo['label'] or '-')}",
            f"ENGAGEMENT: {score if score is not None else '-'}",
        ]
        for i, txt in enumerate(lines):
            y = 30 + i * 30
            cv2.putText(frame, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.store.set_frame(buf.tobytes())

    # ------------------------------------------------------------------
    # Live: audio thread — one mic, two analyzers (Weeks 3 + 4)
    # ------------------------------------------------------------------
    def _audio_loop(self):
        try:
            import sounddevice as sd
            from speech_emotion import SpeechEmotionAnalyzer
            from transcription import TranscriptionAnalyzer
        except Exception as e:
            self.store.set_device_status(mic_ok=False)
            self.store.set_observation(error=f"Audio import failed: {e}")
            return

        try:
            speech = SpeechEmotionAnalyzer()
            transcriber = TranscriptionAnalyzer()
        except Exception as e:
            self.store.set_device_status(mic_ok=False)
            self.store.set_observation(error=f"Speech model load failed: {e}")
            return

        import queue as _queue
        import numpy as np

        # CONTINUOUS capture: an InputStream callback runs on PortAudio's own
        # thread and never stops filling this queue — even while we're busy
        # running Whisper on the previous chunk. That's the fix for the old
        # "record 5s, stop, process, record again" gap that dropped speech.
        audio_q = _queue.Queue()

        def _callback(indata, frames, time_info, status):
            audio_q.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=int(AUDIO_SAMPLE_RATE * 0.5), callback=_callback)
            stream.start()
        except Exception as e:
            self.store.set_device_status(mic_ok=False)
            self.store.set_observation(error=f"Mic capture failed: {e}")
            return
        self.store.set_device_status(mic_ok=True)

        need = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS   # samples per analysis window
        buf = []
        try:
            while not self._stop.is_set():
                try:
                    buf.append(audio_q.get(timeout=0.5))
                except _queue.Empty:
                    continue
                # Drain everything captured so far (including audio recorded
                # WHILE we were running the previous Whisper pass).
                try:
                    while True:
                        buf.append(audio_q.get_nowait())
                except _queue.Empty:
                    pass
                if sum(len(c) for c in buf) < need:
                    continue   # keep accumulating until we have a full window

                # Analyse the MOST RECENT window and drop older backlog. If the
                # CPU can't keep up with real time, captions stay current
                # (recent speech) instead of drifting further behind each pass.
                audio = np.concatenate(buf)[-need:]
                buf = []

                emo = speech.classify(audio, AUDIO_SAMPLE_RATE)        # None if silent
                trans = transcriber.transcribe(audio, AUDIO_SAMPLE_RATE)

                updates = {}
                if emo is not None:
                    updates["speech_emotion"] = emo["label"]
                    updates["speech_conf"] = emo["conf"]
                if trans is not None:
                    updates["confusion"] = trans["confusion"]
                if updates:
                    self.store.update_signals(**updates)
                if trans is not None and trans["text"]:
                    self.store.add_transcript(trans["text"])   # rolling speech-to-text log
        finally:
            stream.stop()
            stream.close()

    # ------------------------------------------------------------------
    # Live: fusion thread — combine signals every second (Weeks 6 + 7)
    # ------------------------------------------------------------------
    def _fusion_loop(self):
        while not self._stop.is_set():
            s = self.store.signals
            snap = SignalSnapshot(
                head_pose_label=s["head_label"],
                eye_state_label=s["eye_label"],
                facial_emotion=s["facial_emotion"],
                facial_emotion_conf=s["facial_conf"],
                speech_emotion=s["speech_emotion"],
                speech_emotion_conf=s["speech_conf"],
                confusion_detected=s["confusion"],
            )
            result = self.fusion.update(snap)
            self.store.set_fusion(result)
            self._stop.wait(FUSION_TICK_SECONDS)

    # ------------------------------------------------------------------
    # Sim: no hardware — synthetic signals through the REAL fusion module
    # ------------------------------------------------------------------
    def _sim_loop(self):
        self.store.set_device_status(camera_ok=False, mic_ok=False)
        # A scripted arc so the dashboard visibly moves: engaged -> drifting
        # -> distressed -> recovering, looping.
        phases = [
            dict(head="ATTENTIVE", eye="ALERT", face="happy", fc=0.9, sp="neutral", spc=0.7, conf=False),
            dict(head="ATTENTIVE", eye="ALERT", face="neutral", fc=0.7, sp=None, spc=0.0, conf=False),
            dict(head="DISTRACTED", eye="ALERT", face="sad", fc=0.6, sp="sad", spc=0.6, conf=False),
            dict(head="DISTRACTED", eye="DROWSY", face="fear", fc=0.75, sp="angry", spc=0.7, conf=True),
            dict(head="AWAY", eye="DROWSY", face="angry", fc=0.8, sp="angry", spc=0.75, conf=True),
            dict(head="DISTRACTED", eye="ALERT", face="neutral", fc=0.65, sp="neutral", spc=0.6, conf=False),
        ]
        i = 0
        transcripts = ["", "could you explain that part again?",
                       "i don't understand the last step", "wait, what?", "ok that makes sense"]
        while not self._stop.is_set():
            p = phases[(i // 4) % len(phases)]
            # jitter confidence a little so it looks live
            self.store.update_signals(
                head_label=p["head"], eye_label=p["eye"],
                facial_emotion=p["face"], facial_conf=round(p["fc"] + random.uniform(-0.05, 0.05), 2),
                speech_emotion=p["sp"], speech_conf=p["spc"],
                confusion=p["conf"],
                face_visible=p["head"] != "AWAY",
                yaw=round(random.uniform(-30, 30), 1),
            )
            if i % 4 == 0:   # add a synthetic spoken line every few ticks
                self.store.add_transcript(transcripts[(i // 4) % len(transcripts)])
            snap = SignalSnapshot(
                head_pose_label=p["head"], eye_state_label=p["eye"],
                facial_emotion=p["face"], facial_emotion_conf=p["fc"],
                speech_emotion=p["sp"], speech_emotion_conf=p["spc"],
                confusion_detected=p["conf"],
            )
            result = self.fusion.update(snap)
            self.store.set_fusion(result)
            i += 1
            self._stop.wait(FUSION_TICK_SECONDS)

    # ------------------------------------------------------------------
    # Observation thread — NVIDIA model comments on the overall feature set
    # (shared by live + sim). Runs on its own clock so a slow model call
    # never stalls the per-second fusion loop.
    # ------------------------------------------------------------------
    def _observation_loop(self):
        self._stop.wait(8)   # let a few fusion samples accumulate first
        while not self._stop.is_set():
            fr = self.store.fusion
            if fr.get("engagement_score") is not None:
                s = self.store.signals
                features = {
                    "head_label": s["head_label"], "eye_label": s["eye_label"],
                    "facial_emotion": s["facial_emotion"], "facial_conf": s["facial_conf"],
                    "speech_emotion": s["speech_emotion"], "speech_conf": s["speech_conf"],
                    "confusion": s["confusion"], "transcript": s["transcript"],
                    "score": fr.get("engagement_score"), "distress": fr.get("distress_flag"),
                    "minutes": int(self.store.elapsed_seconds() // 60), "topic": self.topic,
                }
                try:
                    text = self.engine.observe(features)
                    self.store.set_observation(text=text, error=None)
                except Exception as e:
                    self.store.set_observation(error=f"Observation model call failed: {e}")
            self._stop.wait(OBSERVE_EVERY_SECONDS)
