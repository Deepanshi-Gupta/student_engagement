"""
Week 3 — Speech Emotion Recognition
-------------------------------------
Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
  - Wav2Vec2 (large-xlsr-53 English) fine-tuned on RAVDESS + SAVEE + TESS
  - 7 classes: angry, disgust, fear, happy, neutral, sad, surprise

Captures rolling audio chunks from the mic and classifies the emotion
in each chunk independently. This stream runs separately from the
vision stream (Week 2, 5) — they get combined in Week 6 (fusion).

Run:
    python speech/speech_emotion.py
Ctrl+C to stop.
"""

import time
import threading
import numpy as np
import sounddevice as sd
from transformers import pipeline

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SAMPLE_RATE = 16000   # must match what the model was trained on
CHUNK_SECONDS = 4      # length of the analysis window
HOP_SECONDS = 1        # how often we slide the window forward and re-predict
SILENCE_RMS_THRESHOLD = 0.005  # below this, skip — avoids confident predictions on dead air


class SpeechEmotionAnalyzer:
    """Importable wrapper used by the Week 8 orchestrator. The
    orchestrator owns the mic (one capture loop shared with
    transcription), so this class only does classification on an audio
    array handed to it — it does NOT touch the microphone itself.

    The standalone main() below is unchanged (the Week 3 demo)."""

    def __init__(self):
        self.classifier = pipeline("audio-classification", model=MODEL_NAME)

    def classify(self, audio, sample_rate=SAMPLE_RATE):
        """Returns {label, conf, breakdown} or None for silence."""
        audio = np.asarray(audio, dtype="float32").flatten()
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < SILENCE_RMS_THRESHOLD:
            return None
        preds = self.classifier(audio, sampling_rate=sample_rate)
        top = max(preds, key=lambda p: p["score"])
        return {
            "label": top["label"],
            "conf": float(top["score"]),
            "breakdown": {p["label"]: round(float(p["score"]), 3) for p in preds},
        }


def main():
    print(f"Loading {MODEL_NAME} ... (first run downloads ~1.2GB, then cached locally)")
    classifier = pipeline("audio-classification", model=MODEL_NAME)
    print(
        f"Model loaded. Rolling {CHUNK_SECONDS}s window, re-predicting every "
        f"{HOP_SECONDS}s. Ctrl+C to stop.\n"
    )

    # Ring buffer holding the most recent CHUNK_SECONDS of audio. The mic
    # callback keeps pushing new frames in on its own thread while the main
    # loop reads a snapshot every HOP_SECONDS — so the window slides forward
    # continuously instead of waiting for a fresh 4s block each time.
    window_len = int(CHUNK_SECONDS * SAMPLE_RATE)
    ring = np.zeros(window_len, dtype="float32")
    lock = threading.Lock()

    def on_audio(indata, frames, time_info, status):
        nonlocal ring
        chunk = indata[:, 0]
        with lock:
            ring = np.roll(ring, -frames)
            ring[-frames:] = chunk

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=on_audio,
        ):
            while True:
                time.sleep(HOP_SECONDS)

                with lock:
                    audio = ring.copy()

                rms = float(np.sqrt(np.mean(audio**2)))
                timestamp = time.strftime("%H:%M:%S")

                if rms < SILENCE_RMS_THRESHOLD:
                    print(f"[{timestamp}] (silence, skipped)")
                    continue

                preds = classifier(audio, sampling_rate=SAMPLE_RATE)
                top = max(preds, key=lambda p: p["score"])
                breakdown = "  ".join(f"{p['label']}:{p['score']:.0%}" for p in preds)
                label = top["label"].upper()
                conf = top["score"]
                print(f"[{timestamp}] -> {label} ({conf:.0%})   | {breakdown}")
                print(f"    Detection: speaker sounds {top['label']} ({conf:.0%} confident)")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
