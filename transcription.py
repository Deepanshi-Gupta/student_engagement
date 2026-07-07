"""
Transcription + Confusion Keyword Detection
---------------------------------------------------------
Model: openai/whisper-small, via the `openai-whisper` pip package
(simpler for local mic-chunk transcription than the HF transformers
wrapper, and matches what the original plan specified).

Captures rolling 5-second audio chunks, transcribes them, and scans
the text for confusion markers ("huh?", "I don't understand", etc).
This becomes the confusion_keywords signal in the Week 6 fusion module
(weight 0.10).

Run:
    python speech/transcription.py
Ctrl+C to stop.
"""

import time
import numpy as np
import sounddevice as sd
import whisper

MODEL_SIZE = "small"   # use "base" if this is too slow on your CPU
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
SILENCE_RMS_THRESHOLD = 0.005

# Keep this list short and high-precision rather than exhaustive — a
# false-positive confusion flag pollutes the fusion score more than a
# missed one. Expand based on what you actually see in testing.
CONFUSION_KEYWORDS = [
    "huh", "what", "i don't understand", "i dont understand",
    "can you repeat", "say that again", "i'm confused", "im confused",
    "i'm lost", "im lost", "wait what", "sorry what", "come again",
    "not clear", "doesn't make sense", "no idea",
]


def contains_confusion_marker(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in CONFUSION_KEYWORDS)


class TranscriptionAnalyzer:
    """Importable wrapper used by the Week 8 orchestrator. Like the
    speech-emotion analyzer, it does NOT own the mic — the orchestrator
    hands it the same audio chunk it gave the emotion model, so both run
    on identical audio. Returns the transcript plus the confusion flag
    that becomes the 0.10-weight signal in fusion.

    The standalone main() below is unchanged (the Week 4 demo)."""

    def __init__(self, model_size=MODEL_SIZE):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio, sample_rate=SAMPLE_RATE):
        """Returns {text, confusion} or None for silence."""
        audio = np.asarray(audio, dtype="float32").flatten()
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        if rms < SILENCE_RMS_THRESHOLD:
            return None
        result = self.model.transcribe(audio, fp16=False, language="en")
        text = result["text"].strip()
        return {"text": text, "confusion": contains_confusion_marker(text)}


def main():
    print(f"Loading Whisper '{MODEL_SIZE}' model ... (first run downloads it, then cached)")
    model = whisper.load_model(MODEL_SIZE)
    print(f"Model loaded. Listening in {CHUNK_SECONDS}s windows. Ctrl+C to stop.\n")

    try:
        while True:
            audio = sd.rec(
                int(CHUNK_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()

            rms = float(np.sqrt(np.mean(audio**2)))
            timestamp = time.strftime("%H:%M:%S")

            if rms < SILENCE_RMS_THRESHOLD:
                print(f"[{timestamp}] (silence, skipped)")
                continue

            result = model.transcribe(audio, fp16=False, language="en")
            text = result["text"].strip()

            if not text:
                print(f"[{timestamp}] (no speech detected)")
                continue

            flagged = contains_confusion_marker(text)
            tag = "CONFUSION DETECTED" if flagged else "clear"
            print(f'[{timestamp}] "{text}"   [{tag}]')

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
