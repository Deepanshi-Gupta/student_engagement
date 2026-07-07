"""
Week 1 — Basic understanding & setup
-------------------------------------
Goal: confirm your microphone works and you can capture + save audio.
This does NOT do transcription or emotion yet (that's Week 3 and Week 4) —
it just proves the mic -> audio file pipeline is alive.

Run:
    python speech/test_mic.py
Records 5 seconds, saves to data/mic_test.wav, then prints volume levels
so you can SEE that it actually picked up your voice.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os

DURATION_SECONDS = 5
SAMPLE_RATE = 16000  # 16kHz matches what Whisper / Wav2Vec2 expect later
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "mic_test.wav")


def main():
    print(f"Recording {DURATION_SECONDS} seconds... speak now.")
    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # block until recording finishes
    print("Recording finished.")

    sf.write(OUTPUT_PATH, audio, SAMPLE_RATE)
    print(f"Saved to {os.path.abspath(OUTPUT_PATH)}")

    # Quick sanity readout: split into 0.5s chunks, print RMS volume per chunk.
    # If these are all near-zero, your mic input isn't being picked up.
    chunk_size = SAMPLE_RATE // 2
    print("\nVolume per 0.5s chunk (RMS):")
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        rms = float(np.sqrt(np.mean(chunk**2)))
        bar = "#" * int(rms * 200)
        print(f"  {i / SAMPLE_RATE:4.1f}s  {rms:.4f}  {bar}")


if __name__ == "__main__":
    main()
