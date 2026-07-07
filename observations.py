"""
Week 7 — AI Reasoning layer (NVIDIA), replaces the Gemini alert engine
------------------------------------------------------------------------
Takes the full set of multimodal input features (all 5 signals + the fused
engagement score) and asks an NVIDIA-hosted LLM for a short, plain-language
**observation** about the student's overall state. This is "Layer 4" — the
piece that turns raw numbers into something a human reads.

Uses the same NVIDIA endpoint as nvidia.py (the OpenAI-compatible
`integrate.api.nvidia.com` API).

Model choice
  Default: meta/llama-3.1-8b-instruct — responds in ~2-3s, which is what a
  live, periodically-refreshing panel needs. nvidia.py's
  `nvidia/nemotron-3-ultra-550b-a55b` also works through this engine but
  takes ~110s per call (it's a giant reasoning model), so it's impractical
  here. Override with the NVIDIA_MODEL env var to use any other model.

Key
  Read from NVIDIA_API_KEY (.env), falling back to the key used in nvidia.py
  so it runs out of the box. Move it to .env and rotate it for real use.

Run standalone (one observation from a sample feature set):
    python observations.py
"""

import os
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
_FALLBACK_KEY = "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"


def _fmt(features: dict) -> str:
    """Render the feature dict into a compact, readable block for the prompt."""
    def pct(x):
        try:
            return f" ({float(x):.0%} conf)"
        except Exception:
            return ""
    s = features
    lines = [
        f"Lesson topic: {s.get('topic') or 'the current lesson'}",
        f"Minutes into session: {s.get('minutes', 0)}",
        f"Head pose / attention: {s.get('head_label') or 'unknown'}",
        f"Eye state: {s.get('eye_label') or 'unknown'}",
        f"Facial emotion: {(s.get('facial_emotion') or 'unknown')}{pct(s.get('facial_conf'))}",
        f"Speech emotion: {(s.get('speech_emotion') or 'no speech yet')}"
        f"{pct(s.get('speech_conf')) if s.get('speech_emotion') else ''}",
        f"Confusion keywords detected: {s.get('confusion')}",
        f"Recent words said: \"{s.get('transcript') or '(none)'}\"",
        f"Fused engagement score: {s.get('score')}/100",
        f"Distress flag: {s.get('distress')}",
    ]
    return "\n".join(lines)


def build_observation_messages(features: dict) -> list:
    system = (
        "You are an attentive teaching assistant observing ONE student during a "
        "live lesson. You receive multimodal signals (vision, eye state, facial "
        "and speech emotion, spoken words, and a fused engagement score). Write a "
        "brief OBSERVATION — 1 to 2 calm, plain sentences — describing the "
        "student's overall state right now and what it suggests, grounded "
        "specifically in the signals given. If signals conflict, name the dominant "
        "pattern. Do not use lists or headings. Do not mention cameras, monitoring, "
        "sensors, or AI. Do not invent signals that aren't provided."
    )
    user = "Current signals:\n" + _fmt(features) + "\n\nWrite the observation now."
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


class ObservationEngine:
    """Generates a natural-language observation from the feature set, via the
    NVIDIA OpenAI-compatible API. Stateless apart from the lazily-created
    client — the orchestrator paces how often observe() is called."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY") or _FALLBACK_KEY
        self.model = model or os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("No NVIDIA_API_KEY set.")
            from openai import OpenAI
            self._client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=self.api_key)
        return self._client

    def observe(self, features: dict) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=build_observation_messages(features),
            temperature=0.6, top_p=0.95, max_tokens=220, timeout=60,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if not text:  # some reasoning models put the answer in reasoning_content
            text = (getattr(msg, "reasoning_content", "") or "").strip()
        return text


if __name__ == "__main__":
    engine = ObservationEngine()
    sample = dict(head_label="DISTRACTED", eye_label="DROWSY",
                  facial_emotion="angry", facial_conf=0.85,
                  speech_emotion="sad", speech_conf=0.6, confusion=True,
                  transcript="i don't understand the last step",
                  score=29.3, distress=False, minutes=12, topic="thermodynamics")
    print(f"Model: {engine.model}")
    t = time.time()
    print("Observation:", engine.observe(sample))
    print(f"({time.time() - t:.1f}s)")
