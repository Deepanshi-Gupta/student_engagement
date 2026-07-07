"""
Week 6 — Fusion Module (your original research contribution)
-----------------------------------------------------------------
Combines the 5 independent signals built in Weeks 2-5 into a single
engagement_score (0-100) and a distress_flag (True/False), smoothed
over a rolling 10-second window.

Weights (document your reasoning for these choices in the thesis):
    head_pose          = 0.25
    facial_emotion       = 0.30
    eye_state              = 0.20
    speech_emotion           = 0.15
    confusion_keywords         = 0.10

This module is intentionally decoupled from the live camera/mic loops
in vision/ and speech/ — it just takes already-computed signal labels
and turns them into a score. That decoupling is what makes it
independently unit-testable (see __main__ below) and is exactly the
kind of clean "glue logic" a thesis committee can point to as your
original contribution, separate from any pre-trained model you used.

Wiring this up to the LIVE threads from Weeks 2-5 (so it runs on real
camera/mic input instead of simulated snapshots) is the next
integration step — natural to do once Week 7 (Gemini) needs real
engagement_vectors to reason over.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional


# --- Per-signal label -> sub-score mapping (0-100) -----------------------
# Design decisions — explain WHY in your thesis methodology section.
# e.g. "DISTRACTED" isn't 0 because a momentary head turn isn't full
# disengagement; "DROWSY" eyes score lower than "DISTRACTED" head pose
# because blink-rate alone is noisier evidence than head orientation.
HEAD_POSE_SCORES = {"ATTENTIVE": 100, "DISTRACTED": 40, "AWAY": 0}
EYE_STATE_SCORES = {"ALERT": 100, "DROWSY": 30}

# 7-class emotion -> engagement sub-score (shared by facial + speech emotion,
# since both models use the same 7 FER/RAVDESS-style label set).
EMOTION_SCORES = {
    "happy": 95, "surprise": 80, "neutral": 70,
    "sad": 35, "fear": 25, "angry": 20, "disgust": 15,
}

# Emotions that can additionally raise the distress flag, if confident enough.
DISTRESS_EMOTIONS = {"fear", "angry", "sad", "disgust"}
DISTRESS_EMOTION_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class SignalSnapshot:
    """One moment-in-time reading from each of the 5 streams.
    Any field can stay None if that stream hasn't produced a reading
    yet — e.g. speech updates every 4-5s, vision updates every frame,
    so they won't always tick in sync."""
    head_pose_label: Optional[str] = None       # "ATTENTIVE" | "DISTRACTED" | "AWAY"
    eye_state_label: Optional[str] = None        # "ALERT" | "DROWSY"
    facial_emotion: Optional[str] = None          # e.g. "happy"
    facial_emotion_conf: float = 0.0
    speech_emotion: Optional[str] = None            # e.g. "angry"
    speech_emotion_conf: float = 0.0
    confusion_detected: Optional[bool] = None         # from Week 4 transcription


class EngagementFusion:
    WEIGHTS = {
        "head_pose": 0.25,
        "facial_emotion": 0.30,
        "eye_state": 0.20,
        "speech_emotion": 0.15,
        "confusion_keywords": 0.10,
    }

    def __init__(self, window_seconds: int = 10, samples_per_second: float = 1.0):
        window_len = max(1, int(window_seconds * samples_per_second))
        self._history = deque(maxlen=window_len)

    def _score_snapshot(self, s: SignalSnapshot):
        """Returns (engagement_score_0_100, distress_bool) for one snapshot.
        Missing signals are excluded and the remaining weights
        renormalised — so the system degrades gracefully instead of
        crashing if, say, the mic stream hasn't produced a reading yet."""
        contributions = []  # (weight, sub_score)
        distress_votes = 0

        if s.head_pose_label is not None:
            contributions.append((self.WEIGHTS["head_pose"],
                                   HEAD_POSE_SCORES.get(s.head_pose_label, 50)))
            if s.head_pose_label == "AWAY":
                distress_votes += 1

        if s.facial_emotion is not None:
            contributions.append((self.WEIGHTS["facial_emotion"],
                                   EMOTION_SCORES.get(s.facial_emotion.lower(), 50)))
            if (s.facial_emotion.lower() in DISTRESS_EMOTIONS
                    and s.facial_emotion_conf >= DISTRESS_EMOTION_CONFIDENCE_THRESHOLD):
                distress_votes += 1

        if s.eye_state_label is not None:
            contributions.append((self.WEIGHTS["eye_state"],
                                   EYE_STATE_SCORES.get(s.eye_state_label, 50)))

        if s.speech_emotion is not None:
            contributions.append((self.WEIGHTS["speech_emotion"],
                                   EMOTION_SCORES.get(s.speech_emotion.lower(), 50)))
            if (s.speech_emotion.lower() in DISTRESS_EMOTIONS
                    and s.speech_emotion_conf >= DISTRESS_EMOTION_CONFIDENCE_THRESHOLD):
                distress_votes += 1

        if s.confusion_detected is not None:
            contributions.append((self.WEIGHTS["confusion_keywords"],
                                   0 if s.confusion_detected else 100))
            if s.confusion_detected:
                distress_votes += 1

        if not contributions:
            return None, False

        total_weight = sum(w for w, _ in contributions)
        score = sum(w * sc for w, sc in contributions) / total_weight
        distress = distress_votes >= 2  # require >=2 corroborating signals, not one noisy reading
        return score, distress

    def update(self, snapshot: SignalSnapshot):
        """Feed one new snapshot, get back the smoothed rolling result."""
        score, distress = self._score_snapshot(snapshot)
        if score is not None:
            self._history.append((score, distress))
        return self.current()

    def current(self):
        if not self._history:
            return {"engagement_score": None, "distress_flag": False, "samples": 0}
        scores = [sc for sc, _ in self._history]
        distress_count = sum(1 for _, d in self._history if d)
        # Majority-of-window rule: one bad second shouldn't trigger a
        # teacher alert by itself.
        distress_flag = (distress_count / len(self._history)) > 0.5
        return {
            "engagement_score": round(sum(scores) / len(scores), 1),
            "distress_flag": distress_flag,
            "samples": len(self._history),
        }


# ---------------------------------------------------------------------
# Unit tests / edge cases — run directly: python fusion/fusion.py
# Mirrors the original plan's Week 4 requirement: simulate edge cases
# (all distressed, all engaged, mixed) before trusting the module.
# ---------------------------------------------------------------------
if __name__ == "__main__":

    def run_case(name, snapshots):
        fusion = EngagementFusion(window_seconds=10, samples_per_second=1.0)
        result = None
        for snap in snapshots:
            result = fusion.update(snap)
        print(f"{name:35s} -> score={result['engagement_score']:>5}  distress={result['distress_flag']}")

    all_engaged = [
        SignalSnapshot(head_pose_label="ATTENTIVE", eye_state_label="ALERT",
                        facial_emotion="happy", facial_emotion_conf=0.9,
                        speech_emotion="neutral", speech_emotion_conf=0.7,
                        confusion_detected=False)
        for _ in range(10)
    ]

    all_distressed = [
        SignalSnapshot(head_pose_label="AWAY", eye_state_label="DROWSY",
                        facial_emotion="fear", facial_emotion_conf=0.8,
                        speech_emotion="angry", speech_emotion_conf=0.75,
                        confusion_detected=True)
        for _ in range(10)
    ]

    mixed = [
        SignalSnapshot(head_pose_label="ATTENTIVE", eye_state_label="ALERT",
                        facial_emotion="neutral", facial_emotion_conf=0.6,
                        speech_emotion=None, confusion_detected=False)
        for _ in range(5)
    ] + [
        SignalSnapshot(head_pose_label="DISTRACTED", eye_state_label="DROWSY",
                        facial_emotion="sad", facial_emotion_conf=0.65,
                        speech_emotion="sad", speech_emotion_conf=0.6,
                        confusion_detected=True)
        for _ in range(5)
    ]

    partial_signals = [  # only vision running; speech stream hasn't reported yet
        SignalSnapshot(head_pose_label="ATTENTIVE", eye_state_label="ALERT",
                        facial_emotion="happy", facial_emotion_conf=0.85)
        for _ in range(10)
    ]

    print("Fusion module edge-case tests")
    print("-" * 60)
    run_case("All engaged", all_engaged)
    run_case("All distressed", all_distressed)
    run_case("Mixed (engaged -> distressed)", mixed)
    run_case("Partial signals (vision only)", partial_signals)
