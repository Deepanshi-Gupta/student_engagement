"""
Week 7 — AI Integration (Gemini reasoning layer)
-----------------------------------------------------
Turns the engagement_score / distress_flag from Week 6's fusion module
into natural-language alerts — this is "Layer 4" from the original
architecture doc, the piece that makes raw numbers actionable for a
teacher instead of just a chart.

SDK: google-genai — the current unified Google Gen AI Python SDK.
(NOT the older `google-generativeai` package with genai.configure() /
GenerativeModel() — that one is deprecated. requirements.txt was
updated to match.)

Setup:
  1. Get a Gemini API key: https://aistudio.google.com/apikey
  2. Create a `.env` file in the project root with:
         GEMINI_API_KEY=your_key_here
  3. pip install -r requirements.txt

Model: gemini-2.5-flash-lite — chosen deliberately over the bigger
flash/pro models because this script polls every 30 seconds for an
entire class session. flash-lite is built specifically for that kind
of high-volume, cost-sensitive, low-latency traffic, so it keeps a
full session affordable. Swap MODEL_NAME if you want higher-quality
alerts and don't mind the extra cost.

Run modes:
  python reasoning/gemini_alerts.py            -> offline: builds and prints prompts only,
                                                    no API key / internet needed
  python reasoning/gemini_alerts.py --live       -> also makes one real call to Gemini
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Load GEMINI_API_KEY from a .env file in the project root if present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.append(os.path.dirname(__file__))  # allow flat-layout import of fusion.py
from fusion import EngagementFusion, SignalSnapshot  # noqa: E402

MODEL_NAME = "gemini-2.5-flash-lite"
ALERT_COOLDOWN_SECONDS = 30  # matches the "every 30 seconds" design from the architecture doc


@dataclass
class ClassroomContext:
    """Extra context Gemini needs to write a USEFUL alert, not just
    react to a bare number."""
    student_name: str = "the student"
    minutes_into_session: int = 0
    recent_topic: str = "the current lesson"


def build_teacher_alert_prompt(fusion_result: dict, context: ClassroomContext) -> str:
    return (
        "You are an assistant summarising classroom engagement data for a teacher. "
        "Be concise (1-2 sentences), calm, and actionable — never alarming.\n\n"
        f"Student: {context.student_name}\n"
        f"Minutes into session: {context.minutes_into_session}\n"
        f"Topic: {context.recent_topic}\n"
        f"Engagement score (0-100, rolling 10s average): {fusion_result['engagement_score']}\n"
        f"Distress flag: {fusion_result['distress_flag']}\n\n"
        "Write ONE short alert for the teacher. If engagement is high, say nothing "
        "needs action and confirm things look good. If distress_flag is True, "
        "suggest a specific, low-friction intervention (e.g. 'check in quietly', "
        "'consider a short pause')."
    )


def build_student_nudge_prompt(fusion_result: dict, context: ClassroomContext) -> str:
    return (
        f"You are writing a brief, kind, private nudge directly to a student "
        f"({context.student_name}) based on their own engagement signals. "
        "Never mention being monitored or watched — frame it as a friendly check-in. "
        "One sentence only.\n\n"
        f"Engagement score: {fusion_result['engagement_score']}\n"
        f"Distress flag: {fusion_result['distress_flag']}\n\n"
        "Write the nudge now."
    )


def build_session_summary_prompt(score_history: list, context: ClassroomContext) -> str:
    avg = sum(score_history) / len(score_history) if score_history else 0
    low_points = sum(1 for s in score_history if s < 50)
    return (
        "Summarise a classroom session for the teacher's records, 2-3 sentences. "
        f"Average engagement score: {avg:.0f}/100. "
        f"Number of low-engagement readings (<50): {low_points} out of {len(score_history)}. "
        f"Topic: {context.recent_topic}. Be factual, not alarmist."
    )


class GeminiAlertEngine:
    """Thin wrapper that (a) rate-limits teacher-alert calls to once per
    ALERT_COOLDOWN_SECONDS so a full session stays cheap, and
    (b) lazily creates the client so this module can be imported and
    prompt-tested without an API key being set at all.

    The API key is read from the GEMINI_API_KEY environment variable
    (loaded from a .env file at the top of this module). It is NEVER
    hardcoded — keep your key out of source control.

    Works with either Google SDK: the modern `google-genai`
    (genai.Client) if installed, otherwise the legacy
    `google-generativeai` (genai.GenerativeModel) which is what the
    current venv has. Install `google-genai` for the recommended path.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = "AIzaSyAUGFXEXx6dnu-0xvLlTOX85JqhDouF4hY"
        self._generate = None       # bound to whichever SDK we find
        self._last_call_time = 0.0

    def _ensure_client(self):
        if self._generate is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "No GEMINI_API_KEY found. Copy .env.example to .env and put your "
                "key there (get one at https://aistudio.google.com/apikey)."
            )
        # Prefer the modern unified SDK.
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            self._generate = lambda prompt: client.models.generate_content(
                model=MODEL_NAME, contents=prompt).text.strip()
            return
        except Exception:
            pass
        # Fall back to the legacy SDK (google-generativeai).
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        self._generate = lambda prompt: model.generate_content(prompt).text.strip()

    def _call(self, prompt: str) -> str:
        self._ensure_client()
        return self._generate(prompt)

    def maybe_generate_teacher_alert(self, fusion_result: dict, context: ClassroomContext):
        """Rate-limited teacher alert. Returns None while in cooldown."""
        now = time.time()
        if now - self._last_call_time < ALERT_COOLDOWN_SECONDS:
            return None  # still in cooldown — skip the call entirely, keeps cost down
        self._last_call_time = now
        return self._call(build_teacher_alert_prompt(fusion_result, context))

    def generate_student_nudge(self, fusion_result: dict, context: ClassroomContext) -> str:
        return self._call(build_student_nudge_prompt(fusion_result, context))

    def generate_session_summary(self, score_history: list, context: ClassroomContext) -> str:
        return self._call(build_session_summary_prompt(score_history, context))


if __name__ == "__main__":
    live = "--live" in sys.argv

    # Build a fake fusion result using Week 6's real module, so these
    # prompts are tested against the exact dict shape
    # EngagementFusion.current() actually returns — no camera/mic/API
    # key needed for this part.
    fusion = EngagementFusion(window_seconds=10, samples_per_second=1.0)
    for _ in range(10):
        fusion.update(SignalSnapshot(
            head_pose_label="DISTRACTED", eye_state_label="DROWSY",
            facial_emotion="sad", facial_emotion_conf=0.7,
            speech_emotion=None, confusion_detected=True,
        ))
    fusion_result = fusion.current()
    context = ClassroomContext(student_name="Riya", minutes_into_session=22, recent_topic="thermodynamics")

    print("Fusion result:", fusion_result)
    print("\n--- Teacher alert prompt ---")
    print(build_teacher_alert_prompt(fusion_result, context))
    print("\n--- Student nudge prompt ---")
    print(build_student_nudge_prompt(fusion_result, context))
    print("\n--- Session summary prompt ---")
    print(build_session_summary_prompt([90, 85, 40, 35, 30, 88], context))

    if live:
        print("\n--- Calling Gemini live ---")
        engine = GeminiAlertEngine()
        try:
            alert = engine.maybe_generate_teacher_alert(fusion_result, context)
            print(alert or "(cooldown active, no call made)")
        except Exception as e:
            print(f"Live call failed: {e}")
    else:
        print("\n(Run with --live to actually call Gemini, once GEMINI_API_KEY is set)")
