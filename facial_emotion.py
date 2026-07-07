"""
Facial Emotion Recognition
-------------------------------------
Adds a real facial-emotion model on top of the  webcam pipeline.

Model: trpakov/vit-face-expression
  - Vision Transformer fine-tuned on FER2013
  - 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
  - Chosen because a published comparison study benchmarked 10 HF
    facial-emotion ViTs and found this one generalizes best across
    datasets/scenarios (good thesis citation for your methodology).

Uses MediaPipe Face DETECTION (just a bounding box) to crop the face
before classification — NOT Face Mesh, which is 's job (468
landmarks for head pose / eye tracking). Keeping these decoupled now
makes the  fusion step a clean "combine 5 independent signals"
exercise instead of an untangling exercise.

Run:
    python vision/facial_emotion.py
Press 'q' to quit.
"""

import cv2
import time
import mediapipe as mp
from transformers import pipeline
from PIL import Image

MODEL_NAME = "trpakov/vit-face-expression"
INFER_EVERY_N_FRAMES = 5  # ViT on every single frame is slow on CPU — skip frames

EMOTION_COLORS = {
    "happy": (0, 200, 0),
    "neutral": (200, 200, 200),
    "sad": (255, 120, 0),
    "angry": (0, 0, 255),
    "fear": (0, 140, 255),
    "disgust": (0, 100, 100),
    "surprise": (255, 0, 255),
}


class FacialEmotionAnalyzer:
    """Importable wrapper used by the  orchestrator. Holds one
    persistent ViT classifier + MediaPipe face detector, and only runs
    the (slow) ViT every INFER_EVERY_N_FRAMES frames — caching the last
    label in between so callers always get a usable reading.

    The standalone main() below is unchanged (the  demo)."""

    def __init__(self):
        self.classifier = pipeline("image-classification", model=MODEL_NAME)
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6)
        self._frame_count = 0
        self._last = {"label": None, "conf": 0.0, "box": None}

    def analyze(self, frame_bgr):
        """Returns {label, conf, box}. box is (x1,y1,x2,y2) or None.
        Label persists between inference frames so it never goes blank
        mid-stream."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        self._frame_count += 1

        if not results.detections:
            self._last["box"] = None
            return dict(self._last)

        box = results.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, x1 + int(box.width * w))
        y2 = min(h, y1 + int(box.height * h))
        self._last["box"] = (x1, y1, x2, y2)

        face_crop = rgb[y1:y2, x1:x2]
        if face_crop.size > 0 and self._frame_count % INFER_EVERY_N_FRAMES == 0:
            preds = self.classifier(Image.fromarray(face_crop))
            top = max(preds, key=lambda p: p["score"])
            self._last["label"], self._last["conf"] = top["label"], float(top["score"])
        return dict(self._last)


def main():
    print(f"Loading {MODEL_NAME} ... (first run downloads ~330MB, then cached locally)")
    classifier = pipeline("image-classification", model=MODEL_NAME)
    print("Model loaded.\n")

    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_count = 0
    last_label, last_conf = "warming up...", 0.0
    prev_time = 0

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                det = results.detections[0]  # first/largest face only
                box = det.location_data.relative_bounding_box
                x1 = max(0, int(box.xmin * w))
                y1 = max(0, int(box.ymin * h))
                x2 = min(w, x1 + int(box.width * w))
                y2 = min(h, y1 + int(box.height * h))

                face_crop = rgb[y1:y2, x1:x2]

                if face_crop.size > 0 and frame_count % INFER_EVERY_N_FRAMES == 0:
                    pil_img = Image.fromarray(face_crop)
                    preds = classifier(pil_img)
                    top = max(preds, key=lambda p: p["score"])
                    last_label, last_conf = top["label"], top["score"]

                color = EMOTION_COLORS.get(last_label.lower(), (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{last_label} ({last_conf:.0%})", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, "NO FACE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(" Facial Emotion", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
