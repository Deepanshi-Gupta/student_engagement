"""
 — Head Pose + Eye State (blink rate / drowsiness)
---------------------------------------------------------
Uses MediaPipe Face Mesh (468 landmarks) — no HuggingFace model needed,
this is Google's own solution via `pip install mediapipe`.

Two independent signals come out of this script:
  1. Head pose (yaw/pitch via solvePnP) -> ATTENTIVE / DISTRACTED / AWAY
  2. Eye state (blink rate via Eye Aspect Ratio) -> ALERT / DROWSY

Both feed the  fusion module (weights 0.25 and 0.20).

Run:
    python vision/head_pose_eye.py
Press 'q' to quit.
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# --- Head pose setup -------------------------------------------------
# Generic 3D face model points (arbitrary units — standard solvePnP trick,
# doesn't need to match the actual student's face geometry).
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),            # Nose tip           -> landmark 1
    (0.0, -330.0, -65.0),       # Chin               -> landmark 152
    (-225.0, 170.0, -135.0),    # Left eye corner    -> landmark 33
    (225.0, 170.0, -135.0),     # Right eye corner   -> landmark 263
    (-150.0, -150.0, -125.0),   # Left mouth corner  -> landmark 61
    (150.0, -150.0, -125.0),    # Right mouth corner -> landmark 291
], dtype=np.float64)

LANDMARK_IDS_FOR_POSE = [1, 152, 33, 263, 61, 291]

# Degrees beyond which a head turn counts as "looking away" rather than
# normal micro-movement. Tune these after watching your own FPS/angle readout.
YAW_DISTRACTED_THRESHOLD = 40
PITCH_DISTRACTED_THRESHOLD = 30

# --- Eye state setup ---------------------------------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_BLINK_THRESHOLD = 0.21
BLINK_HISTORY_LEN = 90  # ~3 seconds of frames at 30fps


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(landmarks, eye_ids, w, h):
    p1, p2, p3, p4, p5, p6 = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_ids]
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return vertical / (2.0 * horizontal) if horizontal else 0.0


def estimate_head_pose(landmarks, w, h):
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h)
         for i in LANDMARK_IDS_FOR_POSE],
        dtype=np.float64,
    )

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

    pitch = float(euler_angles[0])
    yaw = float(euler_angles[1])
    roll = float(euler_angles[2])

    # decomposeProjectionMatrix reports pitch near +/-180 when facing the
    # camera: the 3D model has Y up (chin at -330) but image Y points down,
    # which is a 180-deg flip about X. Re-wrap so "looking straight ahead"
    # reads ~0 deg instead of ~180.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return yaw, pitch, roll

class HeadPoseEyeAnalyzer:
    """Importable wrapper around this  logic, used by the 
    orchestrator. It keeps a single persistent FaceMesh instance and a
    rolling blink history, and turns one BGR frame into two labels
    (head pose + eye state) the fusion module can consume.

    The standalone main() below is unchanged — it stays the  demo.
    This class is the same logic, refactored to return data instead of
    drawing it on screen."""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.blink_history = deque(maxlen=BLINK_HISTORY_LEN)

    def analyze(self, frame_bgr):
        """Returns a dict of signals for one frame. Keys are always
        present; values are None when no face is visible."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        out = {"face": False, "head_label": "AWAY", "eye_label": None,
               "yaw": None, "pitch": None, "roll": None,
               "ear": None, "closed_ratio": None}

        if not results.multi_face_landmarks:
            return out

        out["face"] = True
        landmarks = results.multi_face_landmarks[0].landmark

        yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
        if yaw is not None:
            out["yaw"], out["pitch"], out["roll"] = yaw, pitch, roll
            if abs(yaw) > YAW_DISTRACTED_THRESHOLD :
                out["head_label"] = "DISTRACTED"
            else:
                out["head_label"] = "ATTENTIVE"

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        self.blink_history.append(avg_ear < EAR_BLINK_THRESHOLD)
        closed_ratio = sum(self.blink_history) / len(self.blink_history) if self.blink_history else 0
        out["ear"] = round(avg_ear, 3)
        out["closed_ratio"] = round(closed_ratio, 3)
        out["eye_label"] = "DROWSY" if closed_ratio > 0.4 else "ALERT"
        return out

    def close(self):
        self.face_mesh.close()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    blink_history = deque(maxlen=BLINK_HISTORY_LEN)
    prev_time = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            attention_label = "AWAY"
            eye_label = "—"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
                if yaw is not None:
                    if abs(yaw) > YAW_DISTRACTED_THRESHOLD or abs(pitch) > PITCH_DISTRACTED_THRESHOLD:
                        attention_label = "DISTRACTED"
                    else:
                        attention_label = "ATTENTIVE"
                    cv2.putText(frame, f"yaw:{yaw:.1f} pitch:{pitch:.1f} roll:{roll:.1f}",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                blink_history.append(avg_ear < EAR_BLINK_THRESHOLD)

                closed_ratio = sum(blink_history) / len(blink_history) if blink_history else 0
                eye_label = "DROWSY" if closed_ratio > 0.4 else "ALERT"
                cv2.putText(frame, f"EAR:{avg_ear:.2f} closed_ratio:{closed_ratio:.0%}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            color = (0, 200, 0) if attention_label == "ATTENTIVE" else (0, 0, 255)
            cv2.putText(frame, f"HEAD: {attention_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            eye_color = (0, 200, 0) if eye_label == "ALERT" else (0, 140, 255)
            cv2.putText(frame, f"EYES: {eye_label}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, eye_color, 2)

            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(" Head Pose + Eye State", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
