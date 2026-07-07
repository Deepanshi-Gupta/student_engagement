"""
Week 1 — Basic understanding & setup
-------------------------------------
Goal: confirm your webcam works AND that MediaPipe can find a face on it.
This does NOT do head pose or emotion yet (that's Week 5 and Week 2) —
it just proves the camera -> face-detection pipeline is alive.

Run:
    python vision/test_webcam.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions / index.")

    prev_time = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            # MediaPipe wants RGB, OpenCV gives BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                    )
                status = "FACE DETECTED"
                color = (0, 200, 0)
            else:
                status = "NO FACE"
                color = (0, 0, 255)

            # FPS counter — useful later when you stack 3 CV streams (Week 5-6)
            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now

            cv2.putText(frame, f"{status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Week 1 - Webcam + MediaPipe sanity check", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
