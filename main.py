import cv2
import mediapipe as mp
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            lm = handLms.landmark
            finger_tips = [4, 8, 12, 16, 20]

            fingers = []

            fingers.append(1 if lm[4].x < lm[3].x else 0)

            for tip in finger_tips[1:]:
                fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)

            count = sum(fingers)

            if fingers == [1, 0, 0, 0, 0]:
                gesture = "\\o/ Thumbs Up"
                color = (0, 215, 255)
            elif count == 0:
                gesture = "[###] Fist"
                color = (0, 0, 255)
            elif count == 5:
                gesture = "[|||||] Open Palm"
                color = (0, 255, 0)
            elif count == 1 and fingers[1] == 1:
                gesture = "|__ One Finger"
                color = (153, 153, 0)
            else:
                gesture = f"Fingers Up: {count}"
                color = (64, 64, 64) 

            cv2.putText(frame, gesture, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return frame


def run_pipeline(camera_id: int = 0):
    """Run real-time vision inference loop with dual-mode toggle."""

    mode = 'object'
    cap = cv2.VideoCapture(camera_id)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == 'object':
                results = yolo_model(frame)
                frame = results[0].plot()
            else:
                frame = detect_gesture(frame)

            cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("AI Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                mode = 'object'
            elif key == ord('h'):
                mode = 'hand'
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()

