import cv2 
import pyautogui as pyg
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

printed_window_size = False

# ------- NEW: smoothing variables -------
prev_x, prev_y = None, None
SMOOTHING = 0.7   # higher = smoother (0.6 - 0.85 works best)
# ----------------------------------------

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if not printed_window_size:
            h, w, _ = image.shape
            print(f"Webcam frame size: {w} x {h}")
            printed_window_size = True

        TOP_MARGIN = int(0.01 * h)
        LEFT_MARGIN = int(0.01 * w)
        RIGHT_MARGIN = int(0.01 * w)
        BOTTOM_MARGIN = int(0.20 * h)

        box_top = TOP_MARGIN
        box_bottom = h - BOTTOM_MARGIN
        box_left = LEFT_MARGIN
        box_right = w - RIGHT_MARGIN

        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        image.flags.writeable = True

        overlay = image.copy()
        alpha = 0.45
        cv2.rectangle(overlay, (0, 0), (w, box_top), (128, 128, 128), -1)
        cv2.rectangle(overlay, (0, box_bottom), (w, h), (128, 128, 128), -1)
        cv2.rectangle(overlay, (0, box_top), (box_left, box_bottom), (128, 128, 128), -1)
        cv2.rectangle(overlay, (box_right, box_top), (w, box_bottom), (128, 128, 128), -1)

        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                landmark8 = hand_landmarks.landmark[8]
                if landmark8.x != landmark8.x or landmark8.y != landmark8.y:
                    continue

                x8 = int(landmark8.x * w)
                y8 = int(landmark8.y * h)

                inside = (
                    box_left < x8 < box_right and 
                    box_top < y8 < box_bottom
                )

                if inside:
                    cv2.circle(image, (x8, y8), 8, (0, 255, 0), -1)

                    screen_w, screen_h = pyg.size()

                    norm_x = (x8 - box_left) / (box_right - box_left)
                    norm_y = (y8 - box_top) / (box_bottom - box_top)

                    norm_x = 1 - norm_x  # flip horizontally

                    target_x = int(norm_x * screen_w)
                    target_y = int(norm_y * screen_h)

                    # ------- NEW SMOOTHING -------
                    prev_x, prev_y
                    if prev_x is None:
                        prev_x, prev_y = target_x, target_y

                    smooth_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
                    smooth_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

                    prev_x, prev_y = smooth_x, smooth_y

                    pyg.moveTo(smooth_x, smooth_y)
                    # -----------------------------

                else:
                    cv2.circle(image, (x8, y8), 8, (0, 0, 255), -1)

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
