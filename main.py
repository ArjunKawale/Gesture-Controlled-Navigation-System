import cv2
import mediapipe as mp
import pyautogui as pyg
import ctypes
import win32gui

mp_hands = mp.solutions.hands


# -----------------------------------------------------
# MAKE WINDOW ALWAYS ON TOP
# -----------------------------------------------------
def make_window_topmost(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        ctypes.windll.user32.SetWindowPos(
            hwnd, -1, 0, 0, 0, 0,
            0x0001 | 0x0002   # NOSIZE | NOMOVE
        )


# -----------------------------------------------------
# DRAW OVERLAY + BOUNDING BOX
# -----------------------------------------------------
def draw_overlay_box(image, box):
    (box_left, box_top, box_right, box_bottom) = box
    h, w, _ = image.shape

    overlay = image.copy()
    alpha = 0.45

    cv2.rectangle(overlay, (0, 0), (w, box_top), (128, 128, 128), -1)
    cv2.rectangle(overlay, (0, box_bottom), (w, h), (128, 128, 128), -1)
    cv2.rectangle(overlay, (0, box_top), (box_left, box_bottom), (128, 128, 128), -1)
    cv2.rectangle(overlay, (box_right, box_top), (w, box_bottom), (128, 128, 128), -1)

    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)
    return image



# -----------------------------------------------------
# MAP TO SCREEN WITH SMOOTHING
# -----------------------------------------------------
def map_to_screen(x, y, box, screen_w, screen_h, smoothing_state, SMOOTHING=0.7):
    box_left, box_top, box_right, box_bottom = box
    prev_x, prev_y = smoothing_state

    norm_x = (x - box_left) / (box_right - box_left)
    norm_y = (y - box_top) / (box_bottom - box_top)

    norm_x = 1 - norm_x  # horizontal flip

    target_x = int(norm_x * screen_w)
    target_y = int(norm_y * screen_h)

    if prev_x is None:
        prev_x, prev_y = target_x, target_y

    smooth_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
    smooth_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

    return smooth_x, smooth_y, (smooth_x, smooth_y)



# -----------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
click_state = False   # avoid repeated clicks

WINDOW_NAME = "Gesture Mouse"

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, _ = image.shape

        # Dynamic bounding box
        TOP_MARGIN = int(0.01 * h)
        LEFT_MARGIN = int(0.01 * w)
        RIGHT_MARGIN = int(0.01 * w)
        BOTTOM_MARGIN = int(0.20 * h)

        box = (
            LEFT_MARGIN,
            TOP_MARGIN,
            w - RIGHT_MARGIN,
            h - BOTTOM_MARGIN
        )

        # Process hand
        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        image.flags.writeable = True

        # Draw shaded area + green box
        image = draw_overlay_box(image, box)

        # Hand detected?
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # Landmarks
                l8  = hand.landmark[8]
                l12 = hand.landmark[12]
                l16 = hand.landmark[16]
                l20 = hand.landmark[20]
                l7  = hand.landmark[7]

                x8,  y8  = int(l8.x  * w), int(l8.y  * h)
                x12, y12 = int(l12.x * w), int(l12.y * h)
                y16 = int(l16.y * h)
                y20 = int(l20.y * h)
                y7  = int(l7.y  * h)

                inside = (box[0] < x8 < box[2] and box[1] < y8 < box[3])

                # ---------------------------------------------------
                # GESTURE: BLUE = CLICK
                # Conditions:
                # 1. 8 above 16 & 20
                # 2. 12 above 16 & 20
                # 3. 12 above 7
                # ---------------------------------------------------
                blue_click = (
                    y8  < y16 and y8  < y20 and
                    y12 < y16 and y12 < y20 and
                    y12 < y7
                )

                # ---------------------------------------------------
                # DRAW LANDMARK COLORS
                # ---------------------------------------------------
                if blue_click:
                    cv2.circle(image, (x8, y8),  10, (255, 0, 0), -1)
                    cv2.circle(image, (x12, y12), 10, (255, 0, 0), -1)
                else:
                    if inside:
                        cv2.circle(image, (x8, y8), 8, (0, 255, 0), -1)
                    else:
                        cv2.circle(image, (x8, y8), 8, (0, 0, 255), -1)



                # ---------------------------------------------------
                # MOUSE MOVEMENT (DISABLED IN BLUE GESTURE)
                # ---------------------------------------------------
                if inside and not blue_click:
                    sw, sh = pyg.size()
                    mapped_x, mapped_y, (prev_x, prev_y) = map_to_screen(
                        x8, y8, box, sw, sh, (prev_x, prev_y)
                    )
                    pyg.moveTo(mapped_x, mapped_y)



                # ---------------------------------------------------
                # LEFT CLICK (ONLY ON BLUE)
                # ---------------------------------------------------
                if blue_click and not click_state:
                    pyg.click()
                    click_state = True

                if not blue_click:
                    click_state = False



        # Show output
        cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))
        make_window_topmost(WINDOW_NAME)   # ALWAYS on top

        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()
