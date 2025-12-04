import cv2
import mediapipe as mp
import pyautogui as pyg
import ctypes
import win32gui
import win32con

mp_hands = mp.solutions.hands

# -----------------------------------------------------
# WINDOW CONTROLLER: keep topmost + regain focus
# -----------------------------------------------------
def maintain_window(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if not hwnd:
        return

    ctypes.windll.user32.SetWindowPos(
        hwnd,
        win32con.HWND_TOPMOST,
        0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )

    fg = win32gui.GetForegroundWindow()
    if fg != hwnd:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
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
def map_to_screen(x, y, box, screen_w, screen_h, smoothing_state,
                  SMOOTHING=0.7, SENSITIVITY=1.4):
    box_left, box_top, box_right, box_bottom = box
    prev_x, prev_y = smoothing_state

    norm_x = (x - box_left) / (box_right - box_left)
    norm_y = (y - box_top) / (box_bottom - box_top)

    norm_x = 1 - norm_x  # flip

    # BASE target positions
    target_x = int(norm_x * screen_w)
    target_y = int(norm_y * screen_h)

    # APPLY SENSITIVITY (adjust movement relative to center)
    center_x = screen_w // 2
    center_y = screen_h // 2

    target_x = int(center_x + (target_x - center_x) * SENSITIVITY)
    target_y = int(center_y + (target_y - center_y) * SENSITIVITY)

    # smoothing
    if prev_x is None:
        return target_x, target_y, (target_x, target_y)

    smooth_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
    smooth_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

    return smooth_x, smooth_y, (smooth_x, smooth_y)


# -----------------------------------------------------
# LEFT CLICK FUNCTION (GESTURE-SPECIFIC)
# -----------------------------------------------------
def gesture_left_click(image, x8, y8, x12, y12,
                       y16, y20, y7,
                       click_state):

    # Detect left click gesture (blue_click logic)
    blue_click = (
        y8 < y16 and y8 < y20 and
        y12 < y16 and y12 < y20 and
        y12 < y7
    )

    # Draw circles for visual feedback
    if blue_click:
        cv2.circle(image, (x8, y8), 10, (255, 0, 0), -1)
        cv2.circle(image, (x12, y12), 10, (255, 0, 0), -1)

        # Click only once
        if not click_state:
            pyg.click()
            click_state = True

    else:
        click_state = False

    return click_state, blue_click


# -----------------------------------------------------
# HANDLE GENERAL GESTURES (movement etc.)
# -----------------------------------------------------
def handle_gesture_actions(image, x8, y8,
                           inside, blue_click,
                           box, prev_x, prev_y):

    # Color indicator
    col = (0, 255, 0) if inside else (0, 0, 255)
    if not blue_click:
        cv2.circle(image, (x8, y8), 8, col, -1)

    # Move cursor
    if inside and not blue_click:
        sw, sh = pyg.size()
        mapped_x, mapped_y, (prev_x, prev_y) = map_to_screen(
            x8, y8, box, sw, sh, (prev_x, prev_y)
        )
        pyg.moveTo(mapped_x, mapped_y)

    return prev_x, prev_y


# -----------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------
cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None
click_state = False

WINDOW_NAME = "Gesture Mouse"
cv2.namedWindow(WINDOW_NAME)

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

        box = (
            int(0.01 * w),
            int(0.01 * h),
            int(0.99 * w),
            int(0.80 * h)
        )

        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        image.flags.writeable = True

        image = draw_overlay_box(image, box)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                l8  = hand.landmark[8]
                l12 = hand.landmark[12]
                l16 = hand.landmark[16]
                l20 = hand.landmark[20]
                l7  = hand.landmark[7]

                x8,  y8  = int(l8.x * w), int(l8.y * h)
                x12, y12 = int(l12.x * w), int(l12.y * h)
                y16 = int(l16.y * h)
                y20 = int(l20.y * h)
                y7  = int(l7.y * h)

                inside = (box[0] < x8 < box[2] and box[1] < y8 < box[3])

                # ----- LEFT CLICK GESTURE -----
                click_state, blue_click = gesture_left_click(
                    image, x8, y8, x12, y12,
                    y16, y20, y7,
                    click_state
                )

                # ----- MOVEMENT LOGIC -----
                prev_x, prev_y = handle_gesture_actions(
                    image, x8, y8,
                    inside, blue_click,
                    box, prev_x, prev_y
                )

        cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))
        maintain_window(WINDOW_NAME)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
