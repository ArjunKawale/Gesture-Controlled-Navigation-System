import cv2
import mediapipe as mp
import pyautogui as pyg

import ctypes
import win32gui
import win32con
import time
import math

mp_hands = mp.solutions.hands

# ============================
#  WINDOW VISIBILITY TOGGLE
# ============================
WINDOW_VISIBLE = True  # <---- SET True TO SHOW, False TO HIDE


# -----------------------------------------------------
#  SET WINDOW VISIBILITY  (NEW)
# -----------------------------------------------------
def set_window_visibility(hwnd, visible):
    if visible:
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        style = style & ~win32con.WS_EX_LAYERED
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)

        ctypes.windll.user32.SetLayeredWindowAttributes(
            hwnd, 0, 255, win32con.LWA_ALPHA
        )
    else:
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            style | win32con.WS_EX_LAYERED
        )

        ctypes.windll.user32.SetLayeredWindowAttributes(
            hwnd, 0, 0, win32con.LWA_ALPHA
        )


# -----------------------------------------------------
# WINDOW CONTROLLER  (TOPMOST REMOVED)
# -----------------------------------------------------
def send_window_to_background(hwnd):
    ctypes.windll.user32.SetWindowPos(
        hwnd,
        win32con.HWND_BOTTOM,
        0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
    )

def maintain_window(window_name):
    # If window is visible, do NOT push it to background
    if WINDOW_VISIBLE:
        return

    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        send_window_to_background(hwnd)



# -----------------------------------------------------
# REST OF YOUR CODE (UNCHANGED)
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


def map_to_screen(x, y, box, screen_w, screen_h, smoothing_state,
                  SMOOTHING=0.7, SENSITIVITY=1.4):

    box_left, box_top, box_right, box_bottom = box
    prev_x, prev_y = smoothing_state

    norm_x = (x - box_left) / (box_right - box_left)
    norm_y = (y - box_top) / (box_bottom - box_top)

    norm_x = 1 - norm_x
    target_x = int(norm_x * screen_w)
    target_y = int(norm_y * screen_h)

    center_x = screen_w // 2
    center_y = screen_h // 2

    target_x = int(center_x + (target_x - center_x) * SENSITIVITY)
    target_y = int(center_y + (target_y - center_y) * SENSITIVITY)

    if prev_x is None:
        return target_x, target_y, (target_x, target_y)

    smooth_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
    smooth_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

    return smooth_x, smooth_y, (smooth_x, smooth_y)


def gesture_left_click(image, x8, y8, x12, y12, y16, y20, y7, click_state):
    blue_click = (
        y8 < y16 and y8 < y20 and
        y12 < y16 and y12 < y20 and
        y12 < y7
    )

    if blue_click:
        cv2.circle(image, (x8, y8), 10, (255, 0, 0), -1)
        cv2.circle(image, (x12, y12), 10, (255, 0, 0), -1)
        if not click_state:
            pyg.click()
            click_state = True
    else:
        click_state = False

    return click_state, blue_click


def gesture_hold(image, x8, y8, x12, y12, box, prev_x, prev_y, hold_state):
    PURPLE = (255, 0, 255)
    cv2.circle(image, (x8, y8), 15, PURPLE, -1)
    cv2.circle(image, (x12, y12), 15, PURPLE, -1)

    sw, sh = pyg.size()
    mapped_x, mapped_y, (prev_x, prev_y) = map_to_screen(
        x8, y8, box, sw, sh, (prev_x, prev_y),
        SMOOTHING=0.6, SENSITIVITY=1.2
    )

    if not hold_state["down_sent"]:
        pyg.mouseDown()
        hold_state["down_sent"] = True
        hold_state["active"] = True

    pyg.moveTo(mapped_x, mapped_y)
    return prev_x, prev_y, hold_state


def handle_gesture_actions(image, x8, y8, inside, blue_click, box, prev_x, prev_y, suppress_green=False):

    if not blue_click and not suppress_green:
        col = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(image, (x8, y8), 8, col, -1)

    if inside and not blue_click and not suppress_green:
        sw, sh = pyg.size()
        mapped_x, mapped_y, (prev_x, prev_y) = map_to_screen(
            x8, y8, box, sw, sh, (prev_x, prev_y)
        )
        pyg.moveTo(mapped_x, mapped_y)

    return prev_x, prev_y


def gesture_yellow(image, pts):

    y8 = pts["y8"]
    y12 = pts["y12"]
    y16 = pts["y16"]

    ref = [
        pts["y20"], pts["y19"],
        pts["y15"], pts["y11"], pts["y14"]
    ]

    cond = (
        all(y8 < r for r in ref) and
        all(y12 < r for r in ref) and
        all(y16 < r for r in ref)
    )

    if cond:
        Y = (0, 255, 255)
        cv2.circle(image, (pts["x8"], pts["y8"]), 10, Y, -1)
        cv2.circle(image, (pts["x12"], pts["y12"]), 10, Y, -1)
        cv2.circle(image, (pts["x16"], pts["y16"]), 10, Y, -1)

    return cond


def gesture_alt_tab_hardcoded(yellow, state):

    if not yellow:
        if state["active"]:
            pyg.keyUp("alt")
        state["active"] = False
        state["last_time"] = None
        state["start_hold"] = None
        return state

    if state["start_hold"] is None:
        state["start_hold"] = time.time()
        return state

    if not state["active"]:
        if time.time() - state["start_hold"] >= 0.25:
            pyg.keyDown("alt")
            pyg.keyDown("tab")
            pyg.keyUp("tab")
            state["active"] = True
            state["last_time"] = time.time()
        return state

    now = time.time()
    if now - state["last_time"] >= 0.5:
        pyg.keyDown("tab")
        pyg.keyUp("tab")
        state["last_time"] = now

    return state


def gesture_thumb_tip(image, l):
    y4 = int(l[4].y * image.shape[0])

    if (
        y4 < int(l[3].y * image.shape[0]) and
        y4 < int(l[2].y * image.shape[0]) and
        y4 < int(l[8].y * image.shape[0]) and
        y4 < int(l[12].y * image.shape[0]) and
        y4 < int(l[16].y * image.shape[0]) and
        y4 < int(l[20].y * image.shape[0]) and
        y4 < int(l[5].y * image.shape[0]) and
        y4 < int(l[9].y * image.shape[0]) and
        y4 < int(l[13].y * image.shape[0]) and
        y4 < int(l[17].y * image.shape[0])
    ):
        x4 = int(l[4].x * image.shape[1])
        cv2.circle(image, (x4, y4), 10, (255, 255, 255), -1)
        return True
    return False


# -----------------------------
# NEW ENTER KEY GESTURE
# -----------------------------
def gesture_enter_condition(image, l):
    h, w, _ = image.shape

    x5, y5   = int(l[5].x * w), int(l[5].y * h)
    x9, y9   = int(l[9].x * w), int(l[9].y * h)
    x13, y13 = int(l[13].x * w), int(l[13].y * h)
    x17, y17 = int(l[17].x * w), int(l[17].y * h)

    y1 = int(l[1].y * h)
    y2 = int(l[2].y * h)
    y3 = int(l[3].y * h)
    y4 = int(l[4].y * h)

    y8  = int(l[8].y * h)
    y12 = int(l[12].y * h)
    y16 = int(l[16].y * h)
    y20 = int(l[20].y * h)

    cond = (
        y5 < min(y1, y2, y3, y4, y8, y12, y16, y20) and
        y9 < min(y1, y2, y3, y4, y8, y12, y16, y20) and
        y13 < min(y1, y2, y3, y4, y8, y12, y16, y20) and
        y17 < min(y1, y2, y3, y4, y8, y12, y16, y20)
    )

    if cond:
        BLACK = (0, 0, 0)
        for x, y in [(x5, y5), (x9, y9), (x13, y13), (x17, y17)]:
            cv2.circle(image, (x, y), 10, BLACK, -1)

    return cond



# ============================
# BEGIN RUNTIME
# ============================

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
click_state = False

hold_start_time = None

yellow_state = {"active": False, "last_time": None}
hold_state = {"active": False, "down_sent": False}
thumb_state = {"active": False, "pressed": False}
enter_state = {"active": False, "pressed": False}

WINDOW_NAME = "Gesture Mouse"
cv2.namedWindow(WINDOW_NAME)
cv2.resizeWindow(WINDOW_NAME, 1, 1)
cv2.moveWindow(WINDOW_NAME, 0, 0)

hwnd = win32gui.FindWindow(None, WINDOW_NAME)

# APPLY VISIBILITY TOGGLE  ---------------------------
set_window_visibility(hwnd, WINDOW_VISIBLE)
# ----------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter("gesture_record.avi", fourcc, 20.0, (frame_width, frame_height))

start_time = time.time()
COUNTDOWN_DURATION = 5

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, _ = image.shape
        box = (int(0.01*w), int(0.01*h), int(0.99*w), int(0.80*h))

        now = time.time()
        elapsed = now - start_time

        if elapsed < COUNTDOWN_DURATION:
            remaining = int(COUNTDOWN_DURATION - elapsed)

            text = f"Starting in {remaining}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2

            flipped = cv2.flip(image, 1)

            cv2.putText(flipped, text, (20, 40), font, scale,
                        (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(flipped, text, (20, 40), font, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

            image = cv2.flip(flipped, 1)
            out.write(image)

            cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))
            maintain_window(WINDOW_NAME)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            continue

        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        image.flags.writeable = True

        image = draw_overlay_box(image, box)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                l = hand.landmark
                x8, y8 = int(l[8].x*w), int(l[8].y*h)
                x12, y12 = int(l[12].x*w), int(l[12].y*h)
                x16, y16 = int(l[16].x*w), int(l[16].y*h)

                y20 = int(l[20].y*h)
                y7  = int(l[7].y*h)
                y11 = int(l[11].y*h)
                y14 = int(l[14].y*h)
                y15 = int(l[15].y*h)
                y19 = int(l[19].y*h)

                inside = (box[0] < x8 < box[2] and box[1] < y8 < box[3])

                click_state, blue_click = gesture_left_click(
                    image, x8, y8, x12, y12, y16, y20, y7, click_state
                )

                yellow = gesture_yellow(
                    image,
                    {
                        "x8": x8, "y8": y8,
                        "x12": x12, "y12": y12,
                        "x16": x16, "y16": y16,
                        "y20": y20, "y19": y19,
                        "y15": y15, "y11": y11, "y14": y14
                    }
                )
                yellow_state = gesture_alt_tab_hardcoded(yellow, yellow_state)

                thumb_active = gesture_thumb_tip(image, l)
                if thumb_active and not thumb_state["active"]:
                    pyg.press("up")
                    thumb_state["pressed"] = True
                thumb_state["active"] = thumb_active
                if not thumb_active:
                    thumb_state["pressed"] = False

                enter_active = gesture_enter_condition(image, l)
                if enter_active and not enter_state["active"]:
                    pyg.press("enter")
                    enter_state["pressed"] = True
                enter_state["active"] = enter_active
                if not enter_active:
                    enter_state["pressed"] = False

                if blue_click:
                    if hold_start_time is None:
                        hold_start_time = time.time()
                    elif (time.time() - hold_start_time) > 1:
                        prev_x, prev_y, hold_state = gesture_hold(
                            image, x8, y8, x12, y12,
                            box, prev_x, prev_y,
                            hold_state
                        )
                        continue
                else:
                    if hold_state["active"]:
                        pyg.mouseUp()
                    hold_state = {"active": False, "down_sent": False}
                    hold_start_time = None

                prev_x, prev_y = handle_gesture_actions(
                    image, x8, y8,
                    inside, blue_click,
                    box, prev_x, prev_y,
                    yellow or thumb_active or enter_active
                )

        out.write(image)

        cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))

        if not yellow_state["active"]:
            maintain_window(WINDOW_NAME)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
