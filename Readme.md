# Gesture-Based Mouse & Keyboard Controller

This project lets you control your mouse and keyboard using hand gestures captured via a webcam. It uses **MediaPipe** for hand tracking and **PyAutoGUI** for controlling the mouse and keyboard.  
This project works **only on Windows** and is currently **optimized for 1920x1080 screens**, so user experience may vary on other resolutions.

---

## Features

- Move your mouse **with hand gestures**.  
- Perform **left click** and **drag** gestures.  
- Switch windows using the **Alt+Tab** gesture.  
- Trigger **keyboard keys** like `Up Arrow` and `Enter` with hand gestures.  
- Optional **invisible webcam window** for easier screen recording with OBS.  

---

## Gesture Guide

**Important:** All gestures must be performed **inside the green bounding box** to be recognized.  

| Gesture | Action |
|---------|--------|
| **Index finger up** | Move mouse cursor |
| **Index + Middle finger up** | Left click |
| **Index + Middle finger up for some duration** | Drag (hold) left click |
| **Index + Middle + Ring finger up** | Alt + Tab |
| **Thumbs up** | Arrow Up key |
| **Fist (knuckles facing camera / thumb down)** | Enter key |

---

## How to Use

1. Launch the script. The **first 5 seconds** is a countdown to get your hand in position.  
2. Make sure your hand is **fully inside the green bounding box** for gestures to be detected.  
3. Sit in a **stable position with a calm background** — this helps MediaPipe track your hand more accurately.  
4. Ensure your hand is **well-lit**, avoiding harsh shadows.  
5. The webcam window can be toggled **visible/invisible** using the `WINDOW_VISIBLE` variable in the script:  
   - Visible: easier to debug gestures but may interfere with **Alt+Tab** due to Windows auto-focus.  
   - Invisible: better for **screen recording** with OBS.  

---

## Steps to Run

1. Ensure Python 3.8+ is installed.  
2. Install dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure you have a working webcam.  
4. Run the main script:
    ```bash
    python main.py
    ```
5. Wait for the countdown and start performing gestures inside the green bounding box.  
6. Press `Esc` to exit at any time.  
   - If `Esc` doesn't work, you can **Alt+Tab** to the gesture window and press `Esc`, close the IDE running Python, or use keyboard interrupts in the terminal.  

---

## Requirements

- **OpenCV** – for video capture and overlay graphics.  
- **MediaPipe** – for hand landmark detection.  
- **PyAutoGUI** – for controlling mouse and keyboard.  

All dependencies are listed in `requirements.txt`.

---

## Tips for Best Performance

- **Lighting matters:** bright, even lighting helps with accurate hand tracking.  
- **Stable background:** avoid clutter or moving objects behind you.  
- **Stay within the green bounding box:** gestures outside the box may not be detected.  
- **Be patient with gestures:** some actions (like drag or Alt+Tab) require holding the gesture for a short duration.  
- **Screen resolution:** optimized for 1920x1080 screens; experience may vary on other resolutions.  

---

## Possible Improvements

- The code could be **modularized** more cleanly, but I didn’t bother since it’s a personal project for fun.  
- Fix the **Alt+Tab issue** when the OpenCV window is visible; Windows auto-focus can interfere.  
- Add a **config file** for gestures, sensitivity, or bounding box size.  
- Add **visual guides or diagrams** for easier reference.  
- Support **multiple screens** or different aspect ratios for better scaling.  

---

## Disclaimer

This is a **fun, experimental project** for personal use. Accuracy may vary depending on lighting, background, and hand position. Works **only on Windows**.
