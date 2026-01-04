import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import time
from vision_helpers import open_camera

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
CHESSBOARD_SIZE = (8, 5) # (cols, rows) *corners inside
COOLDOWN_TIMER = 1.5
# -------------------------------

# --- Setup ---
cap = open_camera(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_capture_time = 0
img_count = 0

save_dir = os.path.join(script_dir, "camera calib imgs")
os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

print("Press 'Esc' to quit")

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, 
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        display = frame.copy()
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)

        if time.time() - last_capture_time >= COOLDOWN_TIMER:
            filename = os.path.join(save_dir, f"calib_{img_count:03d}.png")
            cv2.imwrite(filename, gray)
            print(f"Saved {filename}")

            img_count += 1
            last_capture_time = time.time()

    cv2.imshow("Calibration capture", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break


cap.release()
cv2.destroyAllWindows()
