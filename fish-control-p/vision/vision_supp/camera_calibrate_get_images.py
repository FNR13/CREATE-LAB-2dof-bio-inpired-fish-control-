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
board_size = (8, 5)          # inner corners (columns, rows)
save_interval = 1.5          # seconds
camera_id = 0
# -------------------------------

# --- Setup ---
cap = open_camera(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

last_save_time = 0
img_count = 0

save_dir = os.path.join(script_dir, "camera calib imgs")
os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

print("Press 'q' to quit")

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        # Refine corners
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Draw corners
        cv2.drawChessboardCorners(frame, board_size, corners, found)

        # Save at most once per second
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            filename = os.path.join(save_dir, f"calib_{img_count:03d}.png")
            cv2.imwrite(filename, gray)
            print(f"Saved {filename}")

            img_count += 1
            last_save_time = current_time

    cv2.imshow("Calibration capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
