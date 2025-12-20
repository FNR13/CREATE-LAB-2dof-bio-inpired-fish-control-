import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
from vision_helpers import open_camera

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
SAVE_LAST_FRAME = True
# -------------------------------

cap = open_camera(CAMERA_INDEX)
print(f"Camera {CAMERA_INDEX} opened successfully.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Live camera feed loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Feed", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Save the last captured frame ---
if SAVE_LAST_FRAME:
    img_path = os.path.join(script_dir, "imgs", "photo.jpg")
    cv2.imwrite(img_path, frame)
    print("Saved photo.jpg")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
