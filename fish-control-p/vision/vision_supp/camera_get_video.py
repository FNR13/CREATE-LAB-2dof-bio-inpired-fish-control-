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
OUTPUT_FILE = "output_video.mp4"
FPS = 20.0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
RECORD_SECONDS = 20   # set None for infinite

# ------------------------
# Open camera
# ------------------------
cap = open_camera(CAMERA_INDEX)
print(f"Camera {CAMERA_INDEX} opened successfully.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera")

# ------------------------
# Video writer
# ------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    os.path.join(script_dir, "imgs", OUTPUT_FILE),
    fourcc,
    FPS,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

print("🎥 Recording... Press 'q' to stop")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- OPTIONAL: draw something ----
    cv2.putText(
        frame,
        "Recording",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    out.write(frame)
    cv2.imshow("Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if RECORD_SECONDS is not None:
        if time.time() - start_time > RECORD_SECONDS:
            break

# ------------------------
# Cleanup
# ------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video saved as: {OUTPUT_FILE}")
