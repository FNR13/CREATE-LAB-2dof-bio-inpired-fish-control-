import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import time

from vision import Fish_Vision  
from vision_helpers import draw_fish_state
from vision_config import DRAWING

# -------------------------------
# Parameters
# -------------------------------
VIDEO_NAME = "pool_test2.mp4"
OUTPUT_VIDEO = "pool_test2_output.mp4"
PAUSE = False
STREAM = False
FPS = 20
# -------------------------------

# --- Initialize Vision ---
vision = Fish_Vision(camera_index=0, delta_t=1/FPS)

calibration_path = os.path.join(parent_dir, "media", "calibration", "calibration_fallback.jpg")
vision.calibrate(cv2.imread(calibration_path))

# Open input video
video_path = os.path.join(parent_dir, "media", VIDEO_NAME)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    exit()

print(f"🎥 Playing video: {VIDEO_NAME}")

# --- Initialize Video Writer ---
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = os.path.join(parent_dir, "media", OUTPUT_VIDEO)
out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

# --- Process video ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video reached")
        break
    
    valid, u1, v1, ang1 = vision.detect_fish(frame, show_output=PAUSE, stream=STREAM)
    valid, state = vision.get_fish_state(img=frame, use_yolo=False)

    img_disp = frame.copy()
    img_disp = draw_fish_state(
        img_disp, valid, state,
        axis_length=DRAWING['axis_length_pixels']
    )

    # Show frame
    cv2.imshow("Video Playback", img_disp)

    # ✅ Write frame to output video
    out.write(img_disp)
    
    key = cv2.waitKey(int(1000/FPS)) & 0xFF
    if key == 27:  # ESC
        break

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"💾 Output video saved to: {output_path}")
