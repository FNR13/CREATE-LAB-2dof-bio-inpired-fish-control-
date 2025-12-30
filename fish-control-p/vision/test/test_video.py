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
VIDEO_NAME = "pool_test2.mp4"  # Video file name
PAUSE = False
STREAM = False
# -------------------------------

# --- Initialize Vision ---
FPS = 20
vision = Fish_Vision(camera_index=0, delta_t=1/FPS) # video fps is 20

calibration_path = os.path.join(parent_dir, "media", "calibration", "calibration_fallback.jpg")
vision.calibrate(cv2.imread(calibration_path))

# Open video
video_path = os.path.join(parent_dir, "media", VIDEO_NAME)  # adjust path if needed

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    exit()
print(f"🎥 Playing video: {VIDEO_NAME}")

# Play video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video reached")
        break
    
    valid, u1, v1, ang1 = vision.detect_fish(frame, show_output=PAUSE, stream=STREAM)
    valid, state  = vision.get_fish_state(img=frame, use_yolo=False)

    img_disp = frame.copy()

    img_disp = draw_fish_state(img_disp, valid, state, axis_length=DRAWING['axis_length_pixels'])

    # Show the frame
    cv2.imshow("Video Playback", img_disp)
    
    key = cv2.waitKey(int(1000/FPS)) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()