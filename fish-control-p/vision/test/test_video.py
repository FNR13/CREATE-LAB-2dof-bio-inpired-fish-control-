import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import time

from vision import Fish_Vision  

# -------------------------------
# Parameters
# -------------------------------
VIDEO_NAME = "pool_test2.mp4"  # Video file name
SHOW_OUTPUT = True
PAUSE = False
# -------------------------------

# --- Initialize Vision ---
vision = Fish_Vision(camera_index=0)

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
    
    valid, u1, v1, ang1 = vision.detect_fish(frame, show_output=PAUSE)

    if not valid:
        img_disp = frame.copy()
        cv2.putText(
            img_disp,
            "No fish detected",
            (700, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:
        img_disp = frame.copy()

        # Draw centroid
        cv2.circle(img_disp, (int(u1), int(v1)), 6, (0, 0, 255), -1)
        
        axis_length = 50
        # Draw orientation axis
        x2 = int(u1 + axis_length * np.cos(ang1))
        y2 = int(v1 + axis_length * np.sin(ang1))
        cv2.line(img_disp, (int(u1), int(v1)), (x2, y2), (255, 0, 0), 3)

        # Overlay text with position and angle
        text = f"Pos: ({int(u1)}, {int(v1)})  Angle: {np.degrees(ang1):.1f}°"
        cv2.putText(
            img_disp, text,
            org=(700, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 255),
            thickness=2
        )

    # Show the frame
    if SHOW_OUTPUT:
        cv2.imshow("Video Playback", img_disp)
        if cv2.waitKey(60) & 0xFF == ord('q'):  
            break

cap.release()
cv2.destroyAllWindows()