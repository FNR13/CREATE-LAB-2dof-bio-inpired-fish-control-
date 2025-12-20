import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import time

from vision_helpers import open_camera
from vision import Fish_Vision  

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = True
IMG_NAME = "photo1.jpg"  
SHOW_OUTPUT = True
# -------------------------------

# --- Load test image or webcam ---
if not USE_CAMERA:
    img_path = os.path.join(parent_dir, "vision_supp", "imgs", IMG_NAME)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        exit()
    print(f"📷 Loaded image: {IMG_NAME}")
else:
    print("📹 Using webcam...")
    cap = open_camera(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for i in range(25):
        ret, img = cap.read()
    cap.release()
    if not ret:
        print("❌ Failed to read from webcam")
        exit()

fv = Fish_Vision(camera_index=0)
print("🔧 Vision class initialized.")

fv.calibrated = True
fv.pool_pixels = [(181, 53), (1099, 659)]

# --- Test detect_fish() - full frame ---
print("\n--- Testing detect_fish() ---")

u1, v1, ang1 = fv.detect_fish(img.copy(), show_output=SHOW_OUTPUT)

if u1 is None:
    print("❌ detect_fish(): No fish detected.")
else:
    print(f"🐟 detect_fish() → pixel=({u1:.1f}, {v1:.1f}), angle={np.degrees(ang1):.2f}°")


# --- Test detect_fish_yolo() - YOLO + ROI ---
print("\n--- Testing detect_fish_yolo() ---")
u2, v2, ang2 = fv.detect_fish_yolo(img.copy(), show_output=SHOW_OUTPUT)

if u2 is None:
    print("❌ detect_fish_yolo(): No fish detected.")
else:
    print(f"🐟 detect_fish_yolo() → pixel=({u2:.1f}, {v2:.1f}), angle={np.degrees(ang2):.2f}°")
    
cv2.destroyAllWindows()
print("\n=== Test complete ===")
