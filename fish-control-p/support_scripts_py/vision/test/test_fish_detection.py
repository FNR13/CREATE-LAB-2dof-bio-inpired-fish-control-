import os
import sys
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

from vision import Fish_Vision  # replace with your actual file name

# -------------------------------
# Parameters
# -------------------------------
use_photo = True
img_name = "fish.png"  # example image
show_steps = True

# -------------------------------
# Load test image or webcam
# -------------------------------
if use_photo:
    img_path = os.path.join(parent_dir, "vision_supp", "imgs", img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        exit()
    print(f"📷 Loaded image: {img_name}")
else:
    print("📹 Using webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, img = cap.read()
    cap.release()
    if not ret:
        print("❌ Failed to read from webcam")
        exit()

# -------------------------------
# Initialize vision system
# -------------------------------
fv = Fish_Vision(camera_index=0)
print("🔧 Vision class initialized.")

# -------------------------------
# Test detect_fish() - full frame
# -------------------------------
print("\n--- Testing detect_fish() ---")
u1, v1, ang1 = fv.detect_fish(img.copy(), show_output=show_steps)

if u1 is None:
    print("❌ detect_fish(): No fish detected.")
else:
    print(f"🐟 detect_fish() → pixel=({u1:.1f}, {v1:.1f}), angle={np.degrees(ang1):.2f}°")

# -------------------------------
# Test detect_fish_yolo() - YOLO + ROI
# -------------------------------
print("\n--- Testing detect_fish_yolo() ---")
u2, v2, ang2 = fv.detect_fish_yolo(img.copy(), show_output=show_steps)

if u2 is None:
    print("❌ detect_fish_yolo(): No fish detected.")
else:
    print(f"🐟 detect_fish_yolo() → pixel=({u2:.1f}, {v2:.1f}), angle={np.degrees(ang2):.2f}°")
    
cv2.destroyAllWindows()
print("\n=== Test complete ===")
