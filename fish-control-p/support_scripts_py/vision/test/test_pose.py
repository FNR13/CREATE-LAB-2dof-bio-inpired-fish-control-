import os
import sys
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

from vision_config import MARKERS

from vision import Fish_Vision 

# -------------------------------
# Parameters
# -------------------------------
use_photo = True
img_name = "4corners.png"

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

# -------------------------------
# Calibration
# -------------------------------
print("\n--- Calibration ---")
ok = fv.calibrate(img)
if not ok:
    print("❌ Calibration failed — cannot continue.")
    exit()
print("✅ Calibration successful!")

# -------------------------------
# Marker world coordinates
# -------------------------------
print("\n--- Marker World Coordinates ---")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(MARKERS["dictionary"])
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=MARKERS["parameters"])

if ids is not None:
    ids = ids.flatten()
    for marker_id in MARKERS["target_ids"]:
        if marker_id not in ids:
            print(f"⚠ Target marker {marker_id} not detected.")
            continue
        idx = np.where(ids == marker_id)[0][0]
        center = corners[idx][0].mean(axis=0)
        u, v = center
        Xw, Yw = fv.get_real_world_position(u, v)
        print(f"Marker {marker_id} pixel=({u:.1f}, {v:.1f}) → world=({Xw:.4f}, {Yw:.4f})")

print("\n=== Test complete ===")
