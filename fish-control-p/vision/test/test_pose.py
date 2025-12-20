import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
from vision_helpers import open_camera
from vision import Fish_Vision
from vision_config import MARKERS

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = False
IMG_NAME =  "photo1.jpg" # "pool_test.jpg"

if not USE_CAMERA:
    ground_truth = {
        0: (0.37, 0.80),
        2: (1.35, 0.70)
    }
# -------------------------------  0.37 0.80 ;1.35 0.70;  

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

    ret, img = cap.read()
    if not ret:
        print("❌ Failed to read from webcam")
        exit()

fv = Fish_Vision(camera_index=0)

# --- Calibration ---
print("\n--- Calibration ---")
MAX_ATTEMPTS = 5
for attempt in range(1, MAX_ATTEMPTS + 1):
    if USE_CAMERA:
        ret, img = cap.read()  # grab a new frame from the camera
        if not ret:
            print(f"⚠️ Attempt {attempt}: Failed to grab frame from camera.")
            continue

    ok = fv.calibrate(img, show_output=True)
    print(ok)
    if ok:
        print(f"✅ Calibration succeeded on attempt {attempt}.")
        break
    else:
        print(f"⚠️ Attempt {attempt}: Calibration failed, retrying...")

if not ok:
    print("❌ Calibration failed after maximum attempts.")
    exit()

print("✅ Calibration successful!")

if USE_CAMERA:
    cap.release()

# --- Marker world coordinates ---
print("\n--- Marker World Coordinates ---")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(MARKERS["dictionary"])
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=MARKERS["parameters"])

if ids is not None:
    ids = ids.flatten()

    for i, marker_id in enumerate(ids):
        # Only use markers that are in ground truth (target IDs)
        if not USE_CAMERA:
            if marker_id not in ground_truth:
                continue

            center = corners[i][0].mean(axis=0)
            u, v = center

            Xw_est, Yw_est = fv.project_pixel_to_world(u, v)
            Xw_gt, Yw_gt = ground_truth[marker_id]

            print(
                f"Marker {marker_id} "
                f"pixel=({u:.1f}, {v:.1f}) → "
                f"est=({Xw_est:.4f}, {Yw_est:.4f}) | "
                f"GT=({Xw_gt:.2f}, {Yw_gt:.2f})"
            )
else:
    print("⚠ No markers detected.")



# --- Show results ---

cv2.destroyAllWindows()
print("\n=== Test complete ===")