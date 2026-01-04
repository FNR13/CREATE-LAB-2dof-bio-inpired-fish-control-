import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import time
import numpy as np
from vision_helpers import open_camera
from vision import Fish_Vision
from vision_config import MARKERS

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = False
IMG_NAME =  "pool_fish2.jpg"
IMG_NAME = "pool_test.jpg"
MAX_ATTEMPTS = 5
# -------------------------------  0.37 0.80 ;1.35 0.70;  

vision = Fish_Vision(camera_index=0)

# --- Load test image or webcam ---
if not USE_CAMERA:
    img_path = os.path.join(parent_dir, "vision_supp", "imgs", IMG_NAME)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        exit()
    print(f"📷 Loaded image: {IMG_NAME}")

    marker_test = False
    if IMG_NAME == "pool_test.jpg":
        marker_test = True

        vision.known_markers = [4, 3, 1]

        TARGET_MARKERS = [2, 0]
        GROUND_THRUTH = {
            0: (0.37, 0.80),
            2: (1.35, 0.70)
        }
    
else:
    print("📹 Using webcam...")
    cap = open_camera(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, img = cap.read()
    if not ret:
        print("❌ Failed to read from webcam")
        exit()
    
# --- Calibration ---
print("\n--- Calibration ---")

if USE_CAMERA:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ret, img = cap.read()
        if not ret:
            print(f"⚠️ Attempt {attempt}: Failed to grab frame from camera.")
            continue

        ok = vision.calibrate(img, show_output=True)
        
        if ok:
            print(f"✅ Calibration succeeded on attempt {attempt}.")
            break
        else:
            print(f"⚠️ Attempt {attempt}: Calibration failed, retrying...")
        time.sleep(0.5)
else:
    ok = vision.calibrate(img, show_output=True)

if not ok:
    print("❌ Calibration failed.")
    calibration_path = os.path.join(parent_dir, "media", "calibration", "calibration_fallback.jpg")
    vision.calibrate(cv2.imread(calibration_path))
    print("Used Fallback calibration image.")

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
        if not USE_CAMERA and IMG_NAME == "pool_test.jpg":
            if marker_id not in GROUND_THRUTH:
                continue

            center = corners[i][0].mean(axis=0)
            u, v = center

            Xw_est, Yw_est = vision.project_pixel_to_world(u, v)
            Xw_gt, Yw_gt = GROUND_THRUTH[marker_id]

            # --- Error ---
            dx = Xw_est - Xw_gt
            dy = Yw_est - Yw_gt
            err = np.sqrt(dx**2 + dy**2)

            print(
                f"Marker {marker_id} → "
                f"EST=({Xw_est:.4f}, {Yw_est:.4f}) | "
                f"GT=({Xw_gt:.2f}, {Yw_gt:.2f}) | "
                f"ERR={err:.4f} m"
            )
else:
    print("⚠ No markers detected.")

# --- Show results ---

cv2.destroyAllWindows()
print("\n=== Test complete ===")