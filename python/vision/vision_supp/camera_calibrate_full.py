import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
from vision_helpers import open_camera
import time

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
CHESSBOARD_SIZE = (8, 5)  # (cols, rows) *corners inside
SQUARE_SIZE_M = 0.043  # in meters
COOLDOWN_TIMER = 1.5
MIN_VIEWS = 5
OUTPUT_FILE = "camera_calib.npz"
# -------------------------------

# --- Setup ---
cap = open_camera(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_capture_time = 0.0

save_dir = os.path.join(script_dir, "camera calibration data")
os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

# --- Prepare object points ---
objp = np.zeros((CHESSBOARD_SIZE[1] * CHESSBOARD_SIZE[0], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

objpoints = []  # 3D world coordinates
imgpoints = []  # 2D image coordinates

print("Press 'Esc' to quit")

# --- Calibration capture loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, found)
        text = "Chessboard detected"

        
        if time.time() - last_capture_time >= COOLDOWN_TIMER:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners_subpix = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp.copy())
            imgpoints.append(corners_subpix)
            print(f"Captured view #{len(objpoints)}")

            last_capture_time = time.time()
    else:
        text = "Show chessboard to the camera"

    cv2.putText(frame, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Captured views: {len(objpoints)}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# --- Perform calibration ---
print(len(objpoints))

if len(objpoints) < MIN_VIEWS:
    print("Not enough views to calibrate (need ~10+ ideally).")
    sys.exit(0)

print("Calibrating...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("Calibration RMS reprojection error:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

np.savez(
    os.path.join(save_dir, OUTPUT_FILE),
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print(f"Calibration saved to {OUTPUT_FILE}")
