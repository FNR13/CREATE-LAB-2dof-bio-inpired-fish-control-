import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np

# -------------------------------
# PARAMETERS
# -------------------------------
IMAGE_DIR     = "camera calib imgs"  # folder with saved chessboard images
CHESSBOARD_SIZE = (8, 5)        # inner corners (columns, rows)
SQUARE_SIZE_M   = 0.043         # square size in meters
OUTPUT_FILE     = "camera_calib.npz"
MIN_VIEWS       = 5             # minimum number of successful chessboard detections
CORNER_SHIFT_THR = 1.0          # max allowed mean shift of corners (pixels)
BLUR_THR         = 100 
# -------------------------------

image_dir = os.path.join(script_dir, IMAGE_DIR)

# Prepare object points for the chessboard corners
objp = np.zeros((CHESSBOARD_SIZE[1]*CHESSBOARD_SIZE[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all images
images = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if not images:
    print(f"No images found in {image_dir}")
    sys.exit(1)

print(f"Found {len(images)} images in {image_dir}")

# Process each image
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    display = img.copy()

    if found:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Check corner shift (quality of refinement)
        shift = np.mean(np.linalg.norm(corners_subpix - corners, axis=2))
        if shift > CORNER_SHIFT_THR:
            print(f"Skipping {fname}: corner shift too large ({shift:.2f}px)")
            continue

        # Check if board is fully visible
        x, y = corners_subpix[:,0,0], corners_subpix[:,0,1]
        margin = 5
        if x.min() < margin or x.max() > w-margin or y.min() < margin or y.max() > h-margin:
            print(f"Skipping {fname}: chessboard partially outside frame")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners_subpix)

        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners_subpix, found)
        text = f"Chessboard detected ({len(objpoints)} views)"
    else:
        text = "No chessboard detected"

    cv2.putText(display, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Calibration", display)
    key = cv2.waitKey(500)  # show each image for 0.5s
    if key == 27:
        break

cv2.destroyAllWindows()

# Check if enough views were collected
if len(objpoints) < MIN_VIEWS:
    print(f"Not enough views ({len(objpoints)}) to calibrate (need at least {MIN_VIEWS}).")
    sys.exit(1)

# Perform camera calibration
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

# Save calibration results
output_path = os.path.join(script_dir, OUTPUT_FILE)
np.savez(output_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print(f"Calibration saved to {output_path}")
