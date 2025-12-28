import cv2
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

from vision_helpers import open_camera, getCenters, drawAxes
from vision_config import MARKERS, CALIBRATION

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = False
IMG_NAME = "pool_fish2.jpg"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
# -------------------------------

KNOWN_MARKERS = [0, 1, 2]
MARKER_POSTIONS_3D = np.array([
    [0.00, 0.00, 0.00],   # ID 4
    [0.00, 1.75, 0.00],   # ID 3
    [2.75, 0.00, 0.00],   # ID 1
], dtype=np.float32)

TARGET_MARKERS = [0, 2]
GROUND_TRUTH = {
    0: (0.37, 0.80),
    2: (1.35, 0.70)
}

# --- Load image ---
if not USE_CAMERA:
    img = cv2.imread(os.path.join(script_dir, "imgs", IMG_NAME))
else:
    cap = open_camera(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cap.read()
    img = frame

# --- Load camera calibration ---
calib_path = os.path.join(script_dir, "camera calibration data", "camera_calib_1_5.npz")
calib_data = np.load(CALIBRATION["path"])
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# --- Axis for visualization ---
axis = np.float32([[0.50, 0, 0], [0, 0.50, 0], [0, 0, 0]])

# --- Convert to grayscale ---
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Detect AprilTags ---
corners, ids, _ = cv2.aruco.detectMarkers(imgGray, aruco_dict, parameters=parameters)

if ids is None or len(ids) < 3:
    print("Not enough markers detected")
else:
    sorted_centers, sorted_ids = getCenters(corners, ids)

    # Dictionary id -> center
    center_dict = {id_: center for id_, center in zip(sorted_ids, sorted_centers)}
    known_centers = np.array(
        [center_dict[m] for m in KNOWN_MARKERS if m in center_dict], 
        dtype=np.float32)

    retval, rvecs, tvecs = cv2.solveP3P(
        MARKER_POSTIONS_3D,
        known_centers.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_P3P
    )

    print(f"Found {retval} solution(s)")

    if retval > 0:
        rvec, tvec = rvecs[0], tvecs[0]
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        first_index = np.where(sorted_ids == KNOWN_MARKERS[0])[0][0]
        img = drawAxes(img, sorted_centers[first_index], imgpts)
        R, _ = cv2.Rodrigues(rvec)

    for marker_id in TARGET_MARKERS:
        if marker_id not in sorted_ids:
            print(f"⚠ Target marker {marker_id} not detected.")
            continue

        if marker_id not in GROUND_TRUTH:
            print(f"⚠ No ground truth for marker {marker_id}.")
            continue

        idx = np.where(sorted_ids == marker_id)[0][0]
        u, v = sorted_centers[idx]

        # --- Image coordinates ---
        print(f"Marker {marker_id} image coordinates (pixels): [{u:.1f}, {v:.1f}]")

        # --- Back-project to world (Z=0 plane) ---
        uv_h = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
        ray_cam = np.linalg.inv(camera_matrix) @ uv_h

        Rcw = R.T
        tcw = -R.T @ tvec
        ray_world = Rcw @ ray_cam

        s = -tcw[2, 0] / ray_world[2, 0]
        Pw = tcw + s * ray_world

        Xw_est, Yw_est = Pw[0, 0], Pw[1, 0]
        Xw_gt, Yw_gt = GROUND_TRUTH[marker_id]

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


    # --- Draw markers ---
    for i, marker_id in enumerate(ids.flatten()):
        c = corners[i][0]
        center_x = int(c[:, 0].mean())
        center_y = int(c[:, 1].mean())
        center = (center_x, center_y)
        cv2.polylines(img, [c.astype(int)], True, (0, 255, 0), 3)
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        cv2.putText(img, f"{marker_id}", (center_x + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # --- Show result ---
    cv2.imshow("AprilTag Pose", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
