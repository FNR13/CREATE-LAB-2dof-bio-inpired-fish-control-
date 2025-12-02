import numpy as np
import os
import cv2

use_photos = True
img_name = "4corners.png"

# Markers initialization
# --- AprilTag detection setup ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11) # DICT_APRILTAG_36h11
parameters = cv2.aruco.DetectorParameters()

# IDs of known markers
KNOWN_MARKERS = [0, 1, 2]

marker_positions_3D = np.array([
    [0.00, 0.00, 0.00],   # ID 0
    [0.05, 0.00, 0.00],   # ID 1
    [0.00, 0.05, 0.00],   # ID 2
], dtype=np.float32)

# Get id 3 and 4 poses
TARGET_MARKERS = [3,4]

def drawAxes(img, corner, imgpts):
    corner = tuple(int(x) for x in corner.ravel())
    imgpts = imgpts.reshape(-1, 2)  # (3,2)

    imgpts = [tuple(map(int, pt)) for pt in imgpts]
    img = cv2.line(img, corner, tuple(imgpts[0]), (255,0,0), 5)  # X - red
    img = cv2.line(img, corner, tuple(imgpts[1]), (0,255,0), 5)  # Y - green
    img = cv2.line(img, corner, tuple(imgpts[2]), (0,0,255), 5)  # Z - blue
    return img

def getSortedCenters(corners, ids):
    ids_flat = ids.flatten()

    centers = []
    for c in corners:
        pts = c[0]            
        centers.append(pts.mean(axis=0))

    centers = np.array(centers, dtype=np.float32)

    sort_index = np.argsort(ids_flat)

    return centers[sort_index], ids_flat[sort_index]

def poseEstimation(img):
    # --- Load camera calibration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(script_dir, "camera_calib.npz")

    calib_data = np.load(calib_path)
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

    axis = np.float32([[0.25,0,0], [0,0.25,0], [0,0,0.00]])

    # Pose estimation
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    corners, ids, _ = cv2.aruco.detectMarkers(imgGray, aruco_dict, parameters=parameters)

    if ids is None or len(ids) < 3:
        print("Not enough markers detected")
        return

    sorted_centers, sorted_ids = getSortedCenters(corners, ids)

   # Build a dictionary id -> center
    center_dict = {id_: center for id_, center in zip(sorted_ids, sorted_centers)}

    # Extract centers in KNOWN_MARKERS order
    known_centers = np.array([center_dict[m] for m in KNOWN_MARKERS if m in center_dict], dtype=np.float32)
    known_ids = [m for m in KNOWN_MARKERS if m in center_dict]

    retval, rvecs, tvecs = cv2.solveP3P(
        marker_positions_3D,
        known_centers.reshape(-1,1,2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_P3P
    )

    print(f"Found {retval} solution(s)")
    
    if retval > 0:
        rvec, tvec = rvecs[0], tvecs[0]
        imgpts,_ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        first_index = np.where(sorted_ids == KNOWN_MARKERS[0])[0][0]
        img = drawAxes(img, sorted_centers[first_index], imgpts)

        R, _ = cv2.Rodrigues(rvec)


    for marker_id in TARGET_MARKERS:
        # Find index of marker in sorted_ids
        idx = np.where(sorted_ids == marker_id)[0][0]
        
        # Get 2D image coordinates
        u, v = sorted_centers[idx]
        print(f"Marker {marker_id} image coordinates (pixels): [{u}, {v}]")

        # Compute world coordinates on plane Z=0
        uv_h = np.array([u, v, 1.0], dtype=np.float32).reshape(3,1)
        ray_cam = np.linalg.inv(camera_matrix) @ uv_h  # normalized ray in camera frame

        # Camera origin in world frame
        Rcw = R.T
        tcw = -R.T @ tvec  # camera origin in world frame

        # Ray direction in world frame
        ray_world = Rcw @ ray_cam

        # Scale so that Z_world = 0
        s = -tcw[2,0] / ray_world[2,0]
        Pw = tcw + s * ray_world

        
        print(f"Marker {marker_id} position in WORLD frame (Z=0 plane):", Pw.ravel())
        
    cv2.imshow("AprilTag Pose", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if use_photos:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img = cv2.imread(os.path.join(script_dir, "imgs", img_name))
    else:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        ret, frame = cap.read()
        img = frame
    poseEstimation(img)

if __name__ == "__main__":
    main()