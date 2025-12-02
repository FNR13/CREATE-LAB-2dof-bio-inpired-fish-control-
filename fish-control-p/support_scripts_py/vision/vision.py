import cv2
from ultralytics import YOLO
import numpy as np

from vision_helpers import drawAxes, getCenters

from vision_config import (
    CALIBRATION,
    MARKERS,
    YOLO_CFG,
    FISH_DETECTION,
)

class Fish_Vision:
    def __init__(self, camera_index):
        
        calib_data = np.load(CALIBRATION["path"])
        self.camera_matrix = calib_data["camera_matrix"]
        self.dist_coeffs = calib_data["dist_coeffs"]

        self.model = YOLO(YOLO_CFG["model_path"])
        self.yolo_conf = YOLO_CFG["confidence"]

        # Pose estimation parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKERS["dictionary"])
        self.aruco_params = MARKERS["parameters"]

        self.known_markers = MARKERS["known_ids"]
        self.target_markers = MARKERS["target_ids"]
        self.marker_positions_3D = MARKERS["positions_3D"]

        self.rvec = None
        self.tvec = None
        self.R = None
        self.calibrated = False

        try:
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        except:
            self.cap = cv2.VideoCapture(camera_index)

    def calibrate(self, img):
        """Uses 3 markers with known positions to calculate translation and rotation matrices"""

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            imgGray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None:
            print("Markers not detected")
            return False
        
        centers, ids = getCenters(corners, ids)

        # Extract ordered centers for known markers
        center_dict = {id_: center for id_, center in zip(ids, centers)}
        known_centers = np.array(
            [center_dict[m] for m in self.known_markers if m in center_dict],
            dtype=np.float32
        )

        if len(known_centers) < 3:
            print("All calibration markers were not detected")
            return False
        
        retval, rvecs, tvecs = cv2.solveP3P(
            self.marker_positions_3D,
            known_centers.reshape(-1,1,2),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_P3P
        )

        if retval == 0:
            print("No valid P3P solution")
            return False

        self.rvec = rvecs[0]
        self.tvec = tvecs[0]
        self.R, _ = cv2.Rodrigues(self.rvec)

        self.calibrated = True
        return True

    def get_real_world_position(self, u, v):
        """Projects camera pixel to z=0 plane in world coordinates."""

        if not self.calibrated:
            print("Please calibrate first")
            return (0, 0)
    
        # Project a ray from the camera pixel to the z=0 plane
        uv_h = np.array([u, v, 1.0], dtype=np.float32).reshape(3,1)
        ray_cam = np.linalg.inv(self.camera_matrix) @ uv_h

        Rcw = self.R.T
        tcw = -Rcw @ self.tvec

        ray_world = Rcw @ ray_cam
        s = -tcw[2,0] / ray_world[2,0]
        Pw = tcw + s * ray_world

        return float(Pw[0]), float(Pw[1])

    def detect_fish(self, img, show_output=False):
        """
        Detects the fish using global thresholding + contour filtering + PCA.
        """

        cfg_t = FISH_DETECTION["threshold"]
        cfg_c = FISH_DETECTION["contours"]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(
            img_gray, 
            cfg_t["binary_min"], cfg_t["binary_max"], 
            cv2.THRESH_BINARY
        )

        contours, hierarchy = cv2.findContours(
            image=thresh, 
            mode=cv2.RETR_TREE, 
            method=cv2.CHAIN_APPROX_NONE
        )

        fish_contours = []

        # Filter low size contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if cfg_c["min_area"] < area < cfg_c["max_area"]:
                fish_contours.append(cnt)

        if len(fish_contours) == 0:
            print("No valid fish detected.")
            return None, None, None

        data_pts = np.vstack(fish_contours).reshape(-1, 2).astype(np.float32)

        # PCA to determine centroid and orientation
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)

        cx, cy = mean[0]  
        u, v = float(cx), float(cy)

        angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])

        if show_output:
            cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)

            axis_length = 150 
            x2 = int(u + axis_length * eigenvectors[0,0])
            y2 = int(v + axis_length * eigenvectors[0,1])
            cv2.line(img, (int(u), int(v)), (x2, y2), (255, 0, 0), 3)
            
            cv2.drawContours(img, fish_contours, -1, (0, 255, 0), 2)
            
            cv2.imshow('Contours', img)
            cv2.waitKey(0)
        return u, v, angle
    
    def detect_fish_yolo(self, img, show_output=False):
        """
        Detects the fish using YOLO to get a bounding box, then applies the
        existing contour + PCA method.
        """

        results = self.model.predict(img, conf=self.yolo_conf, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            print("YOLO: No fish detected.")
            return None, None, None

        # Take the highest confidence detection
        box = detections[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            print("Invalid crop region.")
            return None, None, None

        u_local, v_local, angle = self.detect_fish(crop, show_output=False)

        if u_local is None:
            return None, None, None

        # Convert local centroid → full image coordinates
        u_full = x1 + u_local
        v_full = y1 + v_local

        if show_output:
            # Draw YOLO box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw centroid
            cv2.circle(img, (int(u_full), int(v_full)), 6, (0, 0, 255), -1)

            # Draw orientation axis
            axis_length = 150
            x2_axis = int(u_full + axis_length * np.cos(angle))
            y2_axis = int(v_full + axis_length * np.sin(angle))
            cv2.line(img, (int(u_full), int(v_full)), (x2_axis, y2_axis), (255, 0, 0), 3)

            cv2.imshow("YOLO + PCA Fish Detection", img)
            cv2.waitKey(0)

        return u_full, v_full, angle





