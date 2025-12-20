import cv2
from ultralytics import YOLO
import numpy as np
import math

try:
    from .vision_helpers import open_camera, getCenters, drawAxes

    from .vision_config import (
        CALIBRATION,
        MARKERS,
        YOLO_CFG,
        FISH_DETECTION,
        POOL_HEIGHT_M, POOL_WIDTH_M,
        DRAWING,
    )
except: 
    from vision_helpers import open_camera, getCenters, drawAxes

    from vision_config import (
        CALIBRATION,
        MARKERS,
        YOLO_CFG,
        FISH_DETECTION,
        POOL_HEIGHT_M, POOL_WIDTH_M,
        DRAWING,
    )

class Fish_Vision:
    def __init__(self, camera_index, delta_t=0.1):
        
        # Vision algorithm
        calib_data = np.load(CALIBRATION["path"])
        self.camera_matrix = calib_data["camera_matrix"]
        self.dist_coeffs = calib_data["dist_coeffs"]

        self.model = YOLO(YOLO_CFG["model_path"])
        self.yolo_conf = YOLO_CFG["confidence"]
        self.yolo_target_id = None
        for k, v in self.model.names.items():
            if v == YOLO_CFG["target"]:
                self.yolo_target_id = k
                break
            
        # Pose estimation parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MARKERS["dictionary"])
        self.aruco_params = MARKERS["parameters"]

        self.known_markers = MARKERS["known_ids"]
        self.marker_positions_3D = MARKERS["positions_3D"]

        self.rvec = None
        self.tvec = None
        self.R = None
        self.calibrated = False

        self.pool_pixels = None

        self.cap = open_camera(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # State
        self.delta_t = delta_t
        self.last_x = None
        self.last_y = None
        self.last_yaw = None

    def calibrate(self, img=False, show_output=False):
        """Uses 3 markers with known positions to calculate translation and rotation matrices"""

        if img is False:
            ret, img = self.cap.read()
            if not ret:
                print("[VISION - ERROR] Failed to grab frame from camera")
                return False
            
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            imgGray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None:
            print("[VISION - ERROR] Markers not detected")
            return False
        
        centers, ids = getCenters(corners, ids)

        # Extract ordered centers for known markers
        center_dict = {id_: center for id_, center in zip(ids, centers)}
        known_centers = np.array(
            [center_dict[m] for m in self.known_markers if m in center_dict],
            dtype=np.float32
        )

        if len(known_centers) < 3:
            print("[VISION - ERROR]All calibration markers were not detected")
            return False
    
        retval, rvecs, tvecs = cv2.solveP3P(
            self.marker_positions_3D,
            known_centers.reshape(-1,1,2),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_P3P
        )

        if retval == 0:
            print("[VISION - ERROR]No valid P3P solution")
            return False

        self.rvec = rvecs[0]
        self.tvec = tvecs[0]
        self.R, _ = cv2.Rodrigues(self.rvec)

        self.calibrated = True

        self.pool_pixels = [self.project_world_to_pixel(0, POOL_HEIGHT_M),
                            self.project_world_to_pixel(POOL_WIDTH_M, 0)]      

        if show_output:
                img_copy = img.copy()
                # Draw axis
                axis_length = int(DRAWING["axis_length_meters"])
                
                axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, 0]])
                axis = np.float32([[0.50, 0, 0], [0, 0.50, 0], [0, 0, 0]])
                imgpts, _ = cv2.projectPoints(axis, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
                img_copy = drawAxes(img_copy, known_centers[0], imgpts)
            
                # Draw markers
                for i, marker_id in enumerate(ids.flatten()):
                    c = corners[i][0]
                    center_x = int(c[:, 0].mean())
                    center_y = int(c[:, 1].mean())
                    center = (center_x, center_y)

                    cv2.polylines(img_copy, [c.astype(int)], True, (0, 255, 0), 3)
                    cv2.circle(img_copy, center, 5, (0, 0, 255), -1)
                    cv2.putText(img_copy, f"{marker_id}", (center_x + 10, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Calibration', img_copy)
                cv2.waitKey(0)
                    
        return True

    def project_world_to_pixel(self, X, Y, Z=0.0):
        """Projects world point (X, Y, Z) to image pixel (u, v)."""

        if not self.calibrated:
            print("[VISION - ERROR]Please calibrate first")
            return (0, 0)

        # World point
        Pw = np.array([[X, Y, Z]], dtype=np.float32)

        # Project using OpenCV
        imgpts, _ = cv2.projectPoints(
            Pw,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        u, v = imgpts[0, 0]
        return int(round(u)), int(round(v))

    def project_pixel_to_world(self, u, v):
        """Projects camera pixel to z=0 plane in world coordinates."""

        if not self.calibrated:
            print("[VISION - ERROR]Please calibrate first")
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

        if not self.calibrated:
            print("[VISION - ERROR]Please calibrate first")
            return (0, 0, 0)

        if img.shape[:2] == (720, 1280):
            # Crop to pool area
            x1, y1 = self.pool_pixels[0]
            x2, y2 = self.pool_pixels[1]

            # Reorder for protection 
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            img_analysis = img[y1:y2, x1:x2]
            croped = True
        else:
            img_analysis = img
            croped = False

        cfg_t = FISH_DETECTION["threshold"]
        cfg_c = FISH_DETECTION["contours"]

        img_gray = cv2.cvtColor(img_analysis, cv2.COLOR_BGR2GRAY)

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
            print("[VISION - ERROR]No valid fish detected.")
            return None, None, None

        data_pts = np.vstack(fish_contours).reshape(-1, 2).astype(np.float32)

        # PCA to determine centroid and orientation
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)

        u, v  = mean[0] 
        if croped:
            u += x1
            v += y1

        angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])

        if show_output:
            img_copy = img.copy()
            cv2.circle(img_copy, (int(u), int(v)), 6, (0, 0, 255), -1)

            axis_length = DRAWING["axis_length_pixels"]
            x2 = int(u + axis_length * -eigenvectors[0,0])
            y2 = int(v + axis_length * -eigenvectors[0,1])
            cv2.line(img_copy, (int(u), int(v)), (x2, y2), (255, 0, 0), 3)
            
            if croped:
                offset = np.array([x1, y1])
                fish_contours_full = [cnt + offset for cnt in fish_contours]
            else:
                fish_contours_full = fish_contours

            cv2.drawContours(img_copy, fish_contours_full, -1, (0,255,0), 2)

            cv2.imshow('Contours', img_copy)
            cv2.waitKey(0)
        return int(u), int(v), angle
    
    def detect_fish_yolo(self, img, show_output=False):
        """
        Detects the fish using YOLO to get a bounding box, then applies the
        existing contour + PCA method.
        """

        if not self.calibrated:
            print("[VISION - ERROR]Please calibrate first")
            return (0, 0, 0)
        
        results = self.model.predict(img, conf=self.yolo_conf, verbose=False)
        detections = results[0].boxes

        mask = detections.cls.cpu().numpy() == self.yolo_target_id
        filtered_dets = detections[mask]

        if len(filtered_dets) == 0:
            print("[VISION - ERROR] No fish detected in YOLO.")
            return None, None, None

        # Pick highest confidence detection
        confs = filtered_dets.conf.cpu().numpy()
        best_idx = np.argmax(confs)
        best_det = filtered_dets[best_idx]

        # Extract bounding box
        box = best_det.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            print("[VISION - ERROR] Invalid crop region.")
            return None, None, None

        u_local, v_local, angle = self.detect_fish(crop, show_output=False)

        if u_local is None:
            print("[VISION - ERROR] No fish detected in contours.")
            return None, None, None

        # Convert local centroid → full image coordinates
        u_full = int(x1 + u_local)
        v_full = int(y1 + v_local)

        if show_output:
            img_copy = img.copy()
            # Draw YOLO box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw centroid
            cv2.circle(img_copy, (u_full, v_full), 6, (0, 0, 255), -1)

            # Draw orientation axis
            axis_length = DRAWING["axis_length_pixels"]
            x2_axis = int(u_full - axis_length * np.cos(angle))
            y2_axis = int(v_full - axis_length * np.sin(angle))
            cv2.line(img_copy, (u_full, v_full), (x2_axis, y2_axis), (255, 0, 0), 3)

            cv2.imshow("YOLO + PCA Fish Detection", img_copy)
            cv2.waitKey(0)

        return u_full, v_full, angle

    def get_fish_state(self, use_yolo=False, get_output=False):

        ret, img = self.cap.read()
        if not ret:
            print("[VISION - ERROR] Failed to grab frame")
            return
        
        if not use_yolo:
            u, v, angle = self.detect_fish(img)
        else: 
            u, v, angle = self.detect_fish_yolo(img)

        yaw = angle
        x, y = self.project_pixel_to_world(u, v)

        if self.last_x==None or self.last_y==None or self.last_yaw==None:
            return x, y, yaw, 0, 0,  # return surge and yaw rate as 0 for initialization
        else:
            distance = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            surge = distance/self.delta_t
            yaw_rate = (yaw - self.last_yaw)/self.delta_t

        # Update last known state
        self.last_x = x
        self.last_y = y
        self.last_yaw = yaw

        # Draw results if requested
        if get_output:

            # Draw fish/robot centroid
            cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)

            # Draw orientation arrow
            axis_length = DRAWING["axis_length_pixels"]
            x2 = int(u - axis_length * math.cos(yaw))
            y2 = int(v - axis_length * math.sin(yaw))
            cv2.arrowedLine(img, (int(u), int(v)), (x2, y2), (255, 0, 0), 3, tipLength=0.2)

            # Draw text: position, surge, yaw_rate
            text_lines = [
                f"X: {x:.2f} m",
                f"Y: {y:.2f} m",
                f"Surge: {surge:.2f} m/s",
                f"Yaw rate: {yaw_rate:.2f} rad/s"
            ]
            for i, line in enumerate(text_lines):
                cv2.putText(
                    img,
                    line,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        print("State:", x, y, yaw, surge, yaw_rate)

        return x, y, yaw, surge, yaw_rate, img



