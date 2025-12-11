import os
import cv2
import numpy as np

# Get supp path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT_DIR = os.path.join(SCRIPT_DIR, "vision_supp")


CALIBRATION = {
    "filename": "camera_calib.npz",
    "path": os.path.join(SUPPORT_DIR, "camera_calib.npz")
}


MARKERS = {
    "dictionary": cv2.aruco.DICT_APRILTAG_36h11,
    "parameters": cv2.aruco.DetectorParameters(),

    "known_ids": [1, 4, 3],
    "target_ids": [0, 2],
    # "known_ids": [0, 1, 2],
    # "target_ids": [3, 4],

    "positions_3D": np.array([
        [0.00, 0.00, 0.00],   # ID 0
        [2.00, 0.00, 0.00],   # ID 1
        [2.00, 1.00, 0.00],   # ID 2
    ], dtype=np.float32)
}


YOLO_CFG = {
    "model_name": "yolo11s.pt",
    "model_path": os.path.join(SUPPORT_DIR, "YOLO_MODELS", "yolo11s.pt"),
    "confidence": 0.25,
}


FISH_DETECTION = {
    "threshold": {
        "method": "binary",   # "binary" / "otsu" / "adaptive"
        "binary_min": 125,
        "binary_max": 255,

        # Adaptive threshold parameters
        "adaptive_block": 11,
        "adaptive_C": 2
    },

    "contours": {
        "min_area": 500,
        "max_area": 50000
    }
}
