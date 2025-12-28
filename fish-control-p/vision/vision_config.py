import os
import cv2
import numpy as np

# Get supp path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT_DIR = os.path.join(SCRIPT_DIR, "vision_supp")


CALIBRATION = {
    "path": os.path.join(SUPPORT_DIR, "camera calibration data", "camera_calib_1_5.npz")
}


POOL_WIDTH_M = 2.75
POOL_HEIGHT_M = 1.75

MARKERS = {
    "dictionary": cv2.aruco.DICT_APRILTAG_36h11,
    "parameters": cv2.aruco.DetectorParameters(),

    "known_ids": [0, 1, 2],

    # this depends on how you setup the markers, this is aligned for a normal xy plane with camera view (which is y-inverted to camera axis)
    "positions_3D": np.array([
        [0.00, 0.00, 0.00],          # ID 0
        [0.00, POOL_HEIGHT_M, 0.00], # ID 1
        [POOL_WIDTH_M, 0.00, 0.00],  # ID 2
    ], dtype=np.float32)
}


YOLO_CFG = {
    "model_name": "yolo11s.pt",
    "model_path": os.path.join(SUPPORT_DIR, "YOLO_MODELS", "yolo11s.pt"),
    "confidence": 0.25,
    "target"    : "airplane" 
}


FISH_DETECTION = {
    "Filtering": {
        "gaussian_blur_size": (5, 5),
    },

    "threshold": {
        "method": "adaptive",  # "binary" or "adaptive"
        "binary_min": 125,
        "binary_max": 255,

        "adaptive_block": 251,
        "adaptive_C": 20,

    },

    "morphology": {
        "kernel": cv2.MORPH_ELLIPSE,
        "size": (7, 7),
        "iterations": 1
    },

    "contours": {
        "min_area": 500,
        "max_area": 50000
    }
}

DRAWING = {
        "axis_length_pixels": 50,
        "axis_length_meters": 0.50,
}

