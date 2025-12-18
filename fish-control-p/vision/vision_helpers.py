import cv2
import numpy as np
import platform

def open_camera(index=0):
    system = platform.system()

    if system == "Windows":
        # Use DirectShow (CAP_DSHOW) or Media Foundation (CAP_MSMF)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Linux":
        # Use Video4Linux2
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    elif system == "Darwin":  # macOS
        # Use AVFoundation
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        # Fallback: let OpenCV choose default backend
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index} on {system}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def getCenters(corners, ids):
    ids_flat = ids.flatten()

    centers = np.array([
        c[0].mean(axis=0) for c in corners
    ], dtype=np.float32)

    return centers, ids_flat

def drawAxes(img, corner, imgpts):
    corner = tuple(int(x) for x in corner.ravel())
    imgpts = imgpts.reshape(-1, 2)  # (3,2)

    imgpts = [tuple(map(int, pt)) for pt in imgpts]
    img = cv2.line(img, corner, tuple(imgpts[0]), (255,0,0), 5)  # X - red
    img = cv2.line(img, corner, tuple(imgpts[1]), (0,255,0), 5)  # Y - green
    img = cv2.line(img, corner, tuple(imgpts[2]), (0,0,255), 5)  # Z - blue
    return img
