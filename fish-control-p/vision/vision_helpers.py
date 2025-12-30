import cv2
import numpy as np
import platform

def open_camera(index=0):
    system = platform.system()

    if system == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    elif system == "Darwin":  # macOS
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
    imgpts = imgpts.reshape(-1, 2)
    imgpts = [tuple(map(int, pt)) for pt in imgpts]

    img = cv2.line(img, corner, tuple(imgpts[0]), (255, 0, 0), 5)  # X - red
    img = cv2.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 5)  # Y - green
    img = cv2.line(img, corner, tuple(imgpts[2]), (0, 0, 255), 5)  # Z - blue
    return img

from dataclasses import dataclass

@dataclass
class FishState:
    u: int | None
    v: int | None
    x: float | None
    y: float | None
    yaw: float | None
    surge: float | None
    sway: float | None
    yaw_rate: float | None

def draw_fish_state(img, found, state, axis_length=50):

    origin = (740, 30)

    if not found:
        cv2.putText(
            img, "Fish not detected",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return img

    u, v = int(state.u), int(state.v)

    # --- Draw centroid ---
    cv2.circle(img, (u, v), 6, (255, 0, 0), -1)

    # --- Draw heading axis ---
    angle_disp = (2 * np.pi - state.yaw) % (2 * np.pi) # Pass from [0 2pi] to [-pi pi]
    x2 = int(u + axis_length * np.cos(angle_disp))
    y2 = int(v + axis_length * np.sin(angle_disp))
    cv2.line(img, (u, v), (x2, y2), (255, 0, 0), 3)

    # --- Overlay text ---
    lines = [
        f"Pos   : ({state.x:.1f}, {state.y:.1f}) m",
        f"Yaw   : {np.degrees(state.yaw):.1f} deg",
        f"Surge : {state.surge:.2f} m/s",
        f"Sway  : {state.sway:.2f} m/s",
        f"Yaw rate : {np.degrees(state.yaw_rate):.1f} deg/s",
    ]

    y_text = origin[1]
    for line in lines:
        cv2.putText(
            img,
            line,
            (origin[0], y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )
        y_text += 24

    return img