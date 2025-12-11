import cv2

# -------------------------------
# Parameters
# -------------------------------
MAX_TEST_CAMERAS = 5   
# -------------------------------

# --- Test camera indices for availability ---
available_cameras = []

for cam_index in range(MAX_TEST_CAMERAS):
    cap = cv2.VideoCapture(cam_index)

    if cap is None or not cap.isOpened():
        cap.release()
        continue

    available_cameras.append(cam_index)
    cap.release()

# --- Output result ---
print("Available cameras:", available_cameras)
