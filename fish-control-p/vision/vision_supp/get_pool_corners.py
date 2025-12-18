import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

import cv2
from vision_helpers import open_camera

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = True
IMG_NAME = "pool_test.jpg"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
# -------------------------------

# --- Load image or open camera ---
if not USE_CAMERA:
    try:
        img = cv2.imread(os.path.join(script_dir, "imgs", IMG_NAME))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("Failed to get image")
        exit()
else:
    cap = open_camera(CAMERA_INDEX)

# --- Detection loop ---
while True:

    if USE_CAMERA:
        ret, img = cap.read()
        if not ret:
            print("Failed to get frame.")
            exit()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        if not USE_CAMERA:
            print("No markers detected.")
            break
    else:
        # Draw each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]

            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            center = (center_x, center_y)

            if not USE_CAMERA:
                print(f"Marker ID {marker_id}: center = {center}")

            cv2.polylines(img, [c.astype(int)], True, (0, 255, 0), 3)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            cv2.putText(img, f"{marker_id}", (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Detected markers", img)

    if not USE_CAMERA:
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
if USE_CAMERA:
    cap.release()
cv2.destroyAllWindows()
