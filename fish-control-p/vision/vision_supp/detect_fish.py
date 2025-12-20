import os

script_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np

# -------------------------------
# Parameters
# -------------------------------
IMG_NAME = "fish.png"
SHOW_THRESHOLDS = True
SHOW_CONTOURS = True

# Algorithm tuning
BINARY_MIN = 125
BINARY_MAX = 255
AREA_MIN = 500
AREA_MAX = 50000
# -------------------------------

# --- Load image ---
img = cv2.imread(os.path.join(script_dir, "imgs", IMG_NAME))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Thresholding ---
ret, thresh = cv2.threshold(img_gray, BINARY_MIN, BINARY_MAX, cv2.THRESH_BINARY)

if SHOW_THRESHOLDS:
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()

# --- Find contours ---
contours, hierarchy = cv2.findContours(
    image=thresh,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_NONE
)

fish_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if AREA_MIN < area < AREA_MAX:
        fish_contours.append(cnt)

# --- Combine all contour points ---
all_points = np.vstack(fish_contours).reshape(-1, 2)
data_pts = all_points.astype(np.float32)

# --- PCA ---
mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)
u, v = mean[0]   # centroid (u, v)

print(f"Fish centroid (u, v): ({u:.2f}, {v:.2f})")

# Orientation angle (radians ➜ degrees)
angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
angle_deg = angle * 180.0 / np.pi
print("Fish orientation (degrees):", angle_deg)

# --- Draw centroid ---
cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)

# --- Draw orientation axis ---
length = 150
x2 = int(u + length * -eigenvectors[0,0])
y2 = int(v + length * -eigenvectors[0,1])
cv2.line(img, (int(u), int(v)), (x2, y2), (255, 0, 0), 3)

# --- Show results ---
if SHOW_CONTOURS:
    # cv2.drawContours(image=img, contours=fish_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
