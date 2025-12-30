import os

script_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np

# -------------------------------
# Parameters
# -------------------------------
IMG_NAME = "fish.jpg"
IMG_NAME = "pool_fish2.jpg"
SHOW_BLUR = True
SHOW_MORPH = True
SHOW_THRESHOLDS = True
SHOW_CONTOURS = True

# Algorithm tuning
GAUSSIAN_BLUR_SIZE = (5, 5)

THRESHOLDING_BLOCK_SIZE = 251
THRESHOLDING_C = 20

MORPH_KERNEL = cv2.MORPH_ELLIPSE
MORPH_SIZE = (7, 7)
MORPH_NUM_ITER = 1

AREA_MIN = 500
AREA_MAX = 50000
# -------------------------------

# --- Load image ---
img = cv2.imread(os.path.join(script_dir, "imgs", IMG_NAME))

x1, y1 = 181, 53
x2, y2 = 1099, 659
img_crop = img[y1:y2, x1:x2].copy()

gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

# --- Gaussian blur ---
blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, 0)

if SHOW_BLUR:
    cv2.imshow('Blur image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Thresholding ---
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=THRESHOLDING_BLOCK_SIZE,
    C=THRESHOLDING_C
)

if SHOW_THRESHOLDS:
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Morphology ---
kernel = cv2.getStructuringElement(MORPH_KERNEL, MORPH_SIZE)
morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=MORPH_NUM_ITER)

if SHOW_MORPH:
    cv2.imshow('Morphological opening', morph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Find contours ---
contours, hierarchy = cv2.findContours(
    image=morph.astype(np.uint8),
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_NONE
)

fish_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if AREA_MIN < area < AREA_MAX:
        fish_contours.append(cnt)

# --- Combine all contour points ---
if not fish_contours:
    print("No fish detected")
    exit()

all_points = np.vstack(fish_contours).reshape(-1, 2) 
data_pts = all_points.astype(np.float32)

# --- PCA ---
mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)
principal_vec = eigenvectors[0] # First principal component (legth direction)
perp_vec = eigenvectors[1] 

u, v = mean[0]  

# Get orientation
relative_pts = data_pts - mean[0]

projections_along  = np.dot(relative_pts, principal_vec)
length_pixels = projections_along .max() - projections_along .min()

num_samples = 10
sample_positions = np.linspace(projections_along .min(),  projections_along .max(), num_samples)
sample_widths = []

tolerance = 10

for s in sample_positions:
    mask = np.abs(projections_along  - s) < tolerance
    if np.any(mask):
        local_pts = relative_pts[mask]

        perp_proj = np.dot(local_pts, perp_vec)
        width = perp_proj.max() - perp_proj.min()
    else:
        width = 0
    sample_widths.append(width)

    left_width = sample_widths[0:num_samples//2]
    right_width = sample_widths[num_samples//2:]

if left_width and right_width:
    mean_left_width = np.mean(left_width)
    mean_right_width = np.mean(right_width)

if mean_right_width > mean_left_width:
    principal_vec = -principal_vec

print("Fish length (pixels):", length_pixels)
print("Fish widths at sampled positions (pixels):", ["{:.2f}".format(w) for w in sample_widths])

print(f"Fish centroid (u, v): ({u:.2f}, {v:.2f})")

angle = np.arctan2(principal_vec[1], principal_vec[0])
angle_deg = (-1*(angle * 180.0 / np.pi))%360
print("Fish orientation (degrees):", angle_deg)

# --- Draw centroid ---
cv2.circle(img_crop, (int(u), int(v)), 6, (0, 0, 255), -1)

# --- Draw orientation axis ---
length = 150
x2 = int(u + length * principal_vec[0])
y2 = int(v + length * principal_vec[1])
cv2.line(img_crop, (int(u), int(v)), (x2, y2), (255, 0, 0), 3)

# --- Show results ---
if SHOW_CONTOURS:
    cv2.drawContours(image=img_crop, contours=fish_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('Contours', img_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
