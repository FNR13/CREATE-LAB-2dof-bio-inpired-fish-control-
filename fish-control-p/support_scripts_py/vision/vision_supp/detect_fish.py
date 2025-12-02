import numpy as np
import os
import cv2

img_name = "fish.png"
show_threshold = False
show_contours = True

# Algo tuning
binary_min = 125
binary_max = 255
minArea = 500  
maxArea = 50000  

script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(script_dir, "imgs", img_name))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#-----------------------------------------------------------
## Algorithym
# Thresholding
ret, thresh = cv2.threshold(img_gray, binary_min, binary_max, cv2.THRESH_BINARY)

if show_threshold:
    # visualize the binary image
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()


# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

fish_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if minArea < area < maxArea:
        fish_contours.append(cnt)

# Combine all contour points into ONE array
all_points = np.vstack(fish_contours).reshape(-1, 2)

# all_points is Nx2 array of contour coordinates (float32 recommended)
data_pts = all_points.astype(np.float32)

# PCA
mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)

cx, cy = mean[0]   # centroid (u, v)
u = float(cx)
v = float(cy)

print(f"Fish centroid (u, v): ({u:.2f}, {v:.2f})")

# Orientation angle (radians ➜ degrees)
angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
angle_deg = angle * 180.0 / np.pi

print("Fish orientation (degrees):", angle_deg)

# ---- DRAW CENTROID ----
cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)

# ---- DRAW ORIENTATION AXIS ----
length = 150  # length of axis line
x2 = int(u + length * eigenvectors[0,0])
y2 = int(v + length * eigenvectors[0,1])
cv2.line(img, (int(u), int(v)), (x2, y2), (255, 0, 0), 3)

if show_contours:
    # cv2.drawContours(image=img, contours=fish_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # see the results
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

