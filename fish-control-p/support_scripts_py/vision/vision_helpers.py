import numpy as np
import cv2

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
