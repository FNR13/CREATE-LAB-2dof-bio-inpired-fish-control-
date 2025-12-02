import cv2

def list_cameras(max_tests=5):
    available = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        available.append(i)
        cap.release()
    return available

cams = list_cameras()
print("Available cameras:", cams)
