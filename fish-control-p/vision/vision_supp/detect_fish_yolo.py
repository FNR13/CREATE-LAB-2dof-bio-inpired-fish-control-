import os

script_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
from ultralytics import YOLO

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = False
USE_VIDEO = True
VIDEO_NAME = "pool_test2.mp4"
IMG_NAME = "fish.jpg"
MODEL_NAME = "yolo11l.pt"
# -------------------------------

model = YOLO(os.path.join(script_dir, "YOLO_models", MODEL_NAME))

print("Model loaded")

if USE_CAMERA:
    results = model.predict(
        source=CAMERA_INDEX,  
        show=True,  
        stream=True,  
        verbose=False
    )

    print("📹 Starting webcam detection...")
    print("Press 'Esc' to quit")

    for r in results:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    print("👋 Webcam detection stopped")

elif USE_VIDEO:
    video_path = os.path.join(script_dir, "..", "media",  VIDEO_NAME)
    results = model.predict(
        source=video_path,  
        show=True,  
        stream=True,  
        verbose=False
    )

    print(f"🎥 Starting video detection on {video_path}...")
    print("Press 'Esc' to quit")

    for r in results:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    print("👋 Video detection stopped")
else:
    img = cv2.imread(os.path.join(script_dir, "imgs", IMG_NAME))
    results = model.predict(source=img)

    for r in results:
        print(r)
        r.show()
    