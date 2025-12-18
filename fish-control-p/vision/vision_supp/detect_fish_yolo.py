import os

script_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
from ultralytics import YOLO

# -------------------------------
# Parameters
# -------------------------------
CAMERA_INDEX = 0
USE_CAMERA = False
IMG_NAME = "pool_fish.png"
MODEL_NAME = "yolo11s.pt" # 
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
    print("Press 'q' to quit")

    for r in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("👋 Webcam detection stopped")
else:
    img = cv2.imread(os.path.join(script_dir, "imgs", IMG_NAME))
    results = model.predict(source=img)

    for r in results:
        print(r)
        r.show()
    