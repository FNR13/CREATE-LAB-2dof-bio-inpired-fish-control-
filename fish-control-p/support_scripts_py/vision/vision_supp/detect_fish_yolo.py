import cv2
from ultralytics import YOLO
import os

# Parameters
model_name = "yolo11s.pt" # 
camera = 0
use_camera = False

img_name = "fish.png"

#-----------------------------------------------------------
# model = YOLO(model_name) 
script_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(script_dir, "YOLO_models", model_name))

print("Model loaded")

if use_camera:
    # Use webcam (source=0 means default camera)
    results = model.predict(
        source=camera,  # Webcam
        show=True,  # Show live feed
        stream=True,  # Real-time streaming
        verbose=False  # Less terminal output
    )

    print("📹 Starting webcam detection...")
    print("Press 'q' to quit")

    # Process live stream
    for r in results:
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("👋 Webcam detection stopped")
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(os.path.join(script_dir, "imgs", img_name))
    results = model.predict(source=img)

    for r in results:
        r.show()
    