import cv2

CAMERA_INDEX = 1

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}")
    exit()

print(f"Camera {CAMERA_INDEX} opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Feed", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
