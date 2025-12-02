import cv2
import os


use_photos = False
img_name = "4corners.png"

CAMERA_INDEX = 0

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # DICT_APRILTAG_36h11
parameters = cv2.aruco.DetectorParameters()

if use_photos:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img = cv2.imread(os.path.join(script_dir, "imgs", img_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("Failed to get image")
        exit()
else:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open camera.")
        exit()

while True:

    if not use_photos:
        ret, img = cap.read()

        if not ret:
            print("Failed to get frame .")
            exit()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        if use_photos:
            print("No markers detected.")
            break
    else:
        # draw each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]

            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            center = (center_x, center_y)

            if use_photos:
                print(f"Marker ID {marker_id}: center = {center}")

            cv2.polylines(img, [c.astype(int)], True, (0, 255, 0), 3)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            cv2.putText(img, f"{marker_id}", (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Detected markers", img)

    if use_photos:
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not use_photos:
    cap.release()
cv2.destroyAllWindows()
