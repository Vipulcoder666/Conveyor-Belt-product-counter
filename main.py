import cv2

from utils.detector import detect_objects
from utils.counter import get_center

VIDEO_PATH = "videos/conveyor.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

count = 0
counted = set()

line_y = 300

while True:

    ret, frame = cap.read()

    if not ret:
        break

    contours, mask = detect_objects(frame)

    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (255, 0, 0),
        2
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cx, cy = get_center(x, y, w, h)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        object_id = f"{cx}_{cy}"

        # if cy > line_y and object_id not in counted:
        #     count += 1
        #     counted.add(object_id)

    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()