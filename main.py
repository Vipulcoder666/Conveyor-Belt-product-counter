import cv2

from utils.detector import detect_objects
from utils.counter import get_center

VIDEO_PATH = "videos/conveyor.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

count = 0
counted = False
frame_number = 0

# Put line near rollers
line_y = 300

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    contours, mask = detect_objects(frame)

    # Allow background subtractor to learn
    if frame_number < 50:

        cv2.putText(
            frame,
            "Learning Background...",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(20) & 0xFF == 27:
            break

        continue

    # Blue counting line
    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (255, 0, 0),
        2
    )

    object_detected = False

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noise
        if area < 2000:
            continue

        object_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cx, cy = get_center(x, y, w, h)

        # Green box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Red center point
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        # Show CY on frame
        cv2.putText(
            frame,
            f"CY:{cy}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        print("CY =", cy)

        # Counting logic
        if cy > line_y and not counted:
            count += 1
            counted = True

            print("COUNT =", count)

    # Reset when object disappears
    if not object_detected:
        counted = False

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