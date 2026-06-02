import cv2

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=50,
    detectShadows=False
)

def detect_objects(frame):
    mask = bg_subtractor.apply(frame)

    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, mask