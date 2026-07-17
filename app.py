import cv2

from config import *

from core.hand_tracker import HandTracker


def main():

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker(
        max_hands=MAX_HANDS,
        detection_confidence=MIN_DETECTION_CONFIDENCE,
        tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        # Detect hands
        frame, hands = tracker.process(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    tracker.close()

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()