import cv2
import time

from vision.hand_detector import HandDetector
from vision.gesture_detector import GestureDetector


HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17),
]


def draw_hand_landmarks(frame, hand_landmarks):
    height, width = frame.shape[:2]

    points = []

    for index, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (255, 0, 0),
            2
        )


def main():
    hand_detector = HandDetector(
        model_path="models/hand_landmarker.task",
        num_hands=2,
    )

    gesture_detector = GestureDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = hand_detector.detect(
            frame,
            timestamp_ms
        )

        hand_count = len(
            result.hand_landmarks
        )

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            draw_hand_landmarks(
                frame,
                hand_landmarks
            )

            gesture = gesture_detector.detect(
                hand_landmarks
            )

            handedness = "UNKNOWN"

            if (
                result.handedness
                and hand_index < len(result.handedness)
                and len(result.handedness[hand_index]) > 0
            ):
                handedness = (
                    result.handedness[hand_index][0].category_name
                )

            wrist = hand_landmarks[0]

            wrist_x = int(
                wrist.x * frame.shape[1]
            )

            wrist_y = int(
                wrist.y * frame.shape[0]
            )

            cv2.putText(
                frame,
                handedness,
                (wrist_x, wrist_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                gesture,
                (wrist_x, wrist_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Hands: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Hand Gesture Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()


if __name__ == "__main__":
    main()