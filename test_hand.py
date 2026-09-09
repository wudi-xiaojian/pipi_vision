import cv2
import time

from vision.hand_detector import HandDetector


# MediaPipe Hand Landmarker 的 21 个关键点连接关系
HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
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
    """
    在画面上绘制一只手的 21 个关键点和骨架。
    """

    height, width = frame.shape[:2]

    points = []

    # 1. 把归一化坐标转换成像素坐标
    for index, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        # 画关键点
        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 255, 0),
            -1
        )

        # 显示关键点编号
        cv2.putText(
            frame,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # 2. 绘制手部骨架
    for start_index, end_index in HAND_CONNECTIONS:
        start = points[start_index]
        end = points[end_index]

        cv2.line(
            frame,
            start,
            end,
            (255, 0, 0),
            2
        )


def main():
    hand_detector = HandDetector(
        model_path="models/hand_landmarker.task",
        num_hands=2,
    )

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

        # Hand Landmarker
        result = hand_detector.detect(
            frame,
            timestamp_ms
        )

        hand_count = len(result.hand_landmarks)

        # 绘制每只手
        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            draw_hand_landmarks(
                frame,
                hand_landmarks
            )

            # 获取左右手信息
            handedness = "UNKNOWN"

            if (
                result.handedness
                and hand_index < len(result.handedness)
                and len(result.handedness[hand_index]) > 0
            ):
                handedness = (
                    result.handedness[hand_index][0].category_name
                )

            # 找到手掌中心附近的位置
            wrist = hand_landmarks[0]

            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])

            cv2.putText(
                frame,
                handedness,
                (wrist_x, wrist_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # 左上角显示手数量
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
            "Hand Landmarker Test",
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