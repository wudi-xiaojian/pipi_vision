import cv2
import mediapipe as mp


def draw_face_landmarks(frame, face_landmarks):

    height, width, _ = frame.shape

    for landmark in face_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    return frame


def draw_pose_landmarks(frame, pose_landmarks):

    height, width, _ = frame.shape

    # =========================
    # 绘制人体关键点
    # =========================

    for landmark in pose_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 0, 0),
            -1
        )

    # =========================
    # MediaPipe Pose 骨架连接
    # =========================

    connections = (
        mp.solutions.pose.POSE_CONNECTIONS
    )

    for connection in connections:

        start_index = connection[0]
        end_index = connection[1]

        start = pose_landmarks[start_index]
        end = pose_landmarks[end_index]

        start_x = int(start.x * width)
        start_y = int(start.y * height)

        end_x = int(end.x * width)
        end_y = int(end.y * height)

        cv2.line(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            (255, 0, 0),
            2
        )

    return frame