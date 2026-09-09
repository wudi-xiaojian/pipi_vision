import cv2


# ==========================================================
# Hand Landmarker 21 点连接关系
# ==========================================================

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


# ==========================================================
# Face
# ==========================================================

def draw_face_landmarks(
    frame,
    face_landmarks,
):
    """
    绘制 Face Landmarker 关键点。
    """

    height, width = frame.shape[:2]

    for landmark in face_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    return frame


# ==========================================================
# Pose
# ==========================================================

def draw_pose_landmarks(
    frame,
    pose_landmarks,
):
    """
    绘制 Pose Landmarker 关键点和骨架。
    """

    height, width = frame.shape[:2]

    # MediaPipe Pose 常用骨架连接
    pose_connections = [
        # Face / head
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),

        # Shoulders
        (11, 12),

        # Left arm
        (11, 13),
        (13, 15),

        # Right arm
        (12, 14),
        (14, 16),

        # Torso
        (11, 23),
        (12, 24),
        (23, 24),

        # Left leg
        (23, 25),
        (25, 27),

        # Right leg
        (24, 26),
        (26, 28),
    ]

    # ------------------------------------------------------
    # 绘制骨架
    # ------------------------------------------------------

    for start_idx, end_idx in pose_connections:

        if (
            start_idx >= len(pose_landmarks)
            or end_idx >= len(pose_landmarks)
        ):
            continue

        start = pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]

        x1 = int(
            start.x * width
        )

        y1 = int(
            start.y * height
        )

        x2 = int(
            end.x * width
        )

        y2 = int(
            end.y * height
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------
    # 绘制关键点
    # ------------------------------------------------------

    for landmark in pose_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    return frame


# ==========================================================
# Hand
# ==========================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks,
    point_radius=4,
    line_thickness=2,
):
    """
    绘制 MediaPipe Hand Landmarker 的
    21 个关键点和骨架。

    Hand Landmarks:

        0  Wrist

        Thumb:
        1  CMC
        2  MCP
        3  IP
        4  Tip

        Index:
        5  MCP
        6  PIP
        7  DIP
        8  Tip

        Middle:
        9  MCP
        10 PIP
        11 DIP
        12 Tip

        Ring:
        13 MCP
        14 PIP
        15 DIP
        16 Tip

        Pinky:
        17 MCP
        18 PIP
        19 DIP
        20 Tip
    """

    if hand_landmarks is None:
        return frame

    if len(hand_landmarks) < 21:
        return frame

    height, width = frame.shape[:2]

    # ======================================================
    # 1. 绘制骨架
    # ======================================================

    for start_idx, end_idx in HAND_CONNECTIONS:

        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]

        x1 = int(
            start.x * width
        )

        y1 = int(
            start.y * height
        )

        x2 = int(
            end.x * width
        )

        y2 = int(
            end.y * height
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            line_thickness,
            cv2.LINE_AA,
        )

    # ======================================================
    # 2. 绘制 21 个关键点
    # ======================================================

    for index, landmark in enumerate(
        hand_landmarks
    ):

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            point_radius,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    return frame