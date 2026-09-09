import cv2
import time

from vision.face_detector import FaceDetector
from vision.emotion_detector import EmotionDetector
from vision.pose_detector import PoseDetector
from vision.action_detector import ActionDetector
from vision.behavior_state import BehaviorState

from utils.drawing import (
    draw_face_landmarks,
    draw_pose_landmarks
)


# ============================================================
# Model paths
# ============================================================

FACE_MODEL_PATH = (
    "models/face_landmarker.task"
)

POSE_MODEL_PATH = (
    "models/pose_landmarker_full.task"
)


# ============================================================
# Initialize detectors
# ============================================================

face_detector = FaceDetector(
    FACE_MODEL_PATH
)

emotion_detector = EmotionDetector()

pose_detector = PoseDetector(
    POSE_MODEL_PATH
)

action_detector = ActionDetector(
    daydreaming_seconds=5.0
)

behavior_state = BehaviorState()


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Cannot open camera.")

    raise SystemExit


# ============================================================
# Start information
# ============================================================

print()
print("=" * 60)
print("PiPi Vision")
print("=" * 60)

print()
print(
    "表情："
    "HAPPINESS / "
    "SURPRISE / "
    "ANGER / "
    "SADNESS / "
    "NEUTRAL"
)

print(
    "上半身动作："
    "NO ACTION / "
    "HAND_UP / "
    "WAVE / "
    "HEAD_DOWN / "
    "DAYDREAMING / "
    "NOT_IN_SEAT"
)

print()
print("按 Q 退出")
print("=" * 60)
print()


# ============================================================
# Main loop
# ============================================================

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Failed to read frame."
        )

        break

    # --------------------------------------------------------
    # Mirror image
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(
        (
            time.time()
            - start_time
        )
        * 1000
    )

    # ========================================================
    # Default states
    # ========================================================

    emotion = "NO FACE"

    upper_body_action = (
        "NOT_IN_SEAT"
    )

    face_count = 0
    pose_count = 0

    facial_transformation_matrix = (
        None
    )

    # ========================================================
    # FACE
    # ========================================================

    face_result = face_detector.detect(
        frame_rgb,
        timestamp_ms
    )

    face_count = len(
        face_result.face_landmarks
    )

    if face_count > 0:

        # ----------------------------------------------------
        # Face landmarks
        # ----------------------------------------------------

        face_landmarks = (
            face_result.face_landmarks[0]
        )

        frame = draw_face_landmarks(
            frame,
            face_landmarks
        )

        # ----------------------------------------------------
        # BlendShapes
        # ----------------------------------------------------

        if (
            face_result.face_blendshapes
            and len(
                face_result.face_blendshapes
            ) > 0
        ):

            blendshapes = (
                face_result.face_blendshapes[0]
            )

            emotion = (
                emotion_detector.detect(
                    blendshapes
                )
            )

            # ------------------------------------------------
            # BlendShapes
            #
            # 放在右侧，避免和左侧状态信息重叠
            # ------------------------------------------------

            blendshape_x = (
                frame.shape[1] - 300
            )

            blendshape_y = 40

            for i, category in enumerate(
                blendshapes[:20]
            ):

                name = (
                    category.category_name
                )

                score = category.score

                text = (
                    f"{name}: "
                    f"{score:.2f}"
                )

                cv2.putText(
                    frame,
                    text,
                    (
                        blendshape_x,
                        blendshape_y
                        + i * 28
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Facial transformation matrix
        # ----------------------------------------------------

        if (
            hasattr(
                face_result,
                "facial_transformation_matrixes"
            )
            and face_result.facial_transformation_matrixes
        ):

            facial_transformation_matrix = (
                face_result
                .facial_transformation_matrixes[0]
            )

    # ========================================================
    # POSE
    # ========================================================

    pose_result = pose_detector.detect(
        frame_rgb,
        timestamp_ms
    )

    pose_count = len(
        pose_result.pose_landmarks
    )

    if pose_count > 0:

        # ----------------------------------------------------
        # Pose landmarks
        # ----------------------------------------------------

        pose_landmarks = (
            pose_result.pose_landmarks[0]
        )

        frame = draw_pose_landmarks(
            frame,
            pose_landmarks
        )

        # ----------------------------------------------------
        # Action detection
        # ----------------------------------------------------

        upper_body_action = (
            action_detector.detect(
                pose_landmarks,
                facial_transformation_matrix
            )
        )

    else:

        upper_body_action = (
            "NOT_IN_SEAT"
        )

        action_detector.reset()

    # ========================================================
    # Behavior state
    # ========================================================

    behavior_state.update(
        emotion,
        upper_body_action
    )

    description = (
        behavior_state.get_description()
    )

    # ========================================================
    # UI
    #
    # 左侧：状态
    # 右侧：BlendShapes
    # 底部：Description
    #
    # 三个区域完全分开
    # ========================================================

    # --------------------------------------------------------
    # Left panel
    # --------------------------------------------------------

    text_x = 10

    text_y = 40

    text_gap = 45

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.85

    thickness = 3

    # --------------------------------------------------------
    # Faces
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (
            text_x,
            text_y
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Poses
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Poses: {pose_count}",
        (
            text_x,
            text_y
            + text_gap
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Emotion
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (
            text_x,
            text_y
            + text_gap * 2
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Upper Body
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Upper Body: {upper_body_action}",
        (
            text_x,
            text_y
            + text_gap * 3
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # ========================================================
    # Description
    # ========================================================

    description_y = (
        frame.shape[0] - 30
    )

    cv2.putText(
        frame,
        description,
        (
            10,
            description_y
        ),
        font,
        0.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    # ========================================================
    # Show
    # ========================================================

    cv2.imshow(
        "PiPi Vision",
        frame
    )

    # ========================================================
    # Quit
    # ========================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    if key == ord("q"):

        break


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()