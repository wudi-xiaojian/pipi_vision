import cv2
import time

from utils.drawing import (
    draw_face_landmarks,
    draw_pose_landmarks,
    draw_hand_landmarks,
)

from vision.face_detector import FaceDetector
from vision.emotion_detector import EmotionDetector
from vision.pose_detector import PoseDetector
from vision.action_detector import ActionDetector
from vision.hand_detector import HandDetector
from vision.gesture_detector import GestureDetector
from vision.high_five_detector import HighFiveDetector
from vision.behavior_state import BehaviorState


# ==========================================================
# Model paths
# ==========================================================

FACE_MODEL_PATH = "models/face_landmarker.task"
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"


def main():

    # ======================================================
    # 1. 创建检测器
    # ======================================================

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

    hand_detector = HandDetector(
        HAND_MODEL_PATH,
        num_hands=2
    )

    gesture_detector = GestureDetector()

    high_five_detector = HighFiveDetector()

    behavior_state = BehaviorState()


    # ======================================================
    # 2. 打开摄像头
    # ======================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Cannot open camera."
        )

        face_detector.close()
        pose_detector.close()
        hand_detector.close()

        return


    print("PiPi Vision")

    print(
        "表情："
        "HAPPINESS / SURPRISE / ANGER / "
        "SADNESS / NEUTRAL"
    )

    print(
        "上半身动作："
        "NO ACTION / HAND_UP / WAVE / "
        "HEAD_DOWN / DAYDREAMING / NOT_IN_SEAT"
    )

    print(
        "手势："
        "OPEN_PALM / THUMBS_UP / NO GESTURE"
    )

    print(
        "虚拟击掌："
        "NO ACTION / HAND_APPROACHING / HIGH_FIVE"
    )

    print("按 Q 退出")


    # ======================================================
    # 3. 时间
    # ======================================================

    start_time = time.time()


    # ======================================================
    # 4. 主循环
    # ======================================================

    while True:

        # ==================================================
        # 读取摄像头
        # ==================================================

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read frame."
            )

            break


        # ==================================================
        # 镜像
        # ==================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ==================================================
        # BGR → RGB
        # ==================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ==================================================
        # Timestamp
        # ==================================================

        timestamp_ms = int(
            (time.time() - start_time)
            * 1000
        )


        # ==================================================
        # 默认状态
        # ==================================================

        emotion = "NO FACE"

        upper_body_action = "NOT_IN_SEAT"

        hand_gesture = "NO GESTURE"

        left_hand_gesture = "NO GESTURE"

        right_hand_gesture = "NO GESTURE"

        high_five_action = "NO ACTION"

        face_count = 0

        pose_count = 0

        hand_count = 0

        facial_transformation_matrix = None


        # ==================================================
        # 5. FACE
        # ==================================================

        face_result = face_detector.detect(
            frame_rgb,
            timestamp_ms
        )


        face_count = len(
            face_result.face_landmarks
        )


        if face_count > 0:

            # --------------------------------------------------
            # Face Landmarks
            # --------------------------------------------------

            face_landmarks = (
                face_result.face_landmarks[0]
            )


            frame = draw_face_landmarks(
                frame,
                face_landmarks
            )


            # --------------------------------------------------
            # Emotion
            # --------------------------------------------------

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


            # --------------------------------------------------
            # Facial Transformation Matrix
            # --------------------------------------------------

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


        # ==================================================
        # 6. POSE
        # ==================================================

        pose_result = pose_detector.detect(
            frame_rgb,
            timestamp_ms
        )


        pose_count = len(
            pose_result.pose_landmarks
        )


        if pose_count > 0:

            pose_landmarks = (
                pose_result.pose_landmarks[0]
            )


            # --------------------------------------------------
            # Pose Landmarks
            # --------------------------------------------------

            frame = draw_pose_landmarks(
                frame,
                pose_landmarks
            )


            # --------------------------------------------------
            # Action Detector
            # --------------------------------------------------

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


        # ==================================================
        # 7. HAND
        # ==================================================

        hand_result = hand_detector.detect(
            frame_rgb,
            timestamp_ms
        )


        hand_count = len(
            hand_result.hand_landmarks
        )


        # ==================================================
        # 8. 左右手分别识别
        # ==================================================

        for index, hand_landmarks in enumerate(
            hand_result.hand_landmarks
        ):

            # --------------------------------------------------
            # 绘制手部 21 点 + 骨架
            # --------------------------------------------------

            frame = draw_hand_landmarks(
                frame,
                hand_landmarks
            )


            # --------------------------------------------------
            # Gesture
            # --------------------------------------------------

            gesture = (
                gesture_detector.detect(
                    hand_landmarks
                )
            )


            # --------------------------------------------------
            # 获取左右手信息
            # --------------------------------------------------

            handedness = "Unknown"


            if (
                hand_result.handedness
                and index
                < len(
                    hand_result.handedness
                )
            ):

                handedness_info = (
                    hand_result.handedness[index]
                )


                if (
                    handedness_info
                    and len(
                        handedness_info
                    ) > 0
                ):

                    handedness = (
                        handedness_info[0]
                        .category_name
                    )


            # --------------------------------------------------
            # Left Hand
            # --------------------------------------------------

            if handedness == "Left":

                left_hand_gesture = gesture


            # --------------------------------------------------
            # Right Hand
            # --------------------------------------------------

            elif handedness == "Right":

                right_hand_gesture = gesture


        # ==================================================
        # 9. 选择当前主要手势
        #
        # BehaviorState 暂时仍使用：
        #
        #     hand_gesture
        #
        # 下一阶段再升级成双手输入。
        # ==================================================

        if (
            left_hand_gesture
            != "NO GESTURE"
        ):

            hand_gesture = (
                left_hand_gesture
            )

        elif (
            right_hand_gesture
            != "NO GESTURE"
        ):

            hand_gesture = (
                right_hand_gesture
            )

        else:

            hand_gesture = (
                "NO GESTURE"
            )


        # ==================================================
        # 10. HIGH FIVE
        #
        # 只使用一只实际检测到的手。
        #
        # 用户：
        #
        #     🖐 → → → 📷
        #
        #     ↓
        #
        #     HIGH_FIVE
        # ==================================================

        high_five_action = "NO ACTION"


        # --------------------------------------------------
        # 优先使用左手
        # --------------------------------------------------

        if (
            left_hand_gesture
            != "NO GESTURE"
        ):

            for index, hand_landmarks in enumerate(
                hand_result.hand_landmarks
            ):

                handedness = "Unknown"

                if (
                    hand_result.handedness
                    and index
                    < len(
                        hand_result.handedness
                    )
                ):

                    handedness_info = (
                        hand_result.handedness[index]
                    )

                    if (
                        handedness_info
                        and len(
                            handedness_info
                        ) > 0
                    ):

                        handedness = (
                            handedness_info[0]
                            .category_name
                        )


                if handedness == "Left":

                    high_five_action = (
                        high_five_detector.detect(
                            hand_landmarks,
                            left_hand_gesture
                        )
                    )

                    break


        # --------------------------------------------------
        # 如果没有左手，则使用右手
        # --------------------------------------------------

        elif (
            right_hand_gesture
            != "NO GESTURE"
        ):

            for index, hand_landmarks in enumerate(
                hand_result.hand_landmarks
            ):

                handedness = "Unknown"

                if (
                    hand_result.handedness
                    and index
                    < len(
                        hand_result.handedness
                    )
                ):

                    handedness_info = (
                        hand_result.handedness[index]
                    )

                    if (
                        handedness_info
                        and len(
                            handedness_info
                        ) > 0
                    ):

                        handedness = (
                            handedness_info[0]
                            .category_name
                        )


                if handedness == "Right":

                    high_five_action = (
                        high_five_detector.detect(
                            hand_landmarks,
                            right_hand_gesture
                        )
                    )

                    break


        # ==================================================
        # 11. BehaviorState
        # ==================================================

        behavior_state.update(
            emotion,
            upper_body_action,
            hand_gesture
        )


        state = (
            behavior_state.get_state()
        )


        description = (
            behavior_state.get_description()
        )


        # ==================================================
        # 12. 简洁状态面板
        # ==================================================
        # 只显示核心判定结果，不显示 MediaPipe BlendShapes
        # 和 High Five 的内部调试参数。

        text_x = 15
        text_y = 45
        text_gap = 42
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.78
        thickness = 3

        # Emotion
        cv2.putText(
            frame, f"Emotion: {state['emotion']}",
            (text_x, text_y), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Action
        cv2.putText(
            frame, f"Action: {state['pose_action']}",
            (text_x, text_y + text_gap), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Gesture
        cv2.putText(
            frame, f"Gesture: {state['hand_gesture']}",
            (text_x, text_y + text_gap * 2), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Interaction
        cv2.putText(
            frame, f"Interaction: {high_five_action}",
            (text_x, text_y + text_gap * 3), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Behavior
        cv2.putText(
            frame, f"Behavior: {state['final_behavior']}",
            (text_x, text_y + text_gap * 4), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Bottom description
        description_y = frame.shape[0] - 30
        cv2.putText(
            frame, description, (15, description_y),
            font, 0.72, (255, 255, 255), 3, cv2.LINE_AA
        )

        # Hands count
        cv2.putText(
            frame, f"Hands: {hand_count}",
            (frame.shape[1] - 150, frame.shape[0] - 30),
            font, 0.58, (255, 255, 255), 2, cv2.LINE_AA
        )

        # ==================================================
        # 13. 显示
        # ==================================================

        cv2.imshow(
            "PiPi Vision",
            frame
        )


        # ==================================================
        # 14. Q 退出
        # ==================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):

            break


    # ======================================================
    # 15. 释放资源
    # ======================================================

    cap.release()

    cv2.destroyAllWindows()

    face_detector.close()

    pose_detector.close()

    hand_detector.close()


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    main()