import cv2
import time

from utils.drawing import (
    draw_face_landmarks,
    draw_pose_landmarks
)

from vision.face_detector import FaceDetector
from vision.emotion_detector import EmotionDetector
from vision.pose_detector import PoseDetector
from vision.action_detector import ActionDetector
from vision.behavior_state import BehaviorState


def main():

    # =========================
    # 1. 模型路径
    # =========================

    face_model_path = "models/face_landmarker.task"
    pose_model_path = "models/pose_landmarker_full.task"


    # =========================
    # 2. 创建检测器
    # =========================

    face_detector = FaceDetector(
        face_model_path
    )

    emotion_detector = EmotionDetector(
        history_size=10
    )

    pose_detector = PoseDetector(
        pose_model_path
    )

    action_detector = ActionDetector()

    behavior_state = BehaviorState()


    # =========================
    # 3. 打开摄像头
    # =========================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("无法打开摄像头")

        face_detector.close()
        pose_detector.close()

        return


    print("摄像头启动成功！")

    print(
        "正在进行："
        "人脸 + 表情 + 上半身动作检测..."
    )

    print(
        "表情："
        "HAPPINESS / SURPRISE / ANGER / "
        "FEAR / DISGUST / SADNESS / NEUTRAL"
    )

    print(
        "上半身动作："
        "NO ACTION / LEFT_HAND_UP / "
        "RIGHT_HAND_UP / BOTH_HANDS_UP / "
        "WAVE / UNKNOWN_ACTION"
    )

    print("按 q 键退出")


    # =========================
    # 4. 时间
    # =========================

    start_time = time.time()


    # =========================
    # 5. 主循环
    # =========================

    while True:

        emotion = "NO FACE"

        upper_body_action = "NO ACTION"

        face_count = 0
        pose_count = 0


        # =========================
        # 读取摄像头
        # =========================

        ret, frame = cap.read()

        if not ret:

            print("无法读取摄像头画面")

            break


        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        timestamp_ms = int(
            (time.time() - start_time)
            * 1000
        )


        # ==================================================
        # A. 人脸检测
        # ==================================================

        face_result = face_detector.detect(
            frame_rgb,
            timestamp_ms
        )


        face_count = len(
            face_result.face_landmarks
        )


        if face_count > 0:

            for face_landmarks in (
                face_result.face_landmarks
            ):

                frame = draw_face_landmarks(
                    frame,
                    face_landmarks
                )


            if face_result.face_blendshapes:

                blendshapes = (
                    face_result.face_blendshapes[0]
                )


                emotion = emotion_detector.detect(
                    blendshapes
                )


                # =========================
                # 显示 BlendShapes
                # =========================

                display_blendshapes = (
                    blendshapes[:20]
                )


                for i, category in enumerate(
                    display_blendshapes
                ):

                    name = (
                        category.category_name
                    )

                    score = category.score


                    text = (
                        f"{name}: {score:.2f}"
                    )


                    if i < 10:

                        x = 30
                        y = 80 + i * 30

                    else:

                        x = 350
                        y = 80 + (
                            i - 10
                        ) * 30


                    cv2.putText(
                        frame,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        3
                    )


        # ==================================================
        # B. Pose 人体检测
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


            frame = draw_pose_landmarks(
                frame,
                pose_landmarks
            )


            upper_body_action = (
                action_detector.detect(
                    pose_landmarks
                )
            )


        # ==================================================
        # C. 更新 BehaviorState
        # ==================================================

        behavior_state.update(
            emotion,
            upper_body_action
        )


        state = behavior_state.get_state()

        description = (
            behavior_state.get_description()
        )


        # ==================================================
        # D. 显示基础状态
        # ==================================================

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )


        cv2.putText(
            frame,
            f"Poses: {pose_count}",
            (30, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 200, 0),
            3
        )


        cv2.putText(
            frame,
            f"Emotion: {state['emotion']}",
            (30, 510),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


        # ==================================================
        # E. 上半身动作
        # ==================================================

        action_color = (
            (0, 0, 255)
            if state["action"] == "UNKNOWN_ACTION"
            else (0, 165, 255)
        )


        cv2.putText(
            frame,
            f"Upper Body: {state['action']}",
            (30, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            action_color,
            3
        )


        # ==================================================
        # F. 行为描述
        # ==================================================

        description_color = (
            (0, 0, 255)
            if state["action"] == "UNKNOWN_ACTION"
            else (0, 255, 0)
        )


        cv2.putText(
            frame,
            description,
            (30, 590),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            description_color,
            3
        )


        # ==================================================
        # G. 显示窗口
        # ==================================================

        cv2.imshow(
            "PiPi Vision - "
            "Face + Upper Body",
            frame
        )


        # ==================================================
        # H. 退出
        # ==================================================

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


    # =========================
    # 6. 释放资源
    # =========================

    cap.release()

    cv2.destroyAllWindows()

    face_detector.close()

    pose_detector.close()


if __name__ == "__main__":

    main()