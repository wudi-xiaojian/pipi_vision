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
    print("正在进行人脸 + 表情 + Pose + 动作检测...")
    print("当前动作：举手检测")
    print("按 q 键退出")


    # =========================
    # 4. 时间
    # =========================

    start_time = time.time()


    # =========================
    # 5. 主循环
    # =========================

    while True:

        # =========================
        # 默认状态
        # =========================

        emotion = "NO FACE"
        action = "NO ACTION"

        face_count = 0
        pose_count = 0


        # =========================
        # 读取摄像头
        # =========================

        ret, frame = cap.read()

        if not ret:

            print("无法读取摄像头画面")

            break


        # =========================
        # BGR → RGB
        # =========================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # =========================
        # 时间戳
        # =========================

        timestamp_ms = int(
            (time.time() - start_time) * 1000
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

            # -------------------------
            # 绘制人脸关键点
            # -------------------------

            for face_landmarks in face_result.face_landmarks:

                frame = draw_face_landmarks(
                    frame,
                    face_landmarks
                )


            # -------------------------
            # BlendShapes
            # -------------------------

            if face_result.face_blendshapes:

                blendshapes = (
                    face_result.face_blendshapes[0]
                )


                # -------------------------
                # 表情识别
                # -------------------------

                emotion = emotion_detector.detect(
                    blendshapes
                )


                # -------------------------
                # 显示 BlendShapes
                # -------------------------

                display_blendshapes = (
                    blendshapes[:20]
                )


                for i, category in enumerate(
                    display_blendshapes
                ):

                    name = category.category_name
                    score = category.score

                    text = (
                        f"{name}: {score:.2f}"
                    )


                    if i < 10:

                        x = 30
                        y = 80 + i * 30

                    else:

                        x = 350
                        y = 80 + (i - 10) * 30


                    cv2.putText(
                        frame,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
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


            # -------------------------
            # 绘制人体骨架
            # -------------------------

            frame = draw_pose_landmarks(
                frame,
                pose_landmarks
            )


            # -------------------------
            # 动作识别
            # -------------------------

            action = action_detector.detect(
                pose_landmarks
            )


        # ==================================================
        # C. 显示状态
        # ==================================================

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Poses: {pose_count}",
            (30, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )


        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (30, 510),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Action: {action}",
            (30, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2
        )


        # ==================================================
        # D. 显示摄像头
        # ==================================================

        cv2.imshow(
            "PiPi Vision - Face + Pose + Action",
            frame
        )


        # =========================
        # 按 q 退出
        # =========================

        if cv2.waitKey(1) & 0xFF == ord("q"):

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