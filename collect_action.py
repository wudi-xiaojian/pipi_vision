import cv2
import os
import time
import numpy as np

from vision.pose_detector import PoseDetector
from utils.drawing import draw_pose_landmarks


# ==================================================
# 配置
# ==================================================

POSE_MODEL_PATH = "models/pose_landmarker_full.task"

DATA_ROOT = "data/actions"

SEQUENCE_LENGTH = 30


# ==================================================
# 创建数据目录
# ==================================================

def create_data_directory(action_name):

    action_dir = os.path.join(
        DATA_ROOT,
        action_name
    )

    os.makedirs(
        action_dir,
        exist_ok=True
    )

    return action_dir


# ==================================================
# 获取下一个样本编号
# ==================================================

def get_next_sample_number(action_dir):

    files = os.listdir(action_dir)

    numbers = []

    for filename in files:

        if not filename.endswith(".npy"):
            continue

        name = filename.replace(
            ".npy",
            ""
        )

        if not name.startswith("sample_"):
            continue

        try:

            number = int(
                name.replace(
                    "sample_",
                    ""
                )
            )

            numbers.append(number)

        except ValueError:

            continue


    if not numbers:

        return 1


    return max(numbers) + 1


# ==================================================
# 提取上半身关键点
# ==================================================

def extract_upper_body_landmarks(
    pose_landmarks
):

    # MediaPipe Pose
    #
    # 11  Left Shoulder
    # 12  Right Shoulder
    # 13  Left Elbow
    # 14  Right Elbow
    # 15  Left Wrist
    # 16  Right Wrist
    # 23  Left Hip
    # 24  Right Hip

    indexes = [
        11,
        12,
        13,
        14,
        15,
        16,
        23,
        24
    ]


    data = []


    for index in indexes:

        landmark = pose_landmarks[index]


        data.append([
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ])


    return np.array(
        data,
        dtype=np.float32
    )


# ==================================================
# 采集一个动作
# ==================================================

def collect_sequence(
    pose_detector,
    action_name,
    sample_number
):

    action_dir = create_data_directory(
        action_name
    )


    sequence = []


    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print("无法打开摄像头")

        return False


    print()
    print("=" * 50)
    print(
        f"准备采集动作：{action_name}"
    )
    print(
        f"样本编号：{sample_number}"
    )
    print("=" * 50)

    print()
    print(
        "请准备好动作..."
    )

    print(
        "按 SPACE 开始采集"
    )

    print(
        "按 Q 取消"
    )


    started = False


    while True:

        ret, frame = cap.read()


        if not ret:

            print(
                "无法读取摄像头画面"
            )

            cap.release()

            cv2.destroyAllWindows()

            return False


        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        timestamp_ms = int(
            time.time() * 1000
        )


        pose_result = pose_detector.detect(
            frame_rgb,
            timestamp_ms
        )


        pose_count = len(
            pose_result.pose_landmarks
        )


        # ==================================================
        # 显示 Pose
        # ==================================================

        if pose_count > 0:

            pose_landmarks = (
                pose_result.pose_landmarks[0]
            )


            frame = draw_pose_landmarks(
                frame,
                pose_landmarks
            )


            # ==================================================
            # 开始采集
            # ==================================================

            if started:

                landmark_data = (
                    extract_upper_body_landmarks(
                        pose_landmarks
                    )
                )


                sequence.append(
                    landmark_data
                )


        # ==================================================
        # 状态显示
        # ==================================================

        cv2.putText(
            frame,
            f"Action: {action_name}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )


        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            3
        )


        if not started:

            cv2.putText(
                frame,
                "Press SPACE to start",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3
            )

        else:

            cv2.putText(
                frame,
                "Recording...",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )


        cv2.imshow(
            "PiPi Vision - "
            "Action Data Collection",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ==================================================
        # SPACE：开始采集
        # ==================================================

        if key == ord(" "):

            if not started:

                started = True

                sequence = []

                print()
                print(
                    "开始采集..."
                )


        # ==================================================
        # Q：退出
        # ==================================================

        if key == ord("q"):

            print(
                "取消当前采集"
            )

            cap.release()

            cv2.destroyAllWindows()

            return False


        # ==================================================
        # 数据采集完成
        # ==================================================

        if len(sequence) >= SEQUENCE_LENGTH:

            break


    cap.release()

    cv2.destroyAllWindows()


    # ==================================================
    # 保存数据
    # ==================================================

    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    filename = os.path.join(
        action_dir,
        f"sample_{sample_number:03d}.npy"
    )


    np.save(
        filename,
        sequence
    )


    print()
    print("=" * 50)
    print("采集完成！")
    print("=" * 50)

    print(
        f"动作：{action_name}"
    )

    print(
        f"样本：{sample_number}"
    )

    print(
        f"数据形状：{sequence.shape}"
    )

    print(
        f"保存位置：{filename}"
    )

    print("=" * 50)


    return True


# ==================================================
# 主程序
# ==================================================

def main():

    print()
    print("=" * 60)
    print("PiPi Vision - Action Data Collector")
    print("=" * 60)

    print()
    print("可采集的动作：")
    print()
    print("1. WAVE")
    print("2. CLAP")
    print("3. POINT")
    print("4. BOTH_HANDS_UP")
    print("5. LEFT_HAND_UP")
    print("6. RIGHT_HAND_UP")
    print()


    action_name = input(
        "请输入要采集的动作名称："
    ).strip().upper()


    if not action_name:

        print(
            "动作名称不能为空"
        )

        return


    print()
    print(
        f"当前动作：{action_name}"
    )


    # ==================================================
    # 初始化 Pose Detector
    # ==================================================

    pose_detector = PoseDetector(
        POSE_MODEL_PATH
    )


    try:

        # ==================================================
        # 找到下一个样本编号
        # ==================================================

        action_dir = create_data_directory(
            action_name
        )


        sample_number = (
            get_next_sample_number(
                action_dir
            )
        )


        print()
        print(
            f"下一个样本编号："
            f"{sample_number}"
        )


        # ==================================================
        # 开始采集
        # ==================================================

        collect_sequence(
            pose_detector,
            action_name,
            sample_number
        )


    finally:

        pose_detector.close()


# ==================================================
# 程序入口
# ==================================================

if __name__ == "__main__":

    main()