import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        num_hands=2,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, frame, timestamp_ms):
        """
        frame:
            OpenCV BGR image

        timestamp_ms:
            当前视频帧时间戳

        return:
            MediaPipe HandLandmarkerResult
        """

        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame[:, :, ::-1].copy()
        )
        
        result = self.landmarker.detect_for_video(
            rgb_frame,
            timestamp_ms
        )

        return result

    def close(self):
        self.landmarker.close()