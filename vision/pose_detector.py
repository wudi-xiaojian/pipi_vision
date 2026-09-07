import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:

    def __init__(self, model_path):

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )

        self.detector = vision.PoseLandmarker.create_from_options(
            options
        )

    def detect(self, frame, timestamp_ms):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    def close(self):
        self.detector.close()