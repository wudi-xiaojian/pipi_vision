class ActionDetector:

    def __init__(self):
        self.current_action = "NO ACTION"

    def detect(self, pose_landmarks):

        if not pose_landmarks:
            self.current_action = "NO ACTION"
            return self.current_action

        # MediaPipe Pose landmarks
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        # 获取关键点
        left_shoulder = pose_landmarks[LEFT_SHOULDER]
        right_shoulder = pose_landmarks[RIGHT_SHOULDER]

        left_wrist = pose_landmarks[LEFT_WRIST]
        right_wrist = pose_landmarks[RIGHT_WRIST]

        # 左手是否举过左肩
        left_hand_up = (
            left_wrist.y < left_shoulder.y
        )

        # 右手是否举过右肩
        right_hand_up = (
            right_wrist.y < right_shoulder.y
        )

        # =========================
        # 判断动作
        # =========================

        if left_hand_up and right_hand_up:

            action = "BOTH_HANDS_UP"

        elif left_hand_up:

            action = "LEFT_HAND_UP"

        elif right_hand_up:

            action = "RIGHT_HAND_UP"

        else:

            action = "NO ACTION"

        self.current_action = action

        return self.current_action