from collections import deque


class ActionDetector:

    def __init__(self):

        # =========================
        # 已识别动作的历史
        # =========================

        self.action_history = deque(
            maxlen=10
        )

        # =========================
        # WAVE 历史
        # =========================

        self.left_wave_history = deque(
            maxlen=24
        )

        self.right_wave_history = deque(
            maxlen=24
        )

        # =========================
        # 手腕运动历史
        # =========================

        self.left_wrist_history = deque(
            maxlen=12
        )

        self.right_wrist_history = deque(
            maxlen=12
        )


    # ==================================================
    # 判断手是否明显移动
    # ==================================================

    def detect_movement(
        self,
        wrist,
        history
    ):

        history.append(
            (
                wrist.x,
                wrist.y
            )
        )

        if len(history) < 6:

            return False

        x_values = [
            point[0]
            for point in history
        ]

        y_values = [
            point[1]
            for point in history
        ]

        x_range = (
            max(x_values)
            - min(x_values)
        )

        y_range = (
            max(y_values)
            - min(y_values)
        )

        movement = max(
            x_range,
            y_range
        )

        return movement > 0.08


    # ==================================================
    # WAVE 检测
    # ==================================================

    def detect_wave(
        self,
        wrist,
        shoulder,
        history
    ):

        # 手必须高于肩膀
        if wrist.y >= shoulder.y:

            history.clear()

            return False


        history.append(
            wrist.x
        )


        if len(history) < 10:

            return False


        # 手腕左右移动范围
        x_range = (
            max(history)
            - min(history)
        )


        if x_range < 0.15:

            return False


        # =========================
        # 计算运动方向
        # =========================

        directions = []


        for i in range(
            1,
            len(history)
        ):

            delta = (
                history[i]
                - history[i - 1]
            )


            if abs(delta) < 0.008:

                continue


            if delta > 0:

                directions.append(1)

            else:

                directions.append(-1)


        if len(directions) < 4:

            return False


        # =========================
        # 计算方向变化
        # =========================

        direction_changes = 0


        for i in range(
            1,
            len(directions)
        ):

            if (
                directions[i]
                != directions[i - 1]
            ):

                direction_changes += 1


        if direction_changes < 2:

            return False


        if direction_changes > 8:

            return False


        # =========================
        # 左右方向都必须出现
        # =========================

        positive_count = (
            directions.count(1)
        )

        negative_count = (
            directions.count(-1)
        )


        if positive_count < 2:

            return False


        if negative_count < 2:

            return False


        return True


    # ==================================================
    # 原始动作识别
    # ==================================================

    def detect_action_raw(
        self,
        pose_landmarks
    ):

        # =========================
        # 获取关键点
        # =========================

        left_shoulder = (
            pose_landmarks[11]
        )

        right_shoulder = (
            pose_landmarks[12]
        )

        left_wrist = (
            pose_landmarks[15]
        )

        right_wrist = (
            pose_landmarks[16]
        )


        # =========================
        # 判断手是否抬起
        # =========================

        left_hand_up = (
            left_wrist.y
            < left_shoulder.y
        )

        right_hand_up = (
            right_wrist.y
            < right_shoulder.y
        )


        # =========================
        # WAVE
        # =========================

        wave_left = self.detect_wave(
            left_wrist,
            left_shoulder,
            self.left_wave_history
        )


        wave_right = self.detect_wave(
            right_wrist,
            right_shoulder,
            self.right_wave_history
        )


        if (
            wave_left
            or wave_right
        ):

            return "WAVE"


        # =========================
        # 已知动作
        # =========================

        if (
            left_hand_up
            and right_hand_up
        ):

            return "BOTH_HANDS_UP"


        if left_hand_up:

            return "LEFT_HAND_UP"


        if right_hand_up:

            return "RIGHT_HAND_UP"


        # ==================================================
        # 判断是否存在明显运动
        # ==================================================

        left_movement = (
            self.detect_movement(
                left_wrist,
                self.left_wrist_history
            )
        )


        right_movement = (
            self.detect_movement(
                right_wrist,
                self.right_wrist_history
            )
        )


        # =========================
        # 有明显运动，但不是已知动作
        # =========================

        if (
            left_movement
            or right_movement
        ):

            return "UNKNOWN_ACTION"


        # =========================
        # 没有明显动作
        # =========================

        return "NO ACTION"


    # ==================================================
    # 动作稳定
    # ==================================================

    def detect_action(
        self,
        pose_landmarks
    ):

        action = self.detect_action_raw(
            pose_landmarks
        )


        self.action_history.append(
            action
        )


        # =========================
        # 统计最近动作
        # =========================

        counts = {}


        for item in self.action_history:

            if item not in counts:

                counts[item] = 0

            counts[item] += 1


        stable_action = max(
            counts,
            key=counts.get
        )


        return stable_action


    # ==================================================
    # 对外接口
    # ==================================================

    def detect(
        self,
        pose_landmarks
    ):

        if not pose_landmarks:

            self.action_history.clear()

            self.left_wave_history.clear()

            self.right_wave_history.clear()

            self.left_wrist_history.clear()

            self.right_wrist_history.clear()

            return "NO ACTION"


        action = self.detect_action(
            pose_landmarks
        )


        return action