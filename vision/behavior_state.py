class BehaviorState:
    """
    PiPi Vision 行为状态融合层。

    职责：
        1. 保存 Emotion / Pose / Hand 的原始识别结果
        2. 将多个模块的结果组合成最终行为
        3. 根据最终行为生成文字描述

    当前支持：

    Emotion:
        HAPPINESS
        SURPRISE
        ANGER
        SADNESS
        NEUTRAL

    Pose Action:
        NO ACTION
        WAVE
        HAND_UP
        HEAD_DOWN
        DAYDREAMING
        NOT_IN_SEAT

    Hand Gesture:
        NO GESTURE
        OPEN_PALM
        THUMBS_UP

    Final Behavior:
        NO ACTION
        WAVE
        HAND_UP
        HEAD_DOWN
        DAYDREAMING
        NOT_IN_SEAT
        THUMBS_UP
    """

    def __init__(self):
        self.emotion = "NEUTRAL"
        self.pose_action = "NO ACTION"
        self.hand_gesture = "NO GESTURE"

        self.final_behavior = "NO ACTION"
        self.description = "The user is doing nothing."

    def update(
        self,
        emotion,
        pose_action,
        hand_gesture
    ):
        """
        更新三个视觉模块的原始结果，
        然后计算最终行为。
        """

        self.emotion = emotion or "NEUTRAL"
        self.pose_action = pose_action or "NO ACTION"
        self.hand_gesture = hand_gesture or "NO GESTURE"

        self.final_behavior = self.combine_behavior()

        self.description = self.get_description()

    def combine_behavior(self):
        """
        根据 Pose + Hand 结果组合最终行为。

        当前规则：

        WAVE + OPEN_PALM
            → WAVE

        HAND_UP + OPEN_PALM
            → HAND_UP

        THUMBS_UP
            → THUMBS_UP

        NOT_IN_SEAT
            → NOT_IN_SEAT

        HEAD_DOWN
            → HEAD_DOWN

        DAYDREAMING
            → DAYDREAMING

        其他情况
            → NO ACTION
        """

        # ==========================================
        # 1. 人不在座位
        # ==========================================
        if self.pose_action == "NOT_IN_SEAT":
            return "NOT_IN_SEAT"

        # ==========================================
        # 2. 挥手
        #
        # Pose 判断 WAVE
        # +
        # Hand 判断 OPEN_PALM
        # ==========================================
        if (
            self.pose_action == "WAVE"
            and self.hand_gesture == "OPEN_PALM"
        ):
            return "WAVE"

        # ==========================================
        # 3. 举手
        #
        # Pose 判断 HAND_UP
        # +
        # Hand 判断 OPEN_PALM
        # ==========================================
        if (
            self.pose_action == "HAND_UP"
            and self.hand_gesture == "OPEN_PALM"
        ):
            return "HAND_UP"

        # ==========================================
        # 4. 点赞
        #
        # THUMBS_UP 不需要 Pose 配合
        # ==========================================
        if self.hand_gesture == "THUMBS_UP":
            return "THUMBS_UP"

        # ==========================================
        # 5. 低头
        # ==========================================
        if self.pose_action == "HEAD_DOWN":
            return "HEAD_DOWN"

        # ==========================================
        # 6. 发呆
        # ==========================================
        if self.pose_action == "DAYDREAMING":
            return "DAYDREAMING"

        # ==========================================
        # 7. 其他情况
        # ==========================================
        return "NO ACTION"

    def get_description(self):
        """
        根据最终行为生成描述。
        """

        descriptions = {
            "WAVE": "The user is waving.",
            "HAND_UP": "The user has raised a hand.",
            "THUMBS_UP": "The user is giving a thumbs up.",
            "HEAD_DOWN": "The user has lowered their head.",
            "DAYDREAMING": "The user appears to be daydreaming.",
            "NOT_IN_SEAT": "The user is not in the seat.",
            "NO ACTION": "The user is doing nothing.",
        }

        return descriptions.get(
            self.final_behavior,
            "The user is doing nothing."
        )

    def get_state(self):
        """
        返回当前完整行为状态。

        方便后面的 Agent / main.py 使用。
        """

        return {
            "emotion": self.emotion,
            "pose_action": self.pose_action,
            "hand_gesture": self.hand_gesture,
            "final_behavior": self.final_behavior,
            "description": self.description,
        }

    def reset(self):
        """
        重置状态。
        """

        self.emotion = "NEUTRAL"
        self.pose_action = "NO ACTION"
        self.hand_gesture = "NO GESTURE"

        self.final_behavior = "NO ACTION"
        self.description = "The user is doing nothing."