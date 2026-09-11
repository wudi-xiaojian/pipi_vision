class BehaviorState:
    """
    PiPi Vision 行为状态融合层。

    职责：
        1. 保存 Emotion / Pose / Hand / High Five 的原始识别结果
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

    High Five:
        NO ACTION
        HAND_APPROACHING
        HIGH_FIVE
        COOLDOWN

    Final Behavior:
        NO ACTION
        WAVE
        HAND_UP
        THUMBS_UP
        HEAD_DOWN
        DAYDREAMING
        NOT_IN_SEAT
        HAND_APPROACHING
        HIGH_FIVE
    """

    def __init__(self):

        # ============================================================
        # 原始状态
        # ============================================================

        self.emotion = "NEUTRAL"

        self.pose_action = "NO ACTION"

        self.hand_gesture = "NO GESTURE"

        self.high_five_action = "NO ACTION"

        # ============================================================
        # 最终行为
        # ============================================================

        self.final_behavior = "NO ACTION"

        self.description = (
            "The user is doing nothing."
        )

    # ================================================================
    # Update
    # ================================================================

    def update(
        self,
        emotion,
        pose_action,
        hand_gesture,
        high_five_action="NO ACTION"
    ):
        """
        更新视觉模块状态。

        参数：
            emotion:
                当前表情

            pose_action:
                当前上半身动作

            hand_gesture:
                当前手势

            high_five_action:
                当前 High Five 状态

        High Five 使用默认值：
            "NO ACTION"

        这样即使其他旧代码仍然只传三个参数，
        也不会报错。
        """

        # ============================================================
        # Emotion
        # ============================================================

        self.emotion = (
            emotion
            or "NEUTRAL"
        )

        # ============================================================
        # Pose
        # ============================================================

        self.pose_action = (
            pose_action
            or "NO ACTION"
        )

        # ============================================================
        # Hand
        # ============================================================

        self.hand_gesture = (
            hand_gesture
            or "NO GESTURE"
        )

        # ============================================================
        # High Five
        # ============================================================

        self.high_five_action = (
            high_five_action
            or "NO ACTION"
        )

        # ============================================================
        # Fusion
        # ============================================================

        self.final_behavior = (
            self.combine_behavior()
        )

        # ============================================================
        # Description
        # ============================================================

        self.description = (
            self.get_description()
        )

    # ================================================================
    # Behavior Fusion
    # ================================================================

    def combine_behavior(self):
        """
        根据：

            Pose
            +
            Hand Gesture
            +
            High Five

        得到最终行为。

        优先级：

            1. NOT_IN_SEAT
            2. HIGH_FIVE
            3. HAND_APPROACHING
            4. WAVE
            5. HAND_UP
            6. THUMBS_UP
            7. HEAD_DOWN
            8. DAYDREAMING
            9. NO ACTION
        """

        # ============================================================
        # 1. 人不在座位
        # ============================================================

        if self.pose_action == "NOT_IN_SEAT":
            return "NOT_IN_SEAT"

        # ============================================================
        # 2. HIGH FIVE
        #
        # High Five 是明确的交互事件，
        # 优先级高于 WAVE / HAND_UP。
        # ============================================================

        if self.high_five_action == "HIGH_FIVE":
            return "HIGH_FIVE"

        # ============================================================
        # 3. HAND APPROACHING
        #
        # 用户正在向摄像头伸手，
        # High Five 正在发生过程中。
        # ============================================================

        if self.high_five_action == "HAND_APPROACHING":
            return "HAND_APPROACHING"

        # ============================================================
        # 4. WAVE
        # ============================================================

        if (
            self.pose_action == "WAVE"
            and
            self.hand_gesture == "OPEN_PALM"
        ):
            return "WAVE"

        # ============================================================
        # 5. HAND UP
        # ============================================================

        if (
            self.pose_action == "HAND_UP"
            and
            self.hand_gesture == "OPEN_PALM"
        ):
            return "HAND_UP"

        # ============================================================
        # 6. THUMBS UP
        # ============================================================

        if self.hand_gesture == "THUMBS_UP":
            return "THUMBS_UP"

        # ============================================================
        # 7. HEAD DOWN
        # ============================================================

        if self.pose_action == "HEAD_DOWN":
            return "HEAD_DOWN"

        # ============================================================
        # 8. DAYDREAMING
        # ============================================================

        if self.pose_action == "DAYDREAMING":
            return "DAYDREAMING"

        # ============================================================
        # 9. NO ACTION
        # ============================================================

        return "NO ACTION"

    # ================================================================
    # Description
    # ================================================================

    def get_description(self):
        """
        根据最终行为生成英文描述。
        """

        descriptions = {

            "WAVE":
                "The user is waving.",

            "HAND_UP":
                "The user has raised a hand.",

            "THUMBS_UP":
                "The user is giving a thumbs up.",

            "HEAD_DOWN":
                "The user has lowered their head.",

            "DAYDREAMING":
                "The user appears to be daydreaming.",

            "NOT_IN_SEAT":
                "The user is not in the seat.",

            "HAND_APPROACHING":
                "The user is approaching for a high five.",

            "HIGH_FIVE":
                "The user is giving a high five.",

            "NO ACTION":
                "The user is doing nothing.",
        }

        return descriptions.get(
            self.final_behavior,
            "The user is doing nothing."
        )

    # ================================================================
    # Get State
    # ================================================================

    def get_state(self):
        """
        返回当前完整行为状态。

        给 main.py 或后续 Agent 使用。
        """

        return {

            "emotion":
                self.emotion,

            "pose_action":
                self.pose_action,

            "hand_gesture":
                self.hand_gesture,

            "high_five_action":
                self.high_five_action,

            "final_behavior":
                self.final_behavior,

            "description":
                self.description,
        }

    # ================================================================
    # Reset
    # ================================================================

    def reset(self):
        """
        重置所有状态。
        """

        self.emotion = "NEUTRAL"

        self.pose_action = "NO ACTION"

        self.hand_gesture = "NO GESTURE"

        self.high_five_action = "NO ACTION"

        self.final_behavior = "NO ACTION"

        self.description = (
            "The user is doing nothing."
        )