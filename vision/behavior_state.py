class BehaviorState:

    def __init__(self):

        self.emotion = "NEUTRAL"
        self.action = "NO ACTION"


    # ==================================================
    # 更新状态
    # ==================================================

    def update(
        self,
        emotion,
        action
    ):

        self.emotion = emotion
        self.action = action


    # ==================================================
    # 获取结构化状态
    # ==================================================

    def get_state(self):

        return {
            "emotion": self.emotion,
            "action": self.action
        }


    # ==================================================
    # 获取行为描述
    # ==================================================

    def get_description(self):

        emotion = self.emotion
        action = self.action


        # =========================
        # 未知动作
        # =========================

        if action == "UNKNOWN_ACTION":

            if emotion == "HAPPINESS":

                return "用户正在做未知动作，看起来很开心"

            if emotion == "SADNESS":

                return "用户正在做未知动作，看起来有些难过"

            return "检测到用户正在进行未知动作"


        # =========================
        # 开心 + 挥手
        # =========================

        if (
            emotion == "HAPPINESS"
            and action == "WAVE"
        ):

            return "用户正在开心地挥手"


        # =========================
        # 开心 + 左手
        # =========================

        if (
            emotion == "HAPPINESS"
            and action == "LEFT_HAND_UP"
        ):

            return "用户正在开心地举起左手"


        # =========================
        # 开心 + 右手
        # =========================

        if (
            emotion == "HAPPINESS"
            and action == "RIGHT_HAND_UP"
        ):

            return "用户正在开心地举起右手"


        # =========================
        # 开心 + 双手
        # =========================

        if (
            emotion == "HAPPINESS"
            and action == "BOTH_HANDS_UP"
        ):

            return "用户正在开心地举起双手"


        # =========================
        # 普通动作
        # =========================

        if action == "WAVE":

            return "用户正在挥手"


        if action == "LEFT_HAND_UP":

            return "用户正在举起左手"


        if action == "RIGHT_HAND_UP":

            return "用户正在举起右手"


        if action == "BOTH_HANDS_UP":

            return "用户正在举起双手"


        # =========================
        # 表情
        # =========================

        if emotion == "HAPPINESS":

            return "用户看起来很开心"


        if emotion == "SADNESS":

            return "用户看起来有些难过"


        if emotion == "ANGER":

            return "用户看起来有些生气"


        if emotion == "SURPRISE":

            return "用户看起来有些惊讶"


        if emotion == "FEAR":

            return "用户看起来有些害怕"


        if emotion == "DISGUST":

            return "用户看起来有些厌恶"


        if emotion == "NEUTRAL":

            return "用户表情比较自然"


        return "检测到用户行为"