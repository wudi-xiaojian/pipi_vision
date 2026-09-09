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


        # ==================================================
        # NOT_IN_SEAT
        # ==================================================

        if action == "NOT_IN_SEAT":

            return "The user is not in their seat."


        # ==================================================
        # WAVE
        # ==================================================

        if action == "WAVE":

            if emotion == "HAPPINESS":

                return (
                    "The user is happily waving."
                )

            if emotion == "SADNESS":

                return (
                    "The user is waving and looks sad."
                )

            return "The user is waving."


        # ==================================================
        # HAND_UP
        # ==================================================

        if action == "HAND_UP":

            if emotion == "HAPPINESS":

                return (
                    "The user is happily raising a hand."
                )

            if emotion == "SADNESS":

                return (
                    "The user is raising a hand and looks sad."
                )

            return "The user has raised a hand."


        # ==================================================
        # HEAD_DOWN
        # ==================================================

        if action == "HEAD_DOWN":

            if emotion == "SADNESS":

                return (
                    "The user looks sad and is looking down."
                )

            return "The user is looking down."


        # ==================================================
        # DAYDREAMING
        # ==================================================

        if action == "DAYDREAMING":

            return (
                "The user appears to be daydreaming."
            )


        # ==================================================
        # NO ACTION
        # ==================================================

        if action == "NO ACTION":

            if emotion == "HAPPINESS":

                return "The user looks happy."

            if emotion == "SADNESS":

                return "The user looks sad."

            if emotion == "ANGER":

                return "The user looks angry."

            if emotion == "SURPRISE":

                return "The user looks surprised."

            if emotion == "FEAR":

                return "The user looks afraid."

            if emotion == "DISGUST":

                return "The user looks disgusted."

            if emotion == "NEUTRAL":

                return (
                    "The user has a neutral expression."
                )

            return "The user is sitting normally."


        # ==================================================
        # 默认
        # ==================================================

        return "The user's behavior has been detected."