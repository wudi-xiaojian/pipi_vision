from collections import deque


class EmotionDetector:

    def __init__(self, history_size=10):

        # ==================================================
        # 基础配置
        # ==================================================

        self.history = deque(maxlen=history_size)

        # 当前稳定情绪
        self.current_emotion = "NEUTRAL"

        # 当前是否处于哭泣状态
        self.is_crying = False

        # 当前哭泣分数
        self.crying_score = 0.0

        # 当前生气分数
        self.anger_score = 0.0

        # 当前伤心分数
        self.sadness_score = 0.0


    def detect(self, blendshapes):

        # ==================================================
        # 没有 BlendShapes
        # ==================================================

        if not blendshapes:

            self.is_crying = False

            return "UNKNOWN"


        # ==================================================
        # BlendShapes 转换成字典
        # ==================================================

        scores = {
            item.category_name: item.score
            for item in blendshapes
        }


        # ==================================================
        # 获取面部特征
        # ==================================================

        # --------------------------------------------------
        # 嘴角上扬
        # --------------------------------------------------

        smile_left = scores.get(
            "mouthSmileLeft", 0
        )

        smile_right = scores.get(
            "mouthSmileRight", 0
        )


        # --------------------------------------------------
        # 眼角收紧
        # --------------------------------------------------

        cheek_squint_left = scores.get(
            "cheekSquintLeft", 0
        )

        cheek_squint_right = scores.get(
            "cheekSquintRight", 0
        )


        # --------------------------------------------------
        # 嘴角下垂
        # --------------------------------------------------

        mouth_frown_left = scores.get(
            "mouthFrownLeft", 0
        )

        mouth_frown_right = scores.get(
            "mouthFrownRight", 0
        )


        # --------------------------------------------------
        # 下颌张开
        # --------------------------------------------------

        jaw_open = scores.get(
            "jawOpen", 0
        )


        # --------------------------------------------------
        # 眉毛内侧上扬
        # --------------------------------------------------

        brow_inner_up = scores.get(
            "browInnerUp", 0
        )


        # --------------------------------------------------
        # 眉毛下压
        # --------------------------------------------------

        brow_down_left = scores.get(
            "browDownLeft", 0
        )

        brow_down_right = scores.get(
            "browDownRight", 0
        )


        # --------------------------------------------------
        # 眼睛睁大
        # --------------------------------------------------

        eye_wide_left = scores.get(
            "eyeWideLeft", 0
        )

        eye_wide_right = scores.get(
            "eyeWideRight", 0
        )


        # --------------------------------------------------
        # 眼睛眯起
        # --------------------------------------------------

        eye_squint_left = scores.get(
            "eyeSquintLeft", 0
        )

        eye_squint_right = scores.get(
            "eyeSquintRight", 0
        )


        # --------------------------------------------------
        # 嘴巴横向拉伸
        # --------------------------------------------------

        mouth_stretch_left = scores.get(
            "mouthStretchLeft", 0
        )

        mouth_stretch_right = scores.get(
            "mouthStretchRight", 0
        )


        # ==================================================
        # 计算组合特征
        # ==================================================

        smile = (
            smile_left + smile_right
        ) / 2


        cheek_squint = (
            cheek_squint_left + cheek_squint_right
        ) / 2


        mouth_frown = (
            mouth_frown_left + mouth_frown_right
        ) / 2


        brow_down = (
            brow_down_left + brow_down_right
        ) / 2


        eye_wide = (
            eye_wide_left + eye_wide_right
        ) / 2


        eye_squint = (
            eye_squint_left + eye_squint_right
        ) / 2


        mouth_stretch = (
            mouth_stretch_left + mouth_stretch_right
        ) / 2


        # ==================================================
        # 默认状态
        # ==================================================

        emotion = "NEUTRAL"


        # ==================================================
        # 1. 😊 HAPPINESS
        #
        # 保留你目前已经验证比较稳定的判断
        # ==================================================

        if (
            smile > 0.45
            and (
                cheek_squint > 0.15
                or smile > 0.60
            )
        ):

            emotion = "HAPPINESS"


        # ==================================================
        # 2. 😮 SURPRISE
        #
        # 保留目前稳定判断
        # ==================================================

        elif (
            jaw_open > 0.25
            and (
                eye_wide > 0.12
                or brow_inner_up > 0.15
            )
        ):

            emotion = "SURPRISE"


        # ==================================================
        # 3 / 4. 😠 ANGER / 😢 SADNESS
        #
        # 使用双评分模型
        # ==================================================

        else:

            # ==================================================
            # 😠 ANGER SCORE
            #
            # 生气主要关注：
            #
            # browDown
            # eyeSquint
            #
            # 眉毛下压是核心
            # ==================================================

            anger_score = (
                brow_down * 0.60
                + eye_squint * 0.30
            )


            # ==================================================
            # 😢 SADNESS SCORE
            #
            # 伤心主要关注：
            #
            # mouthFrown
            # browInnerUp
            # eyeSquint
            # mouthStretch
            # jawOpen
            #
            # 注意：
            #
            # eyeSquint 在这里允许存在。
            #
            # 因为真实哭泣时眼睛经常会闭合/眯起。
            # ==================================================

            sadness_score = (
                mouth_frown * 0.35
                + brow_inner_up * 0.30
                + eye_squint * 0.15
                + mouth_stretch * 0.10
                + jaw_open * 0.10
            )


            # ==================================================
            # 😭 CRYING SCORE
            #
            # 哭泣不是独立 emotion，
            # 而是 SADNESS 的强化状态。
            #
            # 重点：
            #
            # 嘴角下垂
            # +
            # 眉毛内侧上扬
            # +
            # 眼睛收紧
            # +
            # 嘴巴张开/拉伸
            # ==================================================

            crying_score = (
                mouth_frown * 0.30
                + brow_inner_up * 0.20
                + eye_squint * 0.20
                + jaw_open * 0.15
                + mouth_stretch * 0.15
            )


            # ==================================================
            # 保存当前分数
            # ==================================================

            self.anger_score = anger_score
            self.sadness_score = sadness_score
            self.crying_score = crying_score


            # ==================================================
            # 哭泣检测
            # ==================================================

            # 这里要求至少有：
            #
            # 嘴角下垂
            # +
            # 另外两个以上的哭泣特征
            #
            # 避免单纯眯眼就被认为在哭。
            # ==================================================

            crying_features = 0


            if mouth_frown > 0.15:

                crying_features += 1


            if brow_inner_up > 0.15:

                crying_features += 1


            if eye_squint > 0.15:

                crying_features += 1


            if jaw_open > 0.15:

                crying_features += 1


            if mouth_stretch > 0.15:

                crying_features += 1


            if (
                crying_score > 0.22
                and mouth_frown > 0.15
                and crying_features >= 3
            ):

                self.is_crying = True

            else:

                self.is_crying = False


            # ==================================================
            # 如果确认哭泣
            #
            # 直接进入 SADNESS
            # ==================================================

            if self.is_crying:

                emotion = "SADNESS"


            # ==================================================
            # 😠 ANGER
            # ==================================================

            elif (
                anger_score > 0.25
                and anger_score > sadness_score
            ):

                emotion = "ANGER"


            # ==================================================
            # 😢 SADNESS
            # ==================================================

            elif (
                sadness_score > 0.22
                and sadness_score > anger_score
            ):

                emotion = "SADNESS"


            # ==================================================
            # 😐 NEUTRAL
            # ==================================================

            else:

                emotion = "NEUTRAL"


        # ==================================================
        # 保存历史情绪
        # ==================================================

        self.history.append(
            emotion
        )


        # ==================================================
        # 时间投票
        # ==================================================

        counts = {}

        for item in self.history:

            if item not in counts:

                counts[item] = 0

            counts[item] += 1


        # ==================================================
        # 找出现次数最多的情绪
        # ==================================================

        stable_emotion = max(
            counts,
            key=counts.get
        )


        # ==================================================
        # 更新当前情绪
        # ==================================================

        self.current_emotion = stable_emotion


        return self.current_emotion