from collections import deque


class EmotionDetector:

    def __init__(self, history_size=10):

        # =========================
        # 保存最近若干帧的表情结果
        # =========================

        self.history = deque(maxlen=history_size)

        # 当前稳定表情
        self.current_emotion = "NEUTRAL"


    def detect(self, blendshapes):

        # =========================
        # 没有 BlendShapes
        # =========================

        if not blendshapes:
            return "UNKNOWN"


        # =========================
        # 转换成字典
        # =========================

        scores = {
            item.category_name: item.score
            for item in blendshapes
        }


        # ==================================================
        # 获取面部特征
        # ==================================================

        # 嘴角上扬
        smile_left = scores.get(
            "mouthSmileLeft", 0
        )

        smile_right = scores.get(
            "mouthSmileRight", 0
        )


        # 眼角收紧
        cheek_squint_left = scores.get(
            "cheekSquintLeft", 0
        )

        cheek_squint_right = scores.get(
            "cheekSquintRight", 0
        )


        # 嘴角下垂
        mouth_frown_left = scores.get(
            "mouthFrownLeft", 0
        )

        mouth_frown_right = scores.get(
            "mouthFrownRight", 0
        )


        # 下颌张开
        jaw_open = scores.get(
            "jawOpen", 0
        )


        # 眉毛内侧上扬
        brow_inner_up = scores.get(
            "browInnerUp", 0
        )


        # 眉毛下压
        brow_down_left = scores.get(
            "browDownLeft", 0
        )

        brow_down_right = scores.get(
            "browDownRight", 0
        )


        # 眼睛睁大
        eye_wide_left = scores.get(
            "eyeWideLeft", 0
        )

        eye_wide_right = scores.get(
            "eyeWideRight", 0
        )


        # 眼睛眯起
        eye_squint_left = scores.get(
            "eyeSquintLeft", 0
        )

        eye_squint_right = scores.get(
            "eyeSquintRight", 0
        )


        # 鼻子皱起
        nose_sneer_left = scores.get(
            "noseSneerLeft", 0
        )

        nose_sneer_right = scores.get(
            "noseSneerRight", 0
        )


        # 上唇上提
        mouth_upper_up_left = scores.get(
            "mouthUpperUpLeft", 0
        )

        mouth_upper_up_right = scores.get(
            "mouthUpperUpRight", 0
        )


        # 嘴巴横向拉伸
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


        nose_sneer = (
            nose_sneer_left + nose_sneer_right
        ) / 2


        mouth_upper_up = (
            mouth_upper_up_left
            + mouth_upper_up_right
        ) / 2


        mouth_stretch = (
            mouth_stretch_left
            + mouth_stretch_right
        ) / 2


        # ==================================================
        # 1. 😊 快乐 / 高兴
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
        # 2. 😮 惊讶
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
        # 3. 😠 愤怒 / 生气
        # ==================================================

        elif (
            brow_down > 0.30
            and (
                eye_squint > 0.15
                or jaw_open > 0.20
            )
        ):

            emotion = "ANGER"


        # ==================================================
        # 4. 😨 恐惧 / 害怕
        # ==================================================

        elif (
            brow_inner_up > 0.20
            and eye_wide > 0.20
            and (
                mouth_stretch > 0.15
                or jaw_open > 0.15
            )
        ):

            emotion = "FEAR"


        # ==================================================
        # 5. 🤢 厌恶
        # ==================================================

        elif (
            nose_sneer > 0.20
            or mouth_upper_up > 0.25
        ):

            emotion = "DISGUST"


        # ==================================================
        # 6. 😢 悲伤 / 难过
        # ==================================================

        elif (
            mouth_frown > 0.20
            and brow_inner_up > 0.15
        ):

            emotion = "SADNESS"


        # ==================================================
        # 7. 😐 中性
        # ==================================================

        else:

            emotion = "NEUTRAL"


        # ==================================================
        # 保存历史结果
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
        # 找出现次数最多的表情
        # ==================================================

        stable_emotion = max(
            counts,
            key=counts.get
        )


        # ==================================================
        # 更新当前表情
        # ==================================================

        self.current_emotion = stable_emotion


        return self.current_emotion