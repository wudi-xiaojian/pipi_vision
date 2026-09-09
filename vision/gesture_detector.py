import math


class GestureDetector:
    """
    基于 MediaPipe Hand Landmarks 的基础手势识别。

    当前支持：
        OPEN_PALM
        THUMBS_UP
        NO GESTURE
    """

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    @staticmethod
    def distance(p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y

        return math.sqrt(
            dx * dx + dy * dy
        )

    @staticmethod
    def angle(a, b, c):
        """
        计算 A-B-C 三点中 B 点的夹角。
        """

        ba_x = a.x - b.x
        ba_y = a.y - b.y

        bc_x = c.x - b.x
        bc_y = c.y - b.y

        ba_length = math.sqrt(
            ba_x * ba_x +
            ba_y * ba_y
        )

        bc_length = math.sqrt(
            bc_x * bc_x +
            bc_y * bc_y
        )

        if ba_length == 0 or bc_length == 0:
            return 0.0

        cosine = (
            ba_x * bc_x +
            ba_y * bc_y
        ) / (
            ba_length *
            bc_length
        )

        cosine = max(
            -1.0,
            min(1.0, cosine)
        )

        return math.degrees(
            math.acos(cosine)
        )

    # ---------------------------------------------------------
    # 食指
    # ---------------------------------------------------------

    def is_index_extended(self, landmarks):
        mcp = landmarks[self.INDEX_MCP]
        pip = landmarks[self.INDEX_PIP]
        dip = landmarks[self.INDEX_DIP]
        tip = landmarks[self.INDEX_TIP]

        angle_pip = self.angle(
            mcp,
            pip,
            dip
        )

        angle_dip = self.angle(
            pip,
            dip,
            tip
        )

        return (
            angle_pip > 160 and
            angle_dip > 160
        )

    # ---------------------------------------------------------
    # 中指
    # ---------------------------------------------------------

    def is_middle_extended(self, landmarks):
        mcp = landmarks[self.MIDDLE_MCP]
        pip = landmarks[self.MIDDLE_PIP]
        dip = landmarks[self.MIDDLE_DIP]
        tip = landmarks[self.MIDDLE_TIP]

        angle_pip = self.angle(
            mcp,
            pip,
            dip
        )

        angle_dip = self.angle(
            pip,
            dip,
            tip
        )

        return (
            angle_pip > 160 and
            angle_dip > 160
        )

    # ---------------------------------------------------------
    # 无名指
    # ---------------------------------------------------------

    def is_ring_extended(self, landmarks):
        mcp = landmarks[self.RING_MCP]
        pip = landmarks[self.RING_PIP]
        dip = landmarks[self.RING_DIP]
        tip = landmarks[self.RING_TIP]

        angle_pip = self.angle(
            mcp,
            pip,
            dip
        )

        angle_dip = self.angle(
            pip,
            dip,
            tip
        )

        return (
            angle_pip > 160 and
            angle_dip > 160
        )

    # ---------------------------------------------------------
    # 小拇指
    # ---------------------------------------------------------

    def is_pinky_extended(self, landmarks):
        mcp = landmarks[self.PINKY_MCP]
        pip = landmarks[self.PINKY_PIP]
        dip = landmarks[self.PINKY_DIP]
        tip = landmarks[self.PINKY_TIP]

        angle_pip = self.angle(
            mcp,
            pip,
            dip
        )

        angle_dip = self.angle(
            pip,
            dip,
            tip
        )

        return (
            angle_pip > 160 and
            angle_dip > 160
        )

    # ---------------------------------------------------------
    # 拇指
    # ---------------------------------------------------------

    def is_thumb_extended(self, landmarks):
        wrist = landmarks[self.WRIST]
        mcp = landmarks[self.THUMB_MCP]
        ip = landmarks[self.THUMB_IP]
        tip = landmarks[self.THUMB_TIP]

        angle_ip = self.angle(
            mcp,
            ip,
            tip
        )

        wrist_to_tip = self.distance(
            wrist,
            tip
        )

        wrist_to_mcp = self.distance(
            wrist,
            mcp
        )

        return (
            angle_ip > 150 and
            wrist_to_tip > wrist_to_mcp * 1.2
        )

    # ---------------------------------------------------------
    # 获取五指状态
    # ---------------------------------------------------------

    def get_finger_states(self, landmarks):
        return {
            "thumb": self.is_thumb_extended(
                landmarks
            ),
            "index": self.is_index_extended(
                landmarks
            ),
            "middle": self.is_middle_extended(
                landmarks
            ),
            "ring": self.is_ring_extended(
                landmarks
            ),
            "pinky": self.is_pinky_extended(
                landmarks
            ),
        }

    # ---------------------------------------------------------
    # OPEN PALM
    # ---------------------------------------------------------

    def is_open_palm(self, landmarks):
        states = self.get_finger_states(
            landmarks
        )

        return all(
            states.values()
        )

    # ---------------------------------------------------------
    # THUMBS UP
    # ---------------------------------------------------------

    def is_thumbs_up(self, landmarks):
        states = self.get_finger_states(
            landmarks
        )

        # 拇指必须伸展
        if not states["thumb"]:
            return False

        # 其他四根手指必须弯曲
        if states["index"]:
            return False

        if states["middle"]:
            return False

        if states["ring"]:
            return False

        if states["pinky"]:
            return False

        # 判断拇指方向
        wrist = landmarks[self.WRIST]
        thumb_mcp = landmarks[self.THUMB_MCP]
        thumb_tip = landmarks[self.THUMB_TIP]

        # 拇指尖相对于拇指根部的方向
        dx = thumb_tip.x - thumb_mcp.x
        dy = thumb_tip.y - thumb_mcp.y

        # 图像坐标中 y 越小越靠上
        if dy >= 0:
            return False

        # 拇指必须有一定的向上趋势
        thumb_length = math.sqrt(
            dx * dx +
            dy * dy
        )

        if thumb_length < 0.05:
            return False

        # 排除完全水平的拇指
        upward_ratio = (-dy) / thumb_length

        if upward_ratio < 0.55:
            return False

        return True

    # ---------------------------------------------------------
    # 主识别
    # ---------------------------------------------------------

    def detect(self, landmarks):
        if landmarks is None:
            return "NO GESTURE"

        if len(landmarks) < 21:
            return "NO GESTURE"

        # 优先判断 OPEN_PALM
        if self.is_open_palm(landmarks):
            return "OPEN_PALM"

        # 再判断 THUMBS_UP
        if self.is_thumbs_up(landmarks):
            return "THUMBS_UP"

        return "NO GESTURE"