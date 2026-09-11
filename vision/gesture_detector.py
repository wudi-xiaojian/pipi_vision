import math


class GestureDetector:
    """
    基于 MediaPipe Hand Landmarks 的基础手势识别。

    当前支持：
        OPEN_PALM
        THUMBS_UP
        NO GESTURE

    注意：
        HIGH_FIVE 不在这里判断。
        HIGH_FIVE 由独立的 HighFiveDetector 负责。
    """

    # =========================================================
    # MediaPipe Hand Landmarks Index
    # =========================================================

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

    # =========================================================
    # 基础数学函数
    # =========================================================

    @staticmethod
    def distance(p1, p2):
        """
        计算两个关键点之间的 2D 欧氏距离。
        """

        dx = p1.x - p2.x
        dy = p1.y - p2.y

        return math.sqrt(
            dx * dx +
            dy * dy
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

    # =========================================================
    # 掌心中心
    # =========================================================

    def get_palm_center(self, landmarks):
        """
        计算掌心中心。

        使用：
            Wrist
            Index MCP
            Middle MCP
            Ring MCP
            Pinky MCP

        不使用拇指，避免拇指状态影响掌心计算。
        """

        points = [
            landmarks[self.WRIST],
            landmarks[self.INDEX_MCP],
            landmarks[self.MIDDLE_MCP],
            landmarks[self.RING_MCP],
            landmarks[self.PINKY_MCP],
        ]

        x = sum(
            point.x for point in points
        ) / len(points)

        y = sum(
            point.y for point in points
        ) / len(points)

        class Point:
            pass

        center = Point()

        center.x = x
        center.y = y

        return center

    # =========================================================
    # 掌部尺寸
    # =========================================================

    def get_palm_size(self, landmarks):
        """
        计算掌部尺度。

        使用：

            Wrist -> Middle MCP

        和：

            Index MCP -> Pinky MCP

        的平均值。

        用于归一化不同距离下的手部大小。
        """

        wrist = landmarks[self.WRIST]

        middle_mcp = landmarks[
            self.MIDDLE_MCP
        ]

        index_mcp = landmarks[
            self.INDEX_MCP
        ]

        pinky_mcp = landmarks[
            self.PINKY_MCP
        ]

        palm_length = self.distance(
            wrist,
            middle_mcp
        )

        palm_width = self.distance(
            index_mcp,
            pinky_mcp
        )

        return (
            palm_length +
            palm_width
        ) / 2.0

    # =========================================================
    # 食指
    # =========================================================

    def is_index_extended(self, landmarks):
        """
        判断食指是否伸直。
        """

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

    # =========================================================
    # 中指
    # =========================================================

    def is_middle_extended(self, landmarks):
        """
        判断中指是否伸直。
        """

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

    # =========================================================
    # 无名指
    # =========================================================

    def is_ring_extended(self, landmarks):
        """
        判断无名指是否伸直。
        """

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

    # =========================================================
    # 小拇指
    # =========================================================

    def is_pinky_extended(self, landmarks):
        """
        判断小拇指是否伸直。
        """

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

    # =========================================================
    # 拇指
    # =========================================================

    def is_thumb_extended(self, landmarks):
        """
        判断拇指是否真正伸展。

        不再只依赖：

            IP 角度
            Wrist -> Thumb Tip

        而是增加：

            1. 拇指 IP 关节角度
            2. Wrist -> Thumb Tip 距离
            3. Thumb MCP -> Thumb Tip 距离
            4. Thumb Tip -> Palm Center 距离

        主要用于解决：

            四指伸直 + 拇指弯曲
                    ↓
            错误 OPEN_PALM

        """

        wrist = landmarks[self.WRIST]

        mcp = landmarks[
            self.THUMB_MCP
        ]

        ip = landmarks[
            self.THUMB_IP
        ]

        tip = landmarks[
            self.THUMB_TIP
        ]

        # -----------------------------------------------------
        # 掌心中心
        # -----------------------------------------------------

        palm_center = self.get_palm_center(
            landmarks
        )

        # -----------------------------------------------------
        # 掌部尺寸
        # -----------------------------------------------------

        palm_size = self.get_palm_size(
            landmarks
        )

        if palm_size < 0.001:
            return False

        # -----------------------------------------------------
        # 条件 1：
        # 拇指 IP 关节必须比较直
        # -----------------------------------------------------

        angle_ip = self.angle(
            mcp,
            ip,
            tip
        )

        if angle_ip < 150:
            return False

        # -----------------------------------------------------
        # 条件 2：
        # Wrist -> Thumb Tip
        # -----------------------------------------------------

        wrist_to_tip = self.distance(
            wrist,
            tip
        )

        wrist_to_mcp = self.distance(
            wrist,
            mcp
        )

        if wrist_to_tip < (
            wrist_to_mcp * 1.25
        ):
            return False

        # -----------------------------------------------------
        # 条件 3：
        # Thumb MCP -> Thumb Tip
        #
        # 真正伸展的拇指应该有足够长度。
        # -----------------------------------------------------

        thumb_length = self.distance(
            mcp,
            tip
        )

        thumb_length_ratio = (
            thumb_length /
            palm_size
        )

        if thumb_length_ratio < 0.55:
            return False

        # -----------------------------------------------------
        # 条件 4：
        # Thumb Tip -> Palm Center
        #
        # 拇指如果弯曲收进掌心，
        # 拇指尖通常会明显靠近掌心。
        # -----------------------------------------------------

        thumb_tip_to_palm = self.distance(
            tip,
            palm_center
        )

        palm_separation_ratio = (
            thumb_tip_to_palm /
            palm_size
        )

        if palm_separation_ratio < 0.70:
            return False

        return True

    # =========================================================
    # 获取五指状态
    # =========================================================

    def get_finger_states(self, landmarks):
        """
        返回五根手指的伸展状态。
        """

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

    # =========================================================
    # OPEN PALM
    # =========================================================

    def is_open_palm(self, landmarks):
        """
        判断 OPEN_PALM。

        要求：

            拇指真正伸展
            +
            食指伸展
            +
            中指伸展
            +
            无名指伸展
            +
            小拇指伸展

        这样可以避免：

            四指伸直
            +
            拇指弯曲

        被错误识别成 OPEN_PALM。
        """

        states = self.get_finger_states(
            landmarks
        )

        # -----------------------------------------------------
        # 五指必须全部伸展
        # -----------------------------------------------------

        if not all(
            states.values()
        ):
            return False

        # -----------------------------------------------------
        # 再次确认拇指
        #
        # 防止 MediaPipe 在遮挡情况下，
        # 给出一个看起来比较直的拇指结构。
        # -----------------------------------------------------

        if not self.is_thumb_extended(
            landmarks
        ):
            return False

        return True

    # =========================================================
    # THUMBS UP
    # =========================================================

    def is_thumbs_up(self, landmarks):
        """
        判断 THUMBS_UP。

        条件：

            1. 拇指伸展
            2. 其他四指弯曲
            3. 拇指明显向上
            4. 拇指远离掌心
            5. 拇指不能太靠近食指 MCP
            6. 拇指不能太靠近中指 MCP
        """

        states = self.get_finger_states(
            landmarks
        )

        # -----------------------------------------------------
        # 条件 1：
        # 拇指必须伸展
        # -----------------------------------------------------

        if not states["thumb"]:
            return False

        # -----------------------------------------------------
        # 条件 2：
        # 其他四指必须弯曲
        # -----------------------------------------------------

        if states["index"]:
            return False

        if states["middle"]:
            return False

        if states["ring"]:
            return False

        if states["pinky"]:
            return False

        # -----------------------------------------------------
        # 获取关键点
        # -----------------------------------------------------

        thumb_mcp = landmarks[
            self.THUMB_MCP
        ]

        thumb_tip = landmarks[
            self.THUMB_TIP
        ]

        index_mcp = landmarks[
            self.INDEX_MCP
        ]

        middle_mcp = landmarks[
            self.MIDDLE_MCP
        ]

        # -----------------------------------------------------
        # 掌心中心
        # -----------------------------------------------------

        palm_center = self.get_palm_center(
            landmarks
        )

        # -----------------------------------------------------
        # 掌部尺寸
        # -----------------------------------------------------

        palm_size = self.get_palm_size(
            landmarks
        )

        if palm_size < 0.001:
            return False

        # =====================================================
        # 条件 3：
        # 拇指必须向上
        # =====================================================

        dx = (
            thumb_tip.x -
            thumb_mcp.x
        )

        dy = (
            thumb_tip.y -
            thumb_mcp.y
        )

        # MediaPipe 图像坐标：
        #
        # y 越小 = 越靠上
        #

        if dy >= 0:
            return False

        thumb_length = math.sqrt(
            dx * dx +
            dy * dy
        )

        if thumb_length < 0.05:
            return False

        upward_ratio = (
            -dy /
            thumb_length
        )

        # 必须明显向上
        if upward_ratio < 0.55:
            return False

        # =====================================================
        # 条件 4：
        # 拇指尖必须明显离开掌心
        # =====================================================

        thumb_tip_to_palm = self.distance(
            thumb_tip,
            palm_center
        )

        palm_separation_ratio = (
            thumb_tip_to_palm /
            palm_size
        )

        if palm_separation_ratio < 0.85:
            return False

        # =====================================================
        # 条件 5：
        # 拇指尖不能靠近食指 MCP
        #
        # 握拳时：
        #
        #     Thumb Tip
        #          ↓
        #     Index MCP
        #
        # 两者可能非常接近。
        # =====================================================

        thumb_tip_to_index_mcp = self.distance(
            thumb_tip,
            index_mcp
        )

        index_separation_ratio = (
            thumb_tip_to_index_mcp /
            palm_size
        )

        if index_separation_ratio < 0.55:
            return False

        # =====================================================
        # 条件 6：
        # 拇指尖不能靠近中指 MCP
        #
        # 进一步排除握拳。
        # =====================================================

        thumb_tip_to_middle_mcp = self.distance(
            thumb_tip,
            middle_mcp
        )

        middle_separation_ratio = (
            thumb_tip_to_middle_mcp /
            palm_size
        )

        if middle_separation_ratio < 0.45:
            return False

        # =====================================================
        # 所有条件通过
        # =====================================================

        return True

    # =========================================================
    # 主识别接口
    # =========================================================

    def detect(self, landmarks):
        """
        主手势识别接口。

        输入：
            MediaPipe Hand Landmarks

        输出：

            OPEN_PALM
            THUMBS_UP
            NO GESTURE
        """

        # -----------------------------------------------------
        # 没有关键点
        # -----------------------------------------------------

        if landmarks is None:
            return "NO GESTURE"

        # -----------------------------------------------------
        # MediaPipe Hand 必须有 21 个关键点
        # -----------------------------------------------------

        if len(landmarks) < 21:
            return "NO GESTURE"

        # -----------------------------------------------------
        # 优先判断 OPEN_PALM
        # -----------------------------------------------------

        if self.is_open_palm(
            landmarks
        ):
            return "OPEN_PALM"

        # -----------------------------------------------------
        # 再判断 THUMBS_UP
        # -----------------------------------------------------

        if self.is_thumbs_up(
            landmarks
        ):
            return "THUMBS_UP"

        # -----------------------------------------------------
        # 其他情况
        # -----------------------------------------------------

        return "NO GESTURE"