import time
import math
from collections import deque


class HighFiveDetector:
    """
    检测“用户与摄像头击掌”。

    逻辑：

        OPEN_PALM
            +
        手掌快速靠近摄像头
            +
        手掌面积明显变大
            +
        达到近距离阈值
            ↓
        HIGH_FIVE

    这是“用户 ↔ 摄像头”的虚拟击掌，
    不是左右手互相击掌。
    """

    def __init__(
        self,
        history_size=12,
        min_palm_area=0.035,
        near_palm_area=0.16,
        approach_ratio=1.35,
        cooldown_seconds=1.2,
    ):

        # ==================================================
        # 参数
        # ==================================================

        self.history_size = history_size

        # 最小手掌面积
        #
        # 用归一化坐标计算
        #
        # 例如：
        # 0.035 = 手掌已经比较明显
        #
        self.min_palm_area = (
            min_palm_area
        )

        # 接近摄像头时的手掌面积
        self.near_palm_area = (
            near_palm_area
        )

        # 面积至少增长多少
        #
        # 当前面积 / 初始面积
        #
        # >= 1.35
        #
        self.approach_ratio = (
            approach_ratio
        )

        # HIGH_FIVE 触发后的冷却时间
        self.cooldown_seconds = (
            cooldown_seconds
        )


        # ==================================================
        # 手掌面积历史
        # ==================================================

        self.palm_area_history = deque(
            maxlen=history_size
        )


        # ==================================================
        # 时间
        # ==================================================

        self.last_high_five_time = 0.0

        self.approach_start_time = None


        # ==================================================
        # 状态
        # ==================================================

        self.current_state = (
            "NO ACTION"
        )


    # ======================================================
    # 计算距离
    # ======================================================

    @staticmethod
    def distance(
        p1,
        p2
    ):

        dx = p1.x - p2.x
        dy = p1.y - p2.y

        return math.sqrt(
            dx * dx +
            dy * dy
        )


    # ======================================================
    # 计算手掌面积
    # ======================================================

    def calculate_palm_area(
        self,
        landmarks
    ):
        """
        使用手掌几个关键点构造一个
        简单的手掌面积估计。

        使用：

            0 Wrist
            5 Index MCP
            9 Middle MCP
            13 Ring MCP
            17 Pinky MCP

        将它们形成一个近似多边形。

        因为 MediaPipe 坐标已经归一化，
        所以面积也是归一化面积。
        """

        if (
            landmarks is None
            or len(landmarks) < 21
        ):
            return 0.0


        points = [
            landmarks[0],
            landmarks[5],
            landmarks[9],
            landmarks[13],
            landmarks[17],
        ]


        # ==================================================
        # Shoelace Formula
        # ==================================================

        area = 0.0

        for i in range(
            len(points)
        ):

            p1 = points[i]

            p2 = points[
                (i + 1)
                % len(points)
            ]


            area += (
                p1.x * p2.y
                - p2.x * p1.y
            )


        area = abs(area) / 2.0

        return area


    # ======================================================
    # 计算手掌尺寸
    # ======================================================

    def calculate_palm_size(
        self,
        landmarks
    ):
        """
        通过几个关键点估算手掌尺寸。

        这里使用：

            Wrist → Middle MCP

        以及：

            Index MCP → Pinky MCP

        的距离。
        """

        if (
            landmarks is None
            or len(landmarks) < 21
        ):
            return 0.0


        wrist = landmarks[0]

        middle_mcp = landmarks[9]

        index_mcp = landmarks[5]

        pinky_mcp = landmarks[17]


        length1 = self.distance(
            wrist,
            middle_mcp
        )

        length2 = self.distance(
            index_mcp,
            pinky_mcp
        )


        return (
            length1
            + length2
        ) / 2.0


    # ======================================================
    # 判断是否正在靠近摄像头
    # ======================================================

    def is_approaching(
        self
    ):

        if len(
            self.palm_area_history
        ) < 4:

            return False


        areas = list(
            self.palm_area_history
        )


        # --------------------------------------------------
        # 较早的面积
        # --------------------------------------------------

        old_area = min(
            areas[:3]
        )


        # --------------------------------------------------
        # 当前面积
        # --------------------------------------------------

        new_area = max(
            areas[-3:]
        )


        if old_area <= 0:

            return False


        ratio = (
            new_area
            / old_area
        )


        return (
            ratio
            >= self.approach_ratio
        )


    # ======================================================
    # 判断是否冷却
    # ======================================================

    def is_in_cooldown(
        self
    ):

        return (
            time.time()
            - self.last_high_five_time
            < self.cooldown_seconds
        )


    # ======================================================
    # 检测 HIGH_FIVE
    # ======================================================

    def detect(
        self,
        landmarks,
        gesture
    ):
        """
        参数：

            landmarks:
                MediaPipe Hand Landmarks

            gesture:
                GestureDetector 的结果

        返回：

            HIGH_FIVE
            HAND_APPROACHING
            NO ACTION
        """

        # ==================================================
        # 没有手
        # ==================================================

        if (
            landmarks is None
            or len(landmarks) < 21
        ):

            self.reset()

            return "NO ACTION"


        # ==================================================
        # 必须是 OPEN_PALM
        # ==================================================

        if gesture != "OPEN_PALM":

            self.approach_start_time = None

            self.current_state = (
                "NO ACTION"
            )

            return "NO ACTION"


        # ==================================================
        # 计算手掌面积
        # ==================================================

        palm_area = (
            self.calculate_palm_area(
                landmarks
            )
        )


        # ==================================================
        # 保存历史
        # ==================================================

        self.palm_area_history.append(
            palm_area
        )


        # ==================================================
        # 面积太小
        # ==================================================

        if (
            palm_area
            < self.min_palm_area
        ):

            self.current_state = (
                "NO ACTION"
            )

            return "NO ACTION"


        # ==================================================
        # 冷却
        # ==================================================

        if self.is_in_cooldown():

            self.current_state = (
                "NO ACTION"
            )

            return "NO ACTION"


        # ==================================================
        # 检测是否靠近
        # ==================================================

        approaching = (
            self.is_approaching()
        )


        if approaching:

            if (
                self.approach_start_time
                is None
            ):

                self.approach_start_time = (
                    time.time()
                )


            self.current_state = (
                "HAND_APPROACHING"
            )


        else:

            self.approach_start_time = (
                None
            )

            self.current_state = (
                "NO ACTION"
            )


        # ==================================================
        # 到达近距离
        # ==================================================

        if (
            palm_area
            >= self.near_palm_area
            and approaching
        ):

            self.last_high_five_time = (
                time.time()
            )

            self.current_state = (
                "HIGH_FIVE"
            )

            return "HIGH_FIVE"


        # ==================================================
        # 正在靠近
        # ==================================================

        if approaching:

            return "HAND_APPROACHING"


        return "NO ACTION"


    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.palm_area_history.clear()

        self.approach_start_time = (
            None
        )

        self.current_state = (
            "NO ACTION"
        )