import math
import time
from collections import deque


class HighFiveDetector:
    """
    Virtual High Five detector.

    This detects a user's open palm moving toward the camera,
    rather than two hands physically hitting each other.

    State machine:
        NO ACTION
            -> HAND_APPROACHING
            -> HIGH_FIVE
            -> COOLDOWN
            -> WAIT_REARM
            -> NO ACTION

    Important design points:
    - A normal raised, stationary hand should not trigger High Five.
    - HIGH_FIVE requires a clear palm-size increase and forward motion.
    - HIGH_FIVE must be confirmed for several consecutive frames.
    - Brief hand-tracking loss does not immediately erase history.
    - COOLDOWN is only entered after a real HIGH_FIVE event.
    - WAIT_REARM prevents the same hand being held near the camera
      from repeatedly triggering High Five.
    """

    NO_ACTION = "NO ACTION"
    HAND_APPROACHING = "HAND_APPROACHING"
    HIGH_FIVE = "HIGH_FIVE"
    COOLDOWN = "COOLDOWN"
    WAIT_REARM = "WAIT_REARM"

    def __init__(
        self,
        history_size=15,
        min_palm_size=0.06,
        near_palm_size=0.16,
        growth_ratio=1.18,
        fast_growth_ratio=1.05,
        depth_approach_threshold=0.008,
        confirm_frames=3,
        lost_tolerance=5,
        cooldown_seconds=0.8,
        retreat_ratio=0.75,
        rearm_palm_size=0.12,
        rearm_frames=4,
        max_approach_seconds=3.0,
    ):
        self.history_size = history_size
        self.min_palm_size = min_palm_size
        self.near_palm_size = near_palm_size
        self.growth_ratio_threshold = growth_ratio
        self.fast_growth_ratio_threshold = fast_growth_ratio
        self.depth_approach_threshold = depth_approach_threshold
        self.confirm_frames_required = confirm_frames
        self.lost_tolerance = lost_tolerance
        self.cooldown_seconds = cooldown_seconds
        self.retreat_ratio = retreat_ratio
        self.rearm_palm_size = rearm_palm_size
        self.rearm_frames_required = rearm_frames
        self.max_approach_seconds = max_approach_seconds

        self.palm_size_history = deque(maxlen=history_size)
        self.palm_area_history = deque(maxlen=history_size)
        self.depth_history = deque(maxlen=history_size)

        self.current_state = self.NO_ACTION
        self.last_high_five_time = 0.0
        self.approach_start_time = None

        self.confirm_count = 0
        self.rearm_count = 0
        self.lost_frames = 0

        self.current_palm_size = 0.0
        self.current_palm_area = 0.0
        self.current_depth = 0.0

        self.growth_ratio_value = 1.0
        self.fast_growth_ratio_value = 1.0
        self.depth_approach_value = 0.0
        self.approach_score = 0.0

        self.near_camera = False
        self.growth_ok = False
        self.fast_growth_ok = False
        self.depth_ok = False
        self.trend_ok = False

        self.high_five_triggered = False
        self.previous_palm_size = 0.0

    # ==========================================================
    # Geometry
    # ==========================================================

    @staticmethod
    def distance(p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(dx * dx + dy * dy)

    def calculate_palm_size(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            return 0.0

        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        pinky_mcp = landmarks[17]

        wrist_to_middle = self.distance(wrist, middle_mcp)
        palm_width = self.distance(index_mcp, pinky_mcp)

        return (wrist_to_middle + palm_width) / 2.0

    def calculate_palm_area(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            return 0.0

        points = [
            landmarks[0],
            landmarks[5],
            landmarks[9],
            landmarks[13],
            landmarks[17],
        ]

        area = 0.0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            area += p1.x * p2.y - p2.x * p1.y

        return abs(area) / 2.0

    def calculate_depth(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            return 0.0

        points = [
            landmarks[0],
            landmarks[5],
            landmarks[9],
            landmarks[13],
            landmarks[17],
        ]

        return sum(getattr(p, "z", 0.0) for p in points) / len(points)

    # ==========================================================
    # Temporal features
    # ==========================================================

    def calculate_growth(self):
        if len(self.palm_size_history) < 5:
            self.growth_ratio_value = 1.0
            self.fast_growth_ratio_value = 1.0
            return

        sizes = list(self.palm_size_history)
        old_size = sum(sizes[:5]) / 5.0
        recent_size = sum(sizes[-5:]) / 5.0

        if old_size > 1e-6:
            self.growth_ratio_value = recent_size / old_size
        else:
            self.growth_ratio_value = 1.0

        if len(sizes) >= 4:
            previous = sum(sizes[-4:-2]) / 2.0
            current = sum(sizes[-2:]) / 2.0
            if previous > 1e-6:
                self.fast_growth_ratio_value = current / previous
            else:
                self.fast_growth_ratio_value = 1.0
        else:
            self.fast_growth_ratio_value = 1.0

    def calculate_depth_approach(self):
        """
        MediaPipe hand z normally becomes smaller when the hand gets
        closer to the camera. Therefore:

            old_z - recent_z > threshold

        means the hand is moving toward the camera.
        """
        if len(self.depth_history) < 5:
            self.depth_approach_value = 0.0
            return

        depths = list(self.depth_history)
        old_depth = sum(depths[:5]) / 5.0
        recent_depth = sum(depths[-5:]) / 5.0

        self.depth_approach_value = old_depth - recent_depth

    def calculate_trend(self):
        if len(self.palm_size_history) < 5:
            return False

        sizes = list(self.palm_size_history)
        increasing = 0
        total = len(sizes) - 1

        for i in range(1, len(sizes)):
            if sizes[i] > sizes[i - 1] * 0.995:
                increasing += 1

        return total > 0 and increasing / total >= 0.55

    def update_features(self):
        self.calculate_growth()
        self.calculate_depth_approach()

        self.trend_ok = self.calculate_trend()
        self.growth_ok = (
            self.growth_ratio_value >= self.growth_ratio_threshold
        )
        self.fast_growth_ok = (
            self.fast_growth_ratio_value >= self.fast_growth_ratio_threshold
        )
        self.depth_ok = (
            self.depth_approach_value >= self.depth_approach_threshold
        )

        score = 0
        if self.growth_ok:
            score += 1
        if self.fast_growth_ok:
            score += 1
        if self.depth_ok:
            score += 1
        if self.trend_ok:
            score += 1

        self.approach_score = score / 4.0

    def detect_approaching(self):
        if len(self.palm_size_history) < 5:
            self.approach_score = 0.0
            return False

        self.update_features()

        # Approach is deliberately more permissive than HIGH_FIVE.
        # It is enough to see a sustained palm-size increase and trend,
        # or palm growth together with forward depth movement.
        return (
            (self.growth_ok and self.trend_ok)
            or
            (self.growth_ok and self.depth_ok)
            or
            (self.fast_growth_ok and self.depth_ok)
        )

    # ==========================================================
    # State helpers
    # ==========================================================

    def is_in_cooldown(self):
        if self.last_high_five_time <= 0:
            return False
        return (
            time.time() - self.last_high_five_time
            < self.cooldown_seconds
        )

    def start_approaching(self):
        if self.approach_start_time is None:
            self.approach_start_time = time.time()

    def approach_timeout(self):
        if self.approach_start_time is None:
            return False
        return (
            time.time() - self.approach_start_time
            > self.max_approach_seconds
        )

    def has_retreated(self):
        if len(self.palm_size_history) < 3:
            return False

        current = self.palm_size_history[-1]
        peak = max(self.palm_size_history)

        if peak <= 1e-6:
            return False

        return current / peak <= self.retreat_ratio

    def has_rearmed(self):
        """Require several frames of a clearly smaller palm."""
        if self.current_palm_size <= 0:
            return False

        return self.current_palm_size <= self.rearm_palm_size

    # ==========================================================
    # No-hand / non-open-palm handling
    # ==========================================================

    def update_no_hand(self):
        """
        Short tracking loss is tolerated only while approaching.
        It must never create COOLDOWN by itself.
        """
        if self.current_state == self.HIGH_FIVE:
            return self.HIGH_FIVE

        if self.current_state == self.COOLDOWN:
            if self.is_in_cooldown():
                return self.COOLDOWN

            self.current_state = self.WAIT_REARM
            self.rearm_count = 0
            return self.WAIT_REARM

        if self.current_state == self.WAIT_REARM:
            # No hand is a valid re-arm condition after the cooldown.
            self.rearm_count += 1
            if self.rearm_count >= self.rearm_frames_required:
                self.reset_tracking_only()
                self.current_state = self.NO_ACTION
            return self.current_state

        if self.current_state == self.HAND_APPROACHING:
            self.lost_frames += 1
            if self.lost_frames <= self.lost_tolerance:
                return self.HAND_APPROACHING

        self.reset_tracking_only()
        self.current_state = self.NO_ACTION
        return self.NO_ACTION

    def update_non_open_palm(self):
        """
        A closed/unknown hand is not a High Five.
        Do not turn it into COOLDOWN unless HIGH_FIVE already happened.
        """
        if self.current_state == self.HIGH_FIVE:
            self.current_state = self.COOLDOWN
            return self.COOLDOWN

        if self.current_state == self.COOLDOWN:
            if self.is_in_cooldown():
                return self.COOLDOWN
            self.current_state = self.WAIT_REARM
            self.rearm_count = 0
            return self.WAIT_REARM

        if self.current_state == self.WAIT_REARM:
            self.rearm_count += 1
            if self.rearm_count >= self.rearm_frames_required:
                self.reset_tracking_only()
                self.current_state = self.NO_ACTION
            return self.current_state

        if self.current_state == self.HAND_APPROACHING:
            self.lost_frames += 1
            if self.lost_frames <= self.lost_tolerance:
                return self.HAND_APPROACHING

        self.reset_tracking_only()
        self.current_state = self.NO_ACTION
        return self.NO_ACTION

    # ==========================================================
    # Main detection
    # ==========================================================

    def detect(self, landmarks, gesture):
        # ------------------------------------------------------
        # No landmarks
        # ------------------------------------------------------
        if landmarks is None or len(landmarks) < 21:
            return self.update_no_hand()

        # ------------------------------------------------------
        # Gesture must be OPEN_PALM
        # ------------------------------------------------------
        if gesture != "OPEN_PALM":
            return self.update_non_open_palm()

        # We have a valid open palm again.
        self.lost_frames = 0

        # ------------------------------------------------------
        # Measure current hand
        # ------------------------------------------------------
        palm_size = self.calculate_palm_size(landmarks)
        palm_area = self.calculate_palm_area(landmarks)
        depth = self.calculate_depth(landmarks)

        self.current_palm_size = palm_size
        self.current_palm_area = palm_area
        self.current_depth = depth

        # ------------------------------------------------------
        # COOLDOWN
        # ------------------------------------------------------
        if self.current_state == self.COOLDOWN:
            if self.is_in_cooldown():
                return self.COOLDOWN

            self.current_state = self.WAIT_REARM
            self.rearm_count = 0
            return self.WAIT_REARM

        # ------------------------------------------------------
        # WAIT_REARM
        # ------------------------------------------------------
        if self.current_state == self.WAIT_REARM:
            # The hand must actually move away / become small.
            if self.has_retreated() or self.has_rearmed():
                self.rearm_count += 1
            else:
                self.rearm_count = 0

            if self.rearm_count >= self.rearm_frames_required:
                self.high_five_triggered = False
                self.reset_tracking_only()
                self.current_state = self.NO_ACTION
                return self.NO_ACTION

            return self.WAIT_REARM

        # ------------------------------------------------------
        # Too small to participate
        # ------------------------------------------------------
        if palm_size < self.min_palm_size:
            self.confirm_count = 0
            self.current_state = self.NO_ACTION
            return self.NO_ACTION

        # ------------------------------------------------------
        # Store temporal data
        # ------------------------------------------------------
        self.palm_size_history.append(palm_size)
        self.palm_area_history.append(palm_area)
        self.depth_history.append(depth)

        # ------------------------------------------------------
        # Compute features
        # ------------------------------------------------------
        approaching = self.detect_approaching()

        self.near_camera = palm_size >= self.near_palm_size

        # ------------------------------------------------------
        # HAND_APPROACHING
        # ------------------------------------------------------
        if approaching:
            self.start_approaching()
            self.current_state = self.HAND_APPROACHING

        elif self.current_state == self.HAND_APPROACHING:
            # Keep the state alive through small frame-to-frame noise.
            if self.approach_timeout():
                self.confirm_count = 0
                self.approach_start_time = None
                self.current_state = self.NO_ACTION
            else:
                self.current_state = self.HAND_APPROACHING
        else:
            self.current_state = self.NO_ACTION

        # ------------------------------------------------------
        # HIGH_FIVE candidate
        #
        # This is intentionally stricter than HAND_APPROACHING.
        # A stationary raised hand may be large, but it will not have
        # BOTH clear palm growth AND forward depth movement.
        # ------------------------------------------------------
        high_five_candidate = (
            self.current_state == self.HAND_APPROACHING
            and self.near_camera
            and self.growth_ok
            and self.trend_ok
            and self.depth_ok
        )

        if high_five_candidate:
            self.confirm_count += 1
        else:
            # Decay rather than hard reset to reduce flicker.
            self.confirm_count = max(
                0,
                self.confirm_count - 1
            )

        # ------------------------------------------------------
        # Confirm High Five
        # ------------------------------------------------------
        if self.confirm_count >= self.confirm_frames_required:
            self.last_high_five_time = time.time()
            self.high_five_triggered = True
            self.current_state = self.HIGH_FIVE
            return self.HIGH_FIVE

        self.previous_palm_size = palm_size
        return self.current_state

    # ==========================================================
    # Reset tracking history, but not event cooldown state
    # ==========================================================

    def reset_tracking_only(self):
        self.palm_size_history.clear()
        self.palm_area_history.clear()
        self.depth_history.clear()

        self.current_palm_size = 0.0
        self.current_palm_area = 0.0
        self.current_depth = 0.0

        self.growth_ratio_value = 1.0
        self.fast_growth_ratio_value = 1.0
        self.depth_approach_value = 0.0
        self.approach_score = 0.0

        self.near_camera = False
        self.growth_ok = False
        self.fast_growth_ok = False
        self.depth_ok = False
        self.trend_ok = False

        self.confirm_count = 0
        self.rearm_count = 0
        self.lost_frames = 0
        self.approach_start_time = None
        self.previous_palm_size = 0.0

    def clear_tracking(self):
        self.reset_tracking_only()

    def reset(self):
        self.reset_tracking_only()
        self.current_state = self.NO_ACTION
        self.last_high_five_time = 0.0
        self.high_five_triggered = False

    # ==========================================================
    # Debug information
    # ==========================================================

    def get_debug_info(self):
        return {
            "state": self.current_state,
            "palm_size": self.current_palm_size,
            "palm_area": self.current_palm_area,
            "depth": self.current_depth,
            "depth_approach": self.depth_approach_value,
            "growth_ratio": self.growth_ratio_value,
            "fast_growth_ratio": self.fast_growth_ratio_value,
            "approach_score": self.approach_score,
            "confirm_count": self.confirm_count,
            "confirm_required": self.confirm_frames_required,
            "lost_frames": self.lost_frames,
            "lost_tolerance": self.lost_tolerance,
            "rearm_count": self.rearm_count,
            "rearm_required": self.rearm_frames_required,
            "near_camera": self.near_camera,
            "growth_ok": self.growth_ok,
            "fast_growth_ok": self.fast_growth_ok,
            "depth_ok": self.depth_ok,
            "trend_ok": self.trend_ok,
        }
