from collections import deque
import math
import time

import numpy as np


class ActionDetector:
    """
    PiPi Vision upper-body action detector.

    Supported actions:

        WAVE
        HAND_UP
        HEAD_DOWN
        DAYDREAMING
        NO ACTION
        NOT_IN_SEAT

    DAYDREAMING:
        Uses body + head + hand movement to calculate
        an overall activity level.

        Low activity for a continuous period of time
        is considered DAYDREAMING.
    """

    def __init__(self, daydreaming_seconds=5.0):

        # ============================================================
        # WAVE
        # ============================================================

        self.left_wave_history = deque(maxlen=30)
        self.right_wave_history = deque(maxlen=30)

        self.wave_min_x_range = 0.14
        self.wave_min_dx = 0.012
        self.wave_min_direction_changes = 2

        # ============================================================
        # DAYDREAMING
        # ============================================================

        # Store previous body landmarks.
        self.body_history = deque(maxlen=30)

        # Store previous head position.
        self.head_history = deque(maxlen=30)

        # Store previous hand positions.
        self.left_hand_history = deque(maxlen=30)
        self.right_hand_history = deque(maxlen=30)

        # When low activity started.
        self.low_activity_start_time = None

        # How long the person needs to remain
        # in a low-activity state.
        self.daydreaming_seconds = (
            daydreaming_seconds
        )

        # ------------------------------------------------------------
        # Activity thresholds
        # ------------------------------------------------------------

        # Body movement threshold.
        self.body_movement_threshold = 0.012

        # Head movement threshold.
        self.head_movement_threshold = 0.010

        # Hand movement threshold.
        self.hand_movement_threshold = 0.018

        # A clearly visible movement immediately
        # breaks DAYDREAMING.
        self.active_movement_threshold = 0.025

        # ============================================================
        # HEAD DOWN
        # ============================================================

        self.head_down_history = deque(maxlen=10)

        self.head_down_required_frames = 5

        self.head_down_pitch_threshold = 15.0

        # ============================================================
        # STATE
        # ============================================================

        self.current_action = "NO ACTION"

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def midpoint(p1, p2):

        class Point:
            pass

        point = Point()

        point.x = (
            p1.x + p2.x
        ) / 2.0

        point.y = (
            p1.y + p2.y
        ) / 2.0

        point.z = (
            getattr(p1, "z", 0.0)
            + getattr(p2, "z", 0.0)
        ) / 2.0

        return point

    # ================================================================
    # WAVE
    # ================================================================

    def detect_wave(
        self,
        wrist,
        history
    ):
        """
        Detect horizontal back-and-forth
        wrist movement.

        WAVE does not require the hand to be
        above the shoulder.
        """

        if wrist is None:
            return False

        history.append(
            (
                wrist.x,
                wrist.y
            )
        )

        if len(history) < 8:
            return False

        x_values = [
            point[0]
            for point in history
        ]

        # ------------------------------------------------------------
        # Horizontal movement range
        # ------------------------------------------------------------

        x_range = (
            max(x_values)
            - min(x_values)
        )

        if (
            x_range
            < self.wave_min_x_range
        ):
            return False

        # ------------------------------------------------------------
        # Direction changes
        # ------------------------------------------------------------

        directions = []

        for i in range(
            1,
            len(x_values)
        ):

            dx = (
                x_values[i]
                - x_values[i - 1]
            )

            if (
                abs(dx)
                < self.wave_min_dx
            ):
                continue

            if dx > 0:
                directions.append(1)
            else:
                directions.append(-1)

        if len(directions) < 4:
            return False

        # ------------------------------------------------------------
        # Count direction changes
        # ------------------------------------------------------------

        direction_changes = 0

        for i in range(
            1,
            len(directions)
        ):

            if (
                directions[i]
                != directions[i - 1]
            ):
                direction_changes += 1

        if (
            direction_changes
            < self.wave_min_direction_changes
        ):
            return False

        # Too many direction changes
        # are likely tracking noise.
        if direction_changes > 10:
            return False

        # ------------------------------------------------------------
        # Require both directions
        # ------------------------------------------------------------

        positive_count = (
            directions.count(1)
        )

        negative_count = (
            directions.count(-1)
        )

        if positive_count < 2:
            return False

        if negative_count < 2:
            return False

        return True

    # ================================================================
    # HAND UP
    # ================================================================

    def detect_hand_up(
        self,
        left_wrist,
        right_wrist,
        left_shoulder,
        right_shoulder
    ):
        """
        Any hand above its corresponding shoulder
        counts as HAND_UP.
        """

        if (
            left_wrist is None
            or right_wrist is None
            or left_shoulder is None
            or right_shoulder is None
        ):
            return False

        left_hand_up = (
            left_wrist.y
            < left_shoulder.y
        )

        right_hand_up = (
            right_wrist.y
            < right_shoulder.y
        )

        return (
            left_hand_up
            or right_hand_up
        )

    # ================================================================
    # HEAD POSE
    # ================================================================

    @staticmethod
    def rotation_matrix_to_euler_angles(
        rotation_matrix
    ):
        """
        Convert 3x3 rotation matrix to Euler angles.

        Returns:

            pitch
            yaw
            roll

        in degrees.
        """

        r00 = rotation_matrix[0, 0]
        r10 = rotation_matrix[1, 0]
        r20 = rotation_matrix[2, 0]

        r21 = rotation_matrix[2, 1]
        r22 = rotation_matrix[2, 2]

        r11 = rotation_matrix[1, 1]
        r12 = rotation_matrix[1, 2]

        sy = math.sqrt(
            r00 * r00
            + r10 * r10
        )

        singular = (
            sy < 1e-6
        )

        if not singular:

            pitch = math.atan2(
                r21,
                r22
            )

            yaw = math.atan2(
                -r20,
                sy
            )

            roll = math.atan2(
                r10,
                r00
            )

        else:

            pitch = math.atan2(
                -r12,
                r11
            )

            yaw = math.atan2(
                -r20,
                sy
            )

            roll = 0.0

        return (
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll)
        )

    def get_head_pose(
        self,
        facial_transformation_matrix
    ):
        """
        Extract pitch / yaw / roll from
        Face Landmarker transformation matrix.
        """

        if (
            facial_transformation_matrix
            is None
        ):
            return None

        try:

            matrix = np.asarray(
                facial_transformation_matrix,
                dtype=float
            )

            if matrix.shape == (
                4,
                4
            ):

                rotation_matrix = (
                    matrix[:3, :3]
                )

            elif matrix.shape == (
                3,
                3
            ):

                rotation_matrix = matrix

            else:

                return None

            return (
                self.rotation_matrix_to_euler_angles(
                    rotation_matrix
                )
            )

        except Exception:

            return None

    def get_head_pitch(
        self,
        facial_transformation_matrix
    ):
        """
        Return head pitch in degrees.
        """

        pose = self.get_head_pose(
            facial_transformation_matrix
        )

        if pose is None:
            return None

        pitch, _, _ = pose

        return pitch

    def detect_head_down(
        self,
        facial_transformation_matrix
    ):
        """
        Detect HEAD_DOWN from head pitch.
        """

        pitch = self.get_head_pitch(
            facial_transformation_matrix
        )

        if pitch is None:

            self.head_down_history.append(
                False
            )

            return False

        is_head_down = (
            pitch
            > self.head_down_pitch_threshold
        )

        self.head_down_history.append(
            is_head_down
        )

        if (
            len(
                self.head_down_history
            )
            < self.head_down_required_frames
        ):
            return False

        recent = list(
            self.head_down_history
        )[
            -self.head_down_required_frames:
        ]

        return all(recent)

    # ================================================================
    # BODY MOVEMENT
    # ================================================================

    def calculate_body_movement(
        self,
        pose_landmarks
    ):
        """
        Calculate movement of major body landmarks.

        Used for DAYDREAMING activity detection.
        """

        if not pose_landmarks:
            return 0.0

        landmark_indices = [
            0,      # nose
            11,     # left shoulder
            12,     # right shoulder
            13,     # left elbow
            14,     # right elbow
            15,     # left wrist
            16,     # right wrist
            23,     # left hip
            24      # right hip
        ]

        current_points = []

        for index in landmark_indices:

            if (
                index
                >= len(pose_landmarks)
            ):
                continue

            landmark = (
                pose_landmarks[index]
            )

            current_points.append(
                (
                    landmark.x,
                    landmark.y
                )
            )

        if not current_points:
            return 0.0

        if not self.body_history:

            self.body_history.append(
                current_points
            )

            return 0.0

        previous_points = (
            self.body_history[-1]
        )

        count = min(
            len(current_points),
            len(previous_points)
        )

        if count == 0:

            self.body_history.append(
                current_points
            )

            return 0.0

        total_movement = 0.0

        for i in range(count):

            dx = (
                current_points[i][0]
                - previous_points[i][0]
            )

            dy = (
                current_points[i][1]
                - previous_points[i][1]
            )

            movement = math.sqrt(
                dx * dx
                + dy * dy
            )

            total_movement += movement

        average_movement = (
            total_movement
            / count
        )

        self.body_history.append(
            current_points
        )

        return average_movement

    # ================================================================
    # HEAD MOVEMENT
    # ================================================================

    def calculate_head_movement(
        self,
        pose_landmarks
    ):
        """
        Calculate movement of the nose.

        This provides a simple estimate of
        head movement.
        """

        if (
            not pose_landmarks
            or len(pose_landmarks) <= 0
        ):
            return 0.0

        nose = pose_landmarks[0]

        current_point = (
            nose.x,
            nose.y
        )

        if not self.head_history:

            self.head_history.append(
                current_point
            )

            return 0.0

        previous_point = (
            self.head_history[-1]
        )

        dx = (
            current_point[0]
            - previous_point[0]
        )

        dy = (
            current_point[1]
            - previous_point[1]
        )

        movement = math.sqrt(
            dx * dx
            + dy * dy
        )

        self.head_history.append(
            current_point
        )

        return movement

    # ================================================================
    # HAND MOVEMENT
    # ================================================================

    @staticmethod
    def calculate_point_movement(
        point,
        history
    ):
        """
        Calculate movement of a single point.
        """

        if point is None:

            return 0.0

        current_point = (
            point.x,
            point.y
        )

        if not history:

            history.append(
                current_point
            )

            return 0.0

        previous_point = (
            history[-1]
        )

        dx = (
            current_point[0]
            - previous_point[0]
        )

        dy = (
            current_point[1]
            - previous_point[1]
        )

        movement = math.sqrt(
            dx * dx
            + dy * dy
        )

        history.append(
            current_point
        )

        return movement

    # ================================================================
    # ACTIVITY SCORE
    # ================================================================

    def calculate_activity(
        self,
        pose_landmarks
    ):
        """
        Calculate overall activity from:

            Body
            Head
            Hands

        The score is used only for DAYDREAMING.
        """

        if not pose_landmarks:

            return 0.0

        # ------------------------------------------------------------
        # Body movement
        # ------------------------------------------------------------

        body_movement = (
            self.calculate_body_movement(
                pose_landmarks
            )
        )

        # ------------------------------------------------------------
        # Head movement
        # ------------------------------------------------------------

        head_movement = (
            self.calculate_head_movement(
                pose_landmarks
            )
        )

        # ------------------------------------------------------------
        # Hand movement
        # ------------------------------------------------------------

        left_wrist = None
        right_wrist = None

        if len(pose_landmarks) > 15:

            left_wrist = (
                pose_landmarks[15]
            )

        if len(pose_landmarks) > 16:

            right_wrist = (
                pose_landmarks[16]
            )

        left_hand_movement = (
            self.calculate_point_movement(
                left_wrist,
                self.left_hand_history
            )
        )

        right_hand_movement = (
            self.calculate_point_movement(
                right_wrist,
                self.right_hand_history
            )
        )

        # ------------------------------------------------------------
        # Combine activity
        # ------------------------------------------------------------

        activity_score = max(
            body_movement,
            head_movement,
            left_hand_movement,
            right_hand_movement
        )

        return activity_score

    # ================================================================
    # DAYDREAMING
    # ================================================================

    def detect_daydreaming(
        self,
        pose_landmarks
    ):
        """
        Detect DAYDREAMING.

        Logic:

            1. Calculate body activity.
            2. Calculate head activity.
            3. Calculate hand activity.
            4. Use the highest movement as activity score.
            5. If activity remains low continuously
               for daydreaming_seconds -> DAYDREAMING.
            6. Any obvious movement immediately resets
               the low-activity timer.
        """

        activity = (
            self.calculate_activity(
                pose_landmarks
            )
        )

        # ------------------------------------------------------------
        # Obvious movement
        # ------------------------------------------------------------

        if (
            activity
            > self.active_movement_threshold
        ):

            self.low_activity_start_time = (
                None
            )

            return False

        # ------------------------------------------------------------
        # Low activity
        # ------------------------------------------------------------

        if (
            self.low_activity_start_time
            is None
        ):

            self.low_activity_start_time = (
                time.time()
            )

            return False

        # ------------------------------------------------------------
        # Calculate duration
        # ------------------------------------------------------------

        low_activity_duration = (
            time.time()
            - self.low_activity_start_time
        )

        if (
            low_activity_duration
            >= self.daydreaming_seconds
        ):

            return True

        return False

    # ================================================================
    # RAW ACTION
    # ================================================================

    def detect_action_raw(
        self,
        pose_landmarks,
        facial_transformation_matrix=None
    ):

        if not pose_landmarks:

            return "NOT_IN_SEAT"

        try:

            left_shoulder = (
                pose_landmarks[11]
            )

            right_shoulder = (
                pose_landmarks[12]
            )

            left_wrist = (
                pose_landmarks[15]
            )

            right_wrist = (
                pose_landmarks[16]
            )

        except IndexError:

            return "NO ACTION"

        # ============================================================
        # WAVE
        # ============================================================

        wave_left = self.detect_wave(
            left_wrist,
            self.left_wave_history
        )

        wave_right = self.detect_wave(
            right_wrist,
            self.right_wave_history
        )

        if (
            wave_left
            or wave_right
        ):

            return "WAVE"

        # ============================================================
        # HAND UP
        # ============================================================

        hand_up = self.detect_hand_up(
            left_wrist,
            right_wrist,
            left_shoulder,
            right_shoulder
        )

        if hand_up:

            return "HAND_UP"

        # ============================================================
        # HEAD DOWN
        # ============================================================

        head_down = (
            self.detect_head_down(
                facial_transformation_matrix
            )
        )

        if head_down:

            return "HEAD_DOWN"

        # ============================================================
        # DAYDREAMING
        # ============================================================

        daydreaming = (
            self.detect_daydreaming(
                pose_landmarks
            )
        )

        if daydreaming:

            return "DAYDREAMING"

        # ============================================================
        # DEFAULT
        # ============================================================

        return "NO ACTION"

    # ================================================================
    # PUBLIC API
    # ================================================================

    def detect(
        self,
        pose_landmarks,
        facial_transformation_matrix=None
    ):
        """
        Main detection interface.
        """

        if not pose_landmarks:

            self.reset()

            return "NOT_IN_SEAT"

        action = (
            self.detect_action_raw(
                pose_landmarks,
                facial_transformation_matrix
            )
        )

        self.current_action = action

        return action

    # ================================================================
    # RESET
    # ================================================================

    def reset(self):

        self.left_wave_history.clear()

        self.right_wave_history.clear()

        self.body_history.clear()

        self.head_history.clear()

        self.left_hand_history.clear()

        self.right_hand_history.clear()

        self.head_down_history.clear()

        self.low_activity_start_time = (
            None
        )

        self.current_action = (
            "NO ACTION"
        )