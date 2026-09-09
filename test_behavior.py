from vision.behavior_state import BehaviorState


behavior = BehaviorState()


tests = [
    ("WAVE", "OPEN_PALM"),
    ("WAVE", "NO GESTURE"),
    ("HAND_UP", "OPEN_PALM"),
    ("HAND_UP", "NO GESTURE"),
    ("NO ACTION", "THUMBS_UP"),
    ("HEAD_DOWN", "OPEN_PALM"),
    ("DAYDREAMING", "NO GESTURE"),
    ("NOT_IN_SEAT", "NO GESTURE"),
]


for pose_action, hand_gesture in tests:

    behavior.update(
        emotion="HAPPINESS",
        pose_action=pose_action,
        hand_gesture=hand_gesture
    )

    print(
        f"Pose={pose_action:<15} "
        f"Hand={hand_gesture:<12} "
        f"→ {behavior.final_behavior:<15} "
        f"| {behavior.description}"
    )