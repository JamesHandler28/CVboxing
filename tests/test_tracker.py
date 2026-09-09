"""Tests for ArmMotionTracker: baseline calibration, activation, and firing."""

from wiiboxing.tracker import ArmMotionTracker


def _dummy_frame_data(angle):
    # Only `angle` matters for these tests -- the rest just need to be
    # present since summarize_window reads them.
    return {
        "angle": angle,
        "wrist_x": 0.0, "wrist_y": 0.0,
        "shoulder_x": 0.0, "shoulder_y": 0.0,
        "elbow_x": 0.0, "elbow_y": 0.0,
        "midline_x": 0.0,
        "shoulder_w": 0.2,
    }


def test_tracker_does_not_fire_while_at_rest():
    tracker = ArmMotionTracker(classify_fn=lambda s: "JAB")
    now = 0.0
    for _ in range(20):
        now += 1 / 30
        fired = tracker.update(_dummy_frame_data(15.0), angle=15.0, elbow_dx_norm=0.2, dy_norm=0.3, now=now)
        assert fired is None


def test_tracker_fires_once_after_a_punch_and_return_to_rest():
    tracker = ArmMotionTracker(classify_fn=lambda s: "JAB")
    now = 0.0

    # Establish a resting baseline first.
    for _ in range(10):
        now += 1 / 30
        tracker.update(_dummy_frame_data(15.0), angle=15.0, elbow_dx_norm=0.2, dy_norm=0.3, now=now)

    fires = []
    # Punch: angle rises well above baseline for a few frames...
    for angle in (60, 120, 170, 150, 80):
        now += 1 / 30
        f = tracker.update(_dummy_frame_data(angle), angle=angle, elbow_dx_norm=0.2, dy_norm=0.3, now=now)
        if f:
            fires.append(f)

    # ...then returns to rest, which should trigger exactly one fire.
    for _ in range(5):
        now += 1 / 30
        f = tracker.update(_dummy_frame_data(15.0), angle=15.0, elbow_dx_norm=0.2, dy_norm=0.3, now=now)
        if f:
            fires.append(f)

    assert len(fires) == 1
    label, summary = fires[0]
    assert label == "JAB"
    assert summary["peak_angle"] == 170


def test_tracker_baseline_adapts_to_a_higher_resting_elbow_position():
    # A resting elbow position that would have tripped a fixed threshold
    # should NOT trigger activation once the tracker has calibrated to it.
    tracker = ArmMotionTracker(classify_fn=lambda s: "JAB")
    now = 0.0
    for _ in range(30):
        now += 1 / 30
        fired = tracker.update(_dummy_frame_data(10.0), angle=10.0, elbow_dx_norm=0.30, dy_norm=0.3, now=now)
        assert fired is None
    assert tracker.baseline_elbow is not None
    assert abs(tracker.baseline_elbow - 0.30) < 0.02
