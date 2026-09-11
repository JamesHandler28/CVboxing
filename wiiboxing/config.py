"""
All tunable constants for pose tracking, motion detection, and punch
classification live here. Nothing in the rest of the package should hardcode
a threshold -- import it from this module instead, so retuning after a
retrain only ever touches one file.

## Retraining

The classifier thresholds below (HOOK_ELBOW_NORM_THRESHOLD and friends) were
fit by running `scripts/punch_data_collector.py` to collect labeled reps into
a CSV, then fitting a small decision tree per arm (see
`docs/classifier_notes.md` for the exact fitting steps). If detection
accuracy drifts -- new camera, new lighting, a different person using it --
re-run the collector and refit; don't hand-tune these numbers directly
without data backing the change.
"""

# ---------------------------------------------------------------------------
# Motion detection (ArmMotionTracker): how "activated" a frame must be to
# count as mid-punch rather than guard/idle.
#
# Each arm tracks its OWN resting baseline live (an exponential moving
# average, updated only while at rest) and triggers off DEVIATION from that
# baseline, rather than a fixed absolute number. This matters because a
# fixed threshold silently assumes everyone's guard posture matches whatever
# stance was used when the threshold was picked -- it isn't, in general
# (confirmed by combo testing: resting elbow position varied enough between
# sessions that a fixed "off" threshold could never trigger for one stance).
# ---------------------------------------------------------------------------
BASELINE_EMA_ALPHA = 0.06  # how fast the resting baseline adapts, per frame, while at rest

MARGIN_ON_ANGLE = 20  # degrees above baseline to trigger "punch starting"
MARGIN_OFF_ANGLE = 8  # degrees above baseline that still counts as "still active"
MARGIN_ON_ELBOW = 0.12  # elbow_dx_norm above baseline to trigger "punch starting"
MARGIN_OFF_ELBOW = 0.05
MARGIN_ON_DY = 0.20  # dy_norm below baseline (wrist rising) to trigger "punch starting"
MARGIN_OFF_DY = 0.15  # looser than angle/elbow -- after a strong punch, vertical wrist
# position is the slowest signal to settle back near baseline (follow-through/fatigue)

MAX_BUFFER_FRAMES = 25  # safety cap (~0.8-1.1s depending on camera fps) so a buffer can't grow forever
REARM_NEUTRAL_FRAMES = 2  # consecutive non-active frames required to consider a punch over

# ---------------------------------------------------------------------------
# Classifier thresholds -- RIGHT arm. Fit from RIGHT-labeled rows in
# punch_training_data.csv via a depth-3 decision tree (scikit-learn),
# scoring 23/24 on held-out training reps.
# ---------------------------------------------------------------------------
HOOK_ELBOW_NORM_THRESHOLD = 0.57
JAB_STRAIGHTNESS_THRESHOLD = 0.72
UPPERCUT_DY_NORM_THRESHOLD = 0.11

# ---------------------------------------------------------------------------
# Classifier thresholds -- LEFT arm. Fit SEPARATELY from LEFT-labeled rows.
# Do not reuse the right-arm thresholds for the left arm: stance asymmetry
# means the left arm doesn't look like a mirror image of the right to the
# camera. Applying the right-arm rule to real left-arm data only scored
# 15/24; this separately-fit rule scores 23/24.
# ---------------------------------------------------------------------------
LEFT_JAB_ELBOW_NORM_THRESHOLD = 0.36
LEFT_UPPERCUT_DY_NORM_THRESHOLD = -0.91
LEFT_UPPERCUT_ANGLE_THRESHOLD = 111.5

# ---------------------------------------------------------------------------
# Height zone (HEAD / BODY / LOW), relative to nose and hip landmarks.
# ---------------------------------------------------------------------------
HEAD_ZONE_MARGIN = 0.1  # fraction of shoulder width of slack below the nose that still counts as HEAD

# ---------------------------------------------------------------------------
# Target practice game.
# ---------------------------------------------------------------------------
TARGET_RADIUS_FACTOR = 0.7  # target radius, as a multiple of shoulder width (in pixels)
TARGET_X_SPREAD_FACTOR = 1.4  # how far left/right of body midline a target can spawn (x shoulder width)
TARGET_HIT_FLASH_SECONDS = 0.35
TARGET_MISS_FLASH_SECONDS = 0.35

# ---------------------------------------------------------------------------
# Duel mode: two players, each with HP. Your punch connects on the OTHER
# player and damages them (a real opponent, not a self-miss abstraction),
# unless they block it (their own glove is covering the zone you aimed at
# -- see GLOVE_BLOCK_COVERAGE below) or dodge it (head shots only, via
# leaning far enough). Hit zero HP and you're knocked down -- you get a
# time-limited window to punch a floating recovery target back up, with
# your returned HP scaling with how well you do (see
# DUEL_RECOVERY_MIN_HP_FRACTION below), at the cost of a lower max-HP
# ceiling each time. Fail a recovery window, or reach your MAX_KNOCKDOWNS'th
# knockdown (no recovery chance at all), and you lose the match.
# ---------------------------------------------------------------------------
DUEL_MAX_HP = 100
DUEL_HEAD_DAMAGE = 30
DUEL_BODY_DAMAGE = 18
DUEL_LOW_DAMAGE = 10
DUEL_HP_REDUCTION_PER_KNOCKDOWN = 20  # max HP ceiling lost each successful recovery
DUEL_MAX_HP_FLOOR = 20             # max HP never drops below this

KNOCKDOWN_TIME_LIMIT_SECONDS = 10.0  # time to hit enough targets to get back up
KNOCKDOWN_BASE_REQUIRED_HITS = 3   # hits needed on your first knockdown
KNOCKDOWN_REQUIRED_HITS_INCREMENT = 1  # extra hits required each subsequent knockdown

MAX_KNOCKDOWNS = 3  # the Nth knockdown is an automatic KO -- no recovery
# minigame at all, matching a real three-knockdown-rule finish.

# ---------------------------------------------------------------------------
# Recovery HP scaling: landing exactly `required_hits` gets you back up with
# only a fraction of your (reduced) new max HP. Every extra hit beyond that
# scales your returned HP up continuously, capping out at full new-max-HP
# once you've landed DUEL_RECOVERY_EXTRA_HITS_FOR_FULL_HP hits past the
# requirement -- so recovery performance actually matters, not just clearing
# the bar.
# ---------------------------------------------------------------------------
DUEL_RECOVERY_MIN_HP_FRACTION = 0.4
DUEL_RECOVERY_EXTRA_HITS_FOR_FULL_HP = 5

# ---------------------------------------------------------------------------
# Blocking: a punch aimed at a zone is blocked outright if the DEFENDER's
# own wrist is currently positioned to cover that zone. Boxing gloves are
# bulkier than a bare fist, so the effective coverage extends a bit past
# the wrist's exact tracked position -- see glove_covers_zone() in
# pose_utils.py. Expressed as a fraction of shoulder width, added on both
# sides of each zone boundary.
# ---------------------------------------------------------------------------
GLOVE_BLOCK_COVERAGE = 0.35

# ---------------------------------------------------------------------------
# Hit feedback: a brief impact flash + screen shake when a punch actually
# lands (not blocked or dodged). Shake magnitude scales with the zone hit
# (a head shot should feel like more of a jolt than a low shot). Impact
# position is approximate -- a fixed point in the defender's half per zone,
# not the exact tracked contact point, which keeps this simple and in
# keeping with the rest of the shape-based rendering.
# ---------------------------------------------------------------------------
HIT_FLASH_DURATION_SECONDS = 0.25
HIT_SHAKE_DURATION_SECONDS = 0.15
HIT_SHAKE_MAGNITUDE_PX = {"HEAD": 12, "BODY": 7, "LOW": 4}
IMPACT_POINT_Y_FRAC = {"HEAD": 0.30, "BODY": 0.55, "LOW": 0.75}

# ---------------------------------------------------------------------------
# Dodging: leaning shifts your on-screen position relative to your own
# calibrated neutral center. Lean far enough (normalized by your own
# shoulder width, so it scales with distance from camera) and head shots
# aimed at you miss entirely.
# ---------------------------------------------------------------------------
LEAN_DODGE_THRESHOLD = 0.6   # |lean_offset| beyond this dodges a head shot
LEAN_RENDER_SCALE = 1.0      # how many pixels of on-screen shift per 1.0 of lean_offset, times half-width fraction

# ---------------------------------------------------------------------------
# Pre-fight calibration: checks each player is visible, roughly centered in
# their half of the frame, and at a workable distance from the camera
# (measured via shoulder width in pixels) before the fight starts.
# ---------------------------------------------------------------------------
CALIBRATION_HOLD_SECONDS = 1.5          # how long correct position must be held
CALIBRATION_CENTER_TOLERANCE = 0.18     # fraction of half-width the midline can be off-center by
CALIBRATION_SHOULDER_WIDTH_MIN_FRAC = 0.10  # shoulder width (fraction of half-width) below this = too far away
CALIBRATION_SHOULDER_WIDTH_MAX_FRAC = 0.38  # shoulder width (fraction of half-width) above this = too close

# ---------------------------------------------------------------------------
# MediaPipe pose model.
# ---------------------------------------------------------------------------
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
VISIBILITY_THRESHOLD = 0.6  # per-landmark confidence below which an arm is treated as "not visible"
# this frame -- prevents a noisy off-screen/occluded position estimate from
# being fed into motion detection as if it were a real, trustworthy reading

DEBUG_OVERLAY = True