# Classifier design notes

This documents *why* the classifier looks the way it does, for future-you
(or anyone else) wondering why it isn't a simpler angle/position check.

## Attempt 1: per-frame position/angle thresholds

The first version checked, on every single frame: is the elbow angle above
some threshold (straight punch)? Does the wrist cross the body midline
(hook)? Is the wrist above the shoulder by some margin (uppercut)? This
mostly failed:

- A jab and a hook, viewed by a single front-facing camera, can produce
  very similar single-frame wrist positions/angles depending on the
  person's exact technique and camera angle -- there's no reliable
  per-frame geometric signal that separates them for every stance.
- Fixed thresholds assumed a resting "guard" position that didn't hold
  across sessions/stances. In combo testing specifically, one person's
  resting elbow position sat *already past* the fixed "not punching"
  threshold, meaning the motion detector could never register a return to
  rest.

## Attempt 2: windowed features + a real training set

The fix was two-fold:

1. **Stop classifying single frames.** Instead, detect when an arm starts
   and stops moving (a whole punch's motion), buffer every frame of that
   window, then compute summary features over the *whole window*: peak
   elbow angle, peak elbow flare distance from the shoulder
   (`elbow_dx_norm`), how far the wrist rose (`dy_min_norm`), and how
   straight vs. curved the wrist's path was (`straightness`: net
   displacement / total path length).

2. **Stop hand-picking thresholds.** A small labeled dataset was collected
   with `scripts/punch_data_collector.py` (which auto-cycles a countdown /
   throw / rest sequence so you can label a batch of reps hands-free), then
   a depth-3 `sklearn.tree.DecisionTreeClassifier` was fit per arm. The
   resulting splits became the thresholds in `wiiboxing/config.py`.

`elbow_dx_norm` (how far the elbow flares sideways from the shoulder,
normalized by shoulder width) turned out to be the strongest single signal
for separating a hook from a jab -- much more reliable than wrist position,
which is what all the earlier per-frame attempts were built around.

## Left arm needed its own fit

Applying the right-arm-fit thresholds to left-arm data scored 15/24 (62%).
Stance means the two arms don't look like mirror images of each other to a
single front-facing camera -- visibility, apparent range of motion, and
baseline resting posture all differ. A separate decision tree fit on
left-labeled data alone scored 23/24. **Don't assume left/right symmetry
without checking it against real data for your setup.**

## Motion detection: self-calibrating baseline, not a fixed number

`ArmMotionTracker` doesn't use a fixed "is this arm resting or punching"
threshold. It maintains an exponential moving average of each arm's resting
angle/elbow-flare/wrist-height, updated only while the arm is judged to be
at rest, and triggers activation based on *deviation from that baseline*.

This was a direct fix for a bug found via combo testing: a fixed "resting"
threshold, picked from one person's stance in one session, did not match
that same person's resting elbow position during rapid combo throwing --
the arm's true rest state sat *already past* the fixed threshold, so the
tracker's motion window could never close. Self-calibrating per-arm
baselines fix this by construction; there's no fixed number to go stale.

One asymmetry worth knowing about: after a strong punch (especially an
uppercut), vertical wrist position (`dy`) is the slowest of the three
signals (angle, elbow flare, dy) to settle back near baseline --
follow-through and fatigue keep the arm lower for a bit. The "off" margin
for `dy` is deliberately looser than for angle/elbow for this reason (see
`MARGIN_OFF_DY` in `config.py`). `MAX_BUFFER_FRAMES` is a safety net for the
rare case where a signal doesn't settle within a reasonable window at all --
it forces a classification rather than buffering forever.
