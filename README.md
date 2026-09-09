# wiiboxing

Webcam boxing move detection using MediaPipe pose estimation. Detects jabs,
hooks, and uppercuts (both arms), classifies which zone they land in
(head/body/low), and includes a target-practice mode with an on-screen ball
to punch at.

## Install

```bash
pip install -e ".[dev]"
```

Or, without installing the package:

```bash
pip install -r requirements.txt
```

## Run

```bash
wiiboxing-detect   # console-only punch detection with webcam overlay
wiiboxing-play     # target-practice game: punch the ball
wiiboxing-duel     # 1v1: two players side by side, HP, knockdowns, recovery windows
```

If you didn't install the package, run the modules directly instead:

```bash
python -m wiiboxing.detector
python -m wiiboxing.game
python -m wiiboxing.duel
```

## Duel mode

`wiiboxing-duel` (or `python -m wiiboxing.duel`) is a first-person 1v1 boxing
game:

- **Menu:** press `1` for solo practice against a stationary dummy, `2` for
  a 2-player duel.
- **Calibration:** each player must be visible, roughly centered in their
  half of the frame, and at a workable distance from the camera (checked
  via shoulder width) -- hold that position for a couple seconds and the
  fight starts. On-screen text tells you which way to move if you're not
  in position yet.
- **Fight:** each player sees their own hands (simple circles at their
  tracked wrist positions) and the opponent's silhouette. Leaning shifts
  your on-screen position -- and it's mirrored, so leaning your real right
  shifts you *left* from the opponent's point of view, since you're facing
  each other. Landing a punch (classified jab/hook/uppercut, HEAD/BODY/LOW
  zone, same classifier as freeplay mode) damages the opponent -- unless
  it's a head shot and they leaned far enough to dodge it. Body and low
  shots always land.
- **Knockdown:** hit zero HP and you're knocked down. You get 10 seconds to
  land enough hits on a floating recovery target (reusing the freeplay
  target game) to get back up -- but with lower max HP than before, and the
  next knockdown needs more recovery hits than the last. Fail a recovery
  window and you lose the match.

Two important caveats: MediaPipe's classic Pose model only tracks one
person at a time, so 2-player mode runs two separate Pose instances (one
per half of the frame) -- meaning it does twice the per-frame work of
solo/freeplay modes, and each player needs to stay roughly within their
half. And there's no way to detect real physical contact between two
people from a single webcam, so all hit detection here is the abstract
zone+dodge model above, not literal pixel collision between the two of you.

## How it works

Punches are classified using **windowed features** (peak/min values over an
entire punch's motion), not single-frame checks -- see
`docs/classifier_notes.md` for why that approach was necessary. Each arm's
`ArmMotionTracker` (in `wiiboxing/tracker.py`) maintains a live,
self-calibrating baseline of that arm's resting position, and detects a
punch starting/ending based on deviation from that baseline rather than a
fixed number -- this matters because a fixed threshold silently assumes your
stance matches whatever stance was used when the threshold was picked.

Once a punch's motion window closes, `wiiboxing/classifier.py` extracts a
handful of features (elbow flare distance, how far the wrist rose, how
straight vs. curved the punch's path was) and runs them through a small
decision-tree rule -- one for the right arm, a **separately fit** one for
the left arm, since stance asymmetry means the two sides don't look like
mirror images of each other to the camera.

All the tunable numbers -- motion-detection margins, classifier thresholds,
game constants -- live in one place: `wiiboxing/config.py`.

## Project layout

```
wiiboxing/            the package
  config.py           every tunable constant, with notes on where each came from
  pose_utils.py       angle calculation, height-zone classification
  classifier.py       windowed feature extraction + the punch classifiers
  tracker.py          per-arm motion detection (ArmMotionTracker)
  landmarks.py         per-frame landmark extraction shared by detector.py and game.py
  detector.py         console-only detection mode
  game.py             target-practice game mode (freeplay)
  targets.py          target spawning/hit-testing, shared by game.py and duel.py
  calibration.py      pre-fight position checks (visible, centered, right distance)
  duel_state.py       pure per-player HP/knockdown/dodge state machine for duel mode
  ring_render.py      simple shape-based rendering (opponent silhouette, own hands, ring)
  duel.py             1v1 duel mode: menu, calibration, first-person fight
  cli.py              entry points for the console scripts above

scripts/
  punch_data_collector.py   records labeled reps (single-punch and combo) to CSV for retraining

data/                 CSVs of labeled training reps (see below)
tests/                pytest suite -- classifier logic + a regression test against data/punch_training_data.csv
```

## Retraining the classifier

The classifier thresholds are only as good as the data they were fit from.
If accuracy drifts (new camera, new lighting, a different person, a new
punch type):

1. Run `python scripts/punch_data_collector.py`. It walks you through
   labeled batches (press a key, it auto-cycles countdown -> throw -> rest
   for a set number of reps) for jab/hook/uppercut on each arm, plus a
   combo mode that records full raw traces for debugging.
2. This appends to `punch_training_data.csv` (and `combo_raw_data.csv` for
   raw combo traces). Move/merge these into `data/`.
3. Fit a small decision tree per arm against the normalized features
   (`elbow_dx_norm`, `dy_min_norm`, `straightness`, `peak_angle`) and update
   the thresholds in `wiiboxing/config.py` accordingly.
4. Run `pytest` -- the regression test in `tests/test_classifier.py` will
   tell you if the new thresholds hold up against the existing labeled data.

## Requirements

- Python 3.9+
- A webcam
- opencv-python, mediapipe, numpy (installed automatically via `pip install -e .`)
