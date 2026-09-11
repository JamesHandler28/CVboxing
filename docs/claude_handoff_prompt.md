# Claude Handoff Prompt

Copy everything between the markers below into a new Claude conversation when working on this repository.

---

You are working as a repository-aware Python and computer-vision maintainer on the `cvboxing` project. Read the repository before making changes. Treat the descriptions below as orientation, then verify details against the current files and tests because the code is authoritative.

## Project

`cvboxing` is a webcam boxing application built with Python, OpenCV, and MediaPipe Pose. It detects `JAB`, `HOOK`, and `UPPERCUT` punches from either arm, classifies punch height as `HEAD`, `BODY`, or `LOW`, and provides:

- Webcam punch detection with a pose overlay.
- Target-practice mode.
- Solo dummy practice.
- Two-player split-screen duel mode with HP, knockdowns, recovery targets, and head-shot dodge behavior.
- A labeled punch-data collector for classifier tuning.

The project is interactive and camera-dependent, but most motion, classification, calibration, target, and duel-state logic is unit-testable without a camera.

## Repository map

### Package: `wiiboxing/`

- `config.py`: Central configuration for motion tracking, classifier thresholds, target behavior, duel rules, calibration, rendering, and MediaPipe confidence values. Keep tunable behavior centralized here.
- `pose_utils.py`: Dependency-light geometry helpers, including elbow-angle and height-zone calculations.
- `landmarks.py`: Converts one MediaPipe Pose result into the `FrameLandmarks` representation used by the rest of the application.
- `classifier.py`: Converts buffered punch-frame summaries into arm-specific punch labels. Contains `summarize_window()`, `classify_from_summary_right()`, and `classify_from_summary_left()`.
- `tracker.py`: Defines `ArmMotionTracker`, which learns a rest baseline, detects motion activation/deactivation, buffers frames, and emits classified punch events.
- `targets.py`: Defines `Target`, target spawning, random target placement, and wrist-to-target hit testing.
- `calibration.py`: Pure calibration checks for visibility, body distance, and centering.
- `duel_state.py`: Defines `PlayerState` and the HP, knockdown, recovery, timeout, damage, and dodge state logic.
- `ring_render.py`: OpenCV shape-based rendering for the ring, opponent, limbs, gloves, and hands.
- `detector.py`: Webcam-only punch detector and console output loop.
- `game.py`: Freeplay target-practice loop.
- `duel.py`: Menu, calibration, solo/duel runtime, pose processing, rendering, and punch-event resolution. It also defines live `PlayerRuntime` data.
- `cli.py`: Console-script wrappers for the detector, practice, and duel modes.
- `__init__.py`: Package metadata; current version is `0.1.0`.

### Other directories and files

- `scripts/punch_data_collector.py`: Interactive labeled data capture. It captures eight repetitions per batch, supports labels `1` through `6` for right/left jab, hook, and uppercut, and `7` for right-left combos.
- `data/punch_training_data.csv`: Window-summary training data used for classifier regression checks and threshold fitting.
- `data/combo_raw_data.csv`: Frame-level traces from combo experiments.
- `docs/classifier_notes.md`: Notes on fitting classifier thresholds from collected data.
- `tests/test_classifier.py`: Classifier geometry, decision branches, and training-data regression checks.
- `tests/test_tracker.py`: Tracker activation, firing, idle behavior, and baseline adaptation.
- `tests/test_duel_state.py`: HP, knockdowns, recovery, timeout, reset, damage ordering, and dodge behavior.
- `tests/test_calibration.py`: Visibility, distance, centering, and zero-width calibration behavior.
- `README.md`: User-facing installation, usage, gameplay, and data-collection documentation.
- `pyproject.toml`: Package metadata, dependencies, and console entry points.
- `requirements.txt`: Runtime dependency declarations.

## Runtime architecture

The main runtime pipeline is:

1. Open webcam index `0`.
2. Flip the camera frame horizontally for selfie-style interaction.
3. Run MediaPipe Pose.
4. Convert the result with `landmarks.extract()` into `FrameLandmarks`.
5. Build/update one `ArmMotionTracker` per visible arm.
6. Summarize a completed motion window.
7. Classify it with the appropriate right- or left-arm classifier.
8. Classify the final wrist height as `HEAD`, `BODY`, or `LOW`.
9. Send the resulting event to detector output, target-practice logic, or duel-state logic.
10. Render the camera/pose/game UI with OpenCV.

The detector, practice, and duel loops reuse the same lower-level concepts but have different event-resolution behavior. Do not assume that changing one loop automatically changes the others.

## Coordinate and labeling conventions

These conventions are behaviorally important:

- Camera frames are horizontally mirrored before display and processing.
- Because of the mirrored interaction, `landmarks.py` swaps MediaPipe left/right landmark assignments so internal labels refer to the user's actual arms.
- Landmark positions and most motion features are normalized coordinates, generally relative to the relevant frame or shoulder width.
- Target positions and hit-testing use pixel coordinates in the rendered game area.
- Duel mode crops the camera into player half-frames and runs a Pose model for each half. Rendering and landmark coordinates must remain consistent with the relevant half-frame.
- The two arms have separate classifier rules and thresholds. Do not make them symmetric without checking the fitted data and tests.
- In target practice, a visible wrist can contact a target immediately; this is not necessarily the same moment as a completed classified punch.

## Core abstractions

### `FrameLandmarks`

Defined in `wiiboxing/landmarks.py`. It contains the relevant MediaPipe landmark objects, hip height, elbow angles, body midline, shoulder width, normalized elbow/wrist measurements, visibility flags, and prebuilt per-arm frame data. `extract()` is the main bridge from MediaPipe to the application.

### `ArmMotionTracker`

Defined in `wiiboxing/tracker.py`. It maintains an exponential moving-average rest baseline. A punch becomes active when angle, elbow flare, or wrist-height changes exceed configured margins. The tracker buffers the active motion, waits for neutral frames or a maximum buffer length, calls `summarize_window()`, and invokes an injected classifier. Each arm has its own tracker and classifier function.

### Classifier summaries

`classifier.summarize_window()` computes features such as peak angle, elbow displacement, wrist vertical movement, shoulder displacement, path straightness, and peak wrist coordinates. The right-arm classifier checks hook-like elbow flare, jab-like straightness, and uppercut-like upward motion in a specific order. The left-arm classifier uses a separately fitted rule set. Preserve this arm-specific behavior unless new data justifies a change.

### `PlayerState` and `PlayerRuntime`

`PlayerState` in `duel_state.py` owns logical HP and match state: `PLAYING`, `KNOCKDOWN`, and `GAME_OVER`. Its methods include `take_damage()`, `enter_knockdown()`, `register_recovery_hit()`, `tick()`, and `reset()`.

`PlayerRuntime` in `duel.py` owns live camera/session data such as arm trackers, recovery target, lean baseline, and calibration hold timing. Keep camera/runtime concerns separate from reusable state logic.

### `Target`

Defined in `targets.py`. It represents a target's position and zone and supports spawning and wrist-contact checks.

## Training and data workflow

The training summary CSV uses columns including:

`label`, `peak_angle`, `min_angle`, `peak_dx_from_midline`, `peak_abs_dx_from_shoulder`, `min_dy_from_shoulder`, `max_dy_from_shoulder`, `peak_elbow_dx_from_shoulder`, `peak_elbow_dy_from_shoulder`, `path_length`, `net_displacement`, `straightness_ratio`, and `shoulder_width`.

The runtime classifier normalizes relevant values by shoulder width. The documented workflow is:

1. Collect labeled samples.
2. Put or merge generated CSVs into `data/`.
3. Fit depth-3 scikit-learn decision trees per arm.
4. Translate tree splits into `config.py`.
5. Run the classifier tests.

Before changing classifier thresholds or claiming improved accuracy, inspect the actual CSV labels and test methodology. The checked-in `data/punch_training_data.csv` appears to contain only left-arm summary rows, even though the documentation and configuration discuss separately fitted right-arm data. Also note that the existing training regression test evaluates thresholds on the same data used to derive them; it is a regression sanity check, not a held-out generalization measurement.

The collector duplicates geometry and summary logic instead of importing all runtime feature-building code. Changes to feature definitions can therefore create collector/runtime drift. The collector also writes CSVs relative to the current working directory, while the README describes moving them into `data/`.

## Commands and dependencies

The project requires Python `>=3.9` and uses:

- `opencv-python>=4.8`
- `mediapipe==0.10.21`
- `numpy>=1.24,<2`
- Development/testing: `pytest>=7.0`, `scikit-learn>=1.3`

Console entry points from `pyproject.toml`:

```text
wiiboxing-detect = wiiboxing.cli:detect
wiiboxing-play   = wiiboxing.cli:play
wiiboxing-duel   = wiiboxing.cli:duel
```

Direct alternatives are:

```powershell
python -m wiiboxing.detector
python -m wiiboxing.game
python -m wiiboxing.duel
```

The camera modes assume webcam index `0`, OpenCV GUI support, and a functioning MediaPipe installation. Run the pure-logic test suite with:

```powershell
pytest
```

## Gameplay flow

`wiiboxing-detect` opens the webcam, detects completed punches, prints classifications, and draws an overlay.

`wiiboxing-play` spawns targets in head/body regions. Wrist contact counts as a hit immediately. Completed classified punches may later mark a target as missed, so preserve the distinction between contact timing and classifier completion unless the behavior is intentionally redesigned.

`wiiboxing-duel` has these phases:

- `MENU`: solo practice, two-player duel, mirror preview, fullscreen toggle, or quit.
- `CALIBRATION`: players must be visible, centered, within configured distance limits, and hold position for the configured duration.
- `FIGHT`: classified punches damage the opponent; head shots may be dodged by leaning while body and low shots land.
- `KNOCKDOWN`: the downed player has a timed recovery-target challenge; required hits increase and maximum HP can decrease.
- `GAME_OVER`: recovery timeout ends the match and the winner is determined by `duel.py`.

## Known limitations and inspection targets

Treat these as things to verify before changing related code, not as automatically confirmed bugs:

- The checked-in training data and documented right-arm fitting story may not match.
- The classifier regression test has training-data leakage.
- Collector feature calculations may drift from runtime feature calculations.
- Collector output paths depend on the current working directory.
- Target practice scores raw wrist contact, which may be intentionally more responsive than strict punch-event scoring.
- Tracker updates occur only while an arm is visible; visibility loss during an active motion may leave a motion unresolved until later.
- `duel.py` uses `time.time()` inside `_process_fighting_half()` instead of necessarily reusing the frame timestamp already available to the caller. Verify timing behavior before changing it.
- `targets.py` contains documentation referring to `survival.py`, which is not currently present.
- `ring_render.py` assumes wrist coordinates are normalized to the relevant half-frame.
- The code uses the classic `mp.solutions.pose.Pose` API and creates two Pose models in duel mode. Check compatibility and performance before migrating MediaPipe APIs.
- `PlayerState` is mostly logic-only but prints status messages during some transitions, which may matter for reuse and testing.
- There is limited coverage for MediaPipe extraction, camera failures, OpenCV rendering, random target spawning, target hits, collector CSV writing, full detector/game/duel integration, two-player event resolution, and real webcam performance.

## Working protocol

For every requested change:

1. Read the relevant implementation, neighboring tests, configuration, and call sites before editing.
2. Identify the code that actually decides or mutates the behavior; do not stop at a forwarding wrapper.
3. State one falsifiable hypothesis about the behavior or bug and one focused check that could disprove it.
4. Make the smallest change that addresses the root cause and preserve public APIs unless there is a clear reason to change them.
5. Keep tunable thresholds and gameplay constants in `config.py`.
6. Do not assume right/left arm symmetry, normalized/pixel coordinate equivalence, or detector/game/duel behavioral equivalence.
7. Add or update focused tests for pure logic. Avoid pretending hardware or integration behavior is covered by unit tests.
8. Run `pytest` after changes and report the exact result. For interactive changes, state what could not be verified without a webcam.
9. Do not revert unrelated user changes, add unrelated refactors, or commit changes unless explicitly asked.

## Response format for future tasks

When you finish a task, report:

- **Hypothesis:** what behavior you believed was wrong or missing.
- **Inspected:** the files and symbols that controlled it.
- **Change:** what was modified and why.
- **Validation:** commands run and their results.
- **Remaining risk:** any webcam, MediaPipe, data-quality, rendering, or integration behavior not verified.

Start each task by checking the current working tree and relevant tests. The repository's current code and tests take precedence over this handoff when they disagree.

---

End of handoff prompt.