# Throughline (full)

This branch extends `main`. It adds a detector emulator that replays CasA's recorded per-object errors, forecast uncertainty calibrated against the labels and drawn as 2-sigma ellipses, a side-by-side view of CasA, the emulator and independent noise with the evaluator's errors marked, a latency harness, and tracked Waymo Motion scenarios for a motion planner.

C++ multi-object tracker for Waymo Open Dataset lidar. It tracks the cars, pedestrians and cyclists in CasA's 3D detections, keeps each object's ID through missed detections, predicts each track 3 s ahead, and streams the scene to Lichtblick.

![Tracking through a turn](docs/media/turn.gif)

## Results

MOTA from Waymo's evaluator (LEVEL_2) on CasA's published detections:

| scene | cars | pedestrians |
|---|---|---|
| fast road, 63 km/h | 0.30 | 0.23 |
| junction at walking pace | 0.54 | 0.57 |
| slow turn, 21 km/h | 0.71 | 0.34 |

Mean NEES of the 3 s forecast against the labels over 23 segments (2 degrees of freedom): vehicles 1.92, pedestrians 1.91.

MOTA with the emulated detector against CasA, 23 segments × 3 seeds:

| class | CasA | emulated |
|---|---|---|
| vehicles | 0.522 | 0.522 |
| pedestrians | 0.432 | 0.341 |
| cyclists | 0.226 | 0.186 |

## Method

### Motion model

Each track runs an extended Kalman filter on a constant turn rate and velocity model: position, heading, speed and turn rate. Per-class priors set the starting speed and turn-rate uncertainty and the acceleration noise for vehicles, pedestrians and cyclists. Code: `src/filter`.

### Association

Each frame, detections are matched to the predicted tracks with the Hungarian algorithm on Mahalanobis distance, gated at χ² = 11.34 (99% for position and heading). A track is confirmed after 3 hits and deleted after 5 misses. Code: `src/tracker`.

### Forecast

Each confirmed track's motion model is run 3 s ahead in 0.1 s steps and drawn as a path with its 2-sigma position ellipse. The acceleration noise per class was chosen from a grid over 23 validation segments so that the forecast NEES against the labels is near 2. Code: `src/filter`, `stage/forecast_consistency.py`.

### Detector emulator

The emulator turns Waymo labels into detections with CasA's errors. A donor library holds CasA's per-object error histories, matched against the labels. Each emulated object replays donor histories, hopping between similar library rows. A model-based variant draws errors from a fitted CasA error model, with an independent-noise option. Code: `stage/detector_error.py`, `stage/donor_library.py`, `stage/donor_emulator.py`, `stage/emulate_detector.py`.

### Tracking view

`make tview` writes one MCAP with CasA, the donor emulator and independent noise side by side, with Waymo's misses, false positives and ID switches marked on each. Code: `eval/marks.py`, `eval/tview.sh`, `src/marks`.

### Planner scenarios

`stage/stage_womd.py` stages Waymo Open Motion Dataset scenarios with emulated CasA detections, and `stage/tracked_scenarios.py` replaces each scenario's non-ego history with the tracker's output, for a motion planner.

### Input

`stage/stage_segment.py` reads a Waymo v2 Parquet segment and CasA's detections into an MCAP log of lidar points, vehicle poses and boxes. Detections below a score of 0.5 (vehicles, cyclists) or 0.7 (pedestrians) are dropped.

## Build

macOS on Intel. Homebrew `cmake`, `protobuf`, `zstd`, `lz4`; Python `pyarrow`, `numpy`, `pandas`, `scipy`, `protobuf`, `mcap-protobuf-support`; Docker for the evaluator. Data in `~/waymo-data`: Waymo v2 Parquet and CasA's `result.pkl` from [3D-Multi-Object-Tracker](https://github.com/hailanyi/3D-Multi-Object-Tracker).

```
make build
make test       # C++ and Python unit tests
make check      # clang-format, clang-tidy, ruff
```

## Run

```
make stage              # stage a segment and its CasA boxes
make live               # track live; open ws://127.0.0.1:8765 in Lichtblick
make record             # track live and write demo.mcap
make run                # track at full speed and write tracks.csv
make official           # score tracks.csv with Waymo's evaluator
make stage-emulated     # stage a segment with emulated detections
make tview              # CasA, emulator and independent noise side by side
make consistency        # forecast NEES against the labels
make latency            # frame-time captures with and without a viewer
```

`SEGMENT=<name>` picks the segment. `viz/lichtblick_layout.json` and `viz/tview_layout.json` are the Lichtblick layouts.

## Layout

```
src/     tracker, filter, scene output, log reader and evaluator marks
stage/   Waymo and CasA input to MCAP, detector emulator, forecast calibration, planner scenarios
eval/    Waymo's evaluator in Docker, marks, tracking view, latency summary
tests/   C++ (Catch2) and Python unit tests
viz/     Lichtblick layouts
```

## Status

- Done: tracking, forecast calibration, the detector emulator, the tracking view, the latency harness and the planner scenarios.
- Known limitations: forecasts for parked objects are overconfident; emulated pedestrian MOTA is 0.34 against CasA's 0.43.
- Planned: a track existence probability (IPDA) in place of the fixed 3-hit and 5-miss rule.
