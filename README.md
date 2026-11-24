# Throughline

C++ multi-object tracker for Waymo Open Dataset lidar. It tracks the cars, pedestrians and cyclists in CasA's 3D detections, keeps each object's ID through missed detections, predicts each track 3 s ahead, and streams the scene to Lichtblick.

![Tracking through a turn](docs/media/turn.gif)

## Results

MOTA from Waymo's evaluator (LEVEL_2) on CasA's published detections:

| scene | cars | pedestrians |
|---|---|---|
| fast road, 63 km/h | 0.30 | 0.23 |
| junction at walking pace | 0.54 | 0.57 |
| slow turn, 21 km/h | 0.71 | 0.34 |

## Method

### Motion model

Each track runs an extended Kalman filter on a constant turn rate and velocity model: position, heading, speed and turn rate. Per-class priors set the starting speed and turn-rate uncertainty and the acceleration noise for vehicles, pedestrians and cyclists. Code: `src/filter`.

### Association

Each frame, detections are matched to the predicted tracks with the Hungarian algorithm on Mahalanobis distance, gated at χ² = 11.34 (99% for position and heading). A track is confirmed after 3 hits and deleted after 5 misses. Code: `src/tracker`.

### Forecast

Each confirmed track's motion model is run 3 s ahead in 0.1 s steps and drawn as a path.

### Input

`stage/stage_segment.py` reads a Waymo v2 Parquet segment and CasA's detections into an MCAP log of lidar points, vehicle poses and boxes. Detections below a score of 0.5 (vehicles, cyclists) or 0.7 (pedestrians) are dropped.

## Build

macOS on Intel. Homebrew `cmake`, `protobuf`, `zstd`, `lz4`; Python `pyarrow`, `numpy`, `pandas`, `protobuf`, `mcap-protobuf-support`; Docker for the evaluator. Data in `~/waymo-data`: Waymo v2 Parquet and CasA's `result.pkl` from [3D-Multi-Object-Tracker](https://github.com/hailanyi/3D-Multi-Object-Tracker).

```
make build
make test       # unit tests
make check      # clang-format, clang-tidy, ruff
```

## Run

```
make stage      # stage a segment and its CasA boxes
make live       # track live; open ws://127.0.0.1:8765 in Lichtblick
make record     # track live and write demo.mcap
make run        # track at full speed and write tracks.csv
make official   # score tracks.csv with Waymo's evaluator
```

`SEGMENT=<name>` picks the segment. `viz/lichtblick_layout.json` is the Lichtblick layout.

## Layout

```
src/     tracker, filter, scene output and log reader
stage/   Waymo and CasA input to MCAP
eval/    Waymo's evaluator in Docker
tests/   unit tests (Catch2)
viz/     Lichtblick layout
```
