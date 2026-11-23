# Throughline

Real-time multi-object tracker for Waymo Open Dataset LiDAR, in C++. It keeps a persistent ID
on every car and pedestrian through missed detections, forecasts each one 3 s ahead, and
streams the scene to Lichtblick.

![Tracking through a turn](docs/media/turn.gif)

## Results

MOTA from Waymo's evaluator (LEVEL_2) on CasA's published detections:

| scene                       | cars | people |
|-----------------------------|------|--------|
| fast road, 63 km/h          | 0.30 | 0.23   |
| crawling through a junction | 0.54 | 0.57   |
| slow turn, 21 km/h          | 0.71 | 0.34   |

3 s forecast NEES over 23 segments: vehicles 1.92, pedestrians 1.91 (2 is consistent).

Emulated detector vs CasA, 23 segments x 3 seeds, MOTA:

|             | CasA  | emulated |
|-------------|-------|----------|
| vehicles    | 0.522 | 0.522    |
| pedestrians | 0.432 | 0.341    |
| cyclists    | 0.226 | 0.186    |

## Method

- Extended Kalman filter per track: position, heading, speed, turn rate.
- Hungarian matching; confirm after 3 hits, drop after 5 misses.
- Detector emulator replaying CasA's recorded per-object errors.
- `make tview`: CasA, emulator and independent noise side by side, with Waymo's misses, false
  positives and ID switches marked.

## Build and run

macOS on Intel. Homebrew `cmake`, `protobuf`, `zstd`, `lz4`; Python `pyarrow`, `numpy`,
`pandas`, `protobuf`, `mcap-protobuf-support`; Docker. Data in `~/waymo-data`: Waymo v2
Parquet and CasA's `result.pkl` from
[3D-Multi-Object-Tracker](https://github.com/hailanyi/3D-Multi-Object-Tracker).

```
make test       # unit tests
make stage      # stage a segment and its CasA boxes
make live       # track live on ws://127.0.0.1:8765
make official   # score with Waymo's evaluator
make tview      # CasA vs emulator view, viz/tview_layout.json
```

## Known limitations

- Forecasts for parked objects are overconfident.
- Emulated pedestrian MOTA is 0.34 against CasA's 0.43.
