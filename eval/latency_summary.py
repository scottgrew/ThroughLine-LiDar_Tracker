#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def frame_ms(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None)
    if rows.dtype.names is None or "frame_ms" not in rows.dtype.names:
        raise ValueError(f"{path}: missing frame_ms column")
    if rows.size == 0:
        raise ValueError(f"{path}: no data rows")
    return np.atleast_1d(np.asarray(rows["frame_ms"], dtype=np.float64))


def run_stats(frames: np.ndarray, period_ms: float) -> dict[str, float]:
    return {
        "p50": float(np.percentile(frames, 50, method="lower")),
        "p90": float(np.percentile(frames, 90, method="lower")),
        "p99": float(np.percentile(frames, 99, method="lower")),
        "max": float(np.max(frames)),
        "overruns": float(np.count_nonzero(frames > period_ms)),
    }


def summarise(paths: list[Path], period: float, label: str) -> np.ndarray:
    if len(paths) < 5:
        raise SystemExit(f"TooFewRuns: need at least 5 {label} runs")
    stats = [run_stats(frame_ms(path), period) for path in paths]
    print(f"{label} (n={len(paths)})")
    for key in ("p50", "p90", "p99", "max"):
        vals = np.asarray([row[key] for row in stats], dtype=np.float64)
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {key}: {float(np.mean(vals)):.3f} ± {sd:.3f} ms")
    print(f"  overruns: {int(sum(int(row['overruns']) for row in stats))}")
    return np.asarray([row["p99"] for row in stats], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--viewer", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    head = args.baseline[0]
    rows = np.genfromtxt(head, delimiter=",", names=True, dtype=None)
    micros = np.atleast_1d(
        np.asarray(rows["capture_time_micros"], dtype=np.int64))
    period = float(micros[1] - micros[0]) / 1000.0
    base = summarise(args.baseline, period, "baseline")
    view = summarise(args.viewer, period, "viewer")
    delta = float(np.mean(view) - np.mean(base))
    base_var = np.var(base, ddof=1) / base.size
    view_var = np.var(view, ddof=1) / view.size
    se = float(np.sqrt(base_var + view_var))
    print(f"p99 delta (viewer - baseline): {delta:.3f} ± {se:.3f} ms")


if __name__ == "__main__":
    main()
