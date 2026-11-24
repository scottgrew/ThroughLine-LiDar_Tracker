"""Pooled acceptance driver: donor emulator vs CasA on many segments."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from detector_error import ErrorStats, measure, segment_frames
from donor_emulator import build_index, emulate_donor
from donor_library import default_library, exclude_segment
from stage_segment import (
    emulate_inputs,
    read_detections,
    read_poses,
    scored_labels,
    write_log,
)
from waymo_boxes import TRACKED_OBJECT_CLASSES, read_ground_truth_boxes

VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 4
TYPE_NAMES = {
    VEHICLE: "VEHICLE",
    PEDESTRIAN: "PEDESTRIAN",
    CYCLIST: "CYCLIST",
}
MOTA_TOL = 0.05
MISMATCH_TOL = {VEHICLE: 0.003, PEDESTRIAN: 0.01, CYCLIST: 0.01}
IOU_TOL = {VEHICLE: 0.03, PEDESTRIAN: 0.02, CYCLIST: 0.02}
MISS_TOL = {VEHICLE: 0.03, PEDESTRIAN: 0.02, CYCLIST: 0.02}
FP_SURVIVAL_TOL = 0.03
# A false-positive rate within 10% of CasA's moves MOTA by at most half of
# the pedestrian MOTA tolerance (FP is about 0.24 of MOTA for pedestrians).
FP_RATE_REL_TOL = 0.10
LAG1_TOL = 0.03
MIN_IOU_MATCHED = 1000
MIN_FP_PERSIST = 1000
MIN_LAG1_PAIRS = 10000
_LEVEL2 = re.compile(
    r"OBJECT_TYPE_TYPE_(\w+)_LEVEL_2:.*?"
    r"\[MOTA ([^\]]+)\].*?"
    r"\[Miss ([^\]]+)\].*?"
    r"\[Mismatch ([^\]]+)\].*?"
    r"\[FP ([^\]]+)\]",
    re.DOTALL,
)


class SegmentData(NamedTuple):
    name: str
    times: list[int]
    poses: dict
    ground_truth: list
    casa: list


class ClassScore(NamedTuple):
    mota: float
    miss: float
    mismatch: float
    fp: float


class Check(NamedTuple):
    name: str
    value: float
    tolerance: float
    passed: bool


def load_segment(parquet_root, result_path, name):
    poses = read_poses(parquet_root, name)
    times = sorted(poses)
    casa = read_detections(result_path, name)
    if len(casa) != len(times):
        raise SystemExit(
            f"{name}: {len(casa)} CasA frames, {len(times)} poses")
    ground_truth = read_ground_truth_boxes(parquet_root, name)
    return SegmentData(name, times, poses, ground_truth, casa)


def stage_variant(work, data, label, detections):
    labels_by_time = scored_labels(data.ground_truth)
    log_path = Path(work) / label / f"{data.name}.mcap"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_log(
        str(log_path), data.name, data.times, data.poses, None,
        detections, labels_by_time)
    return log_path


def run_tracker(tracker, log_path, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [tracker, "--segment", str(log_path), "--export", str(csv_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr or error.stdout or str(error)
        raise RuntimeError(message) from error


def parse_level2(text):
    scores: dict[str, ClassScore] = {}
    for match in _LEVEL2.finditer(text):
        kind = match.group(1)
        scores[kind] = ClassScore(
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
            float(match.group(5)),
        )
    return scores


def is_anomaly(score: ClassScore) -> bool:
    return (
        score.mota == 0.0 and score.miss == 0.0
        and score.mismatch == 0.0 and score.fp == 0.0
    )


def score(evaluator, runs, gt_by_segment, work):
    """Pooled evaluator scores over many (segment, tracks.csv) pairs."""
    from waymo_open_dataset.protos import metrics_pb2

    from eval.to_waymo_objects import (
        build_ground_truth_objects,
        build_prediction_objects,
    )

    pred_objects = metrics_pb2.Objects()
    gt_objects = metrics_pb2.Objects()
    for segment_name, csv_path in runs:
        pred = build_prediction_objects(str(csv_path), segment_name)
        gt = build_ground_truth_objects(
            gt_by_segment[segment_name], segment_name)
        pred_objects.objects.extend(pred.objects)
        gt_objects.objects.extend(gt.objects)
    # Inside work, so an evaluator in a container sees the files once work is
    # mounted at the same path.
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        pred_file = Path(tmp) / "pred.pb"
        gt_file = Path(tmp) / "gt.pb"
        with open(pred_file, "wb") as stream:
            stream.write(pred_objects.SerializeToString())
        with open(gt_file, "wb") as stream:
            stream.write(gt_objects.SerializeToString())
        command = [*evaluator.split(), str(pred_file), str(gt_file)]
        completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"evaluator failed: {completed.stderr[-2000:]}")
    return parse_level2(completed.stdout)


def compare_value(name, emulated, reference, tolerance):
    diff = abs(emulated - reference)
    return Check(name, diff, tolerance, diff <= tolerance)


def stats_checks(casa: ErrorStats, emulated: ErrorStats):
    checks: list[Check] = []
    casa_bands = {
        (row.object_class, row.band): row for row in casa.pooled_bands}
    emu_bands = {
        (row.object_class, row.band): row for row in emulated.pooled_bands}
    for cls in TRACKED_OBJECT_CLASSES:
        for band in range(3):
            key = (cls, band)
            if key not in casa_bands:
                continue
            ref = casa_bands[key]
            if ref.matched < MIN_IOU_MATCHED:
                continue
            emu = emu_bands.get(key)
            if emu is None:
                continue
            tag = TYPE_NAMES[cls]
            checks.append(compare_value(
                f"iou_pass_{tag}_band{band}", emu.iou_pass, ref.iou_pass,
                IOU_TOL[cls]))
            checks.append(compare_value(
                f"miss_rate_{tag}_band{band}", emu.miss_rate, ref.miss_rate,
                MISS_TOL[cls]))
    casa_fp = {row.object_class: row for row in casa.false_positives}
    emu_fp = {row.object_class: row for row in emulated.false_positives}
    for cls in TRACKED_OBJECT_CLASSES:
        ref = casa_fp.get(cls)
        emu = emu_fp.get(cls)
        if ref is None or ref.fp_persist_n < MIN_FP_PERSIST:
            continue
        if emu is None:
            continue
        checks.append(compare_value(
            f"fp_survival_{TYPE_NAMES[cls]}", emu.fp_survival,
            ref.fp_survival, FP_SURVIVAL_TOL))
        ref_rate = ref.fp_count / casa.total_frames
        checks.append(compare_value(
            f"fp_rate_{TYPE_NAMES[cls]}", emu.fp_count / emulated.total_frames,
            ref_rate, FP_RATE_REL_TOL * ref_rate))
    casa_models = {row.object_class: row for row in casa.class_models}
    emu_models = {row.object_class: row for row in emulated.class_models}
    for cls in TRACKED_OBJECT_CLASSES:
        ref = casa_models.get(cls)
        emu = emu_models.get(cls)
        if ref is None or ref.lag1_pairs < MIN_LAG1_PAIRS:
            continue
        if emu is None:
            continue
        for comp, label in enumerate(
                ("along", "across", "z", "yaw", "length", "width", "height")):
            checks.append(compare_value(
                f"rank_lag1_{TYPE_NAMES[cls]}_{label}",
                emu.rank_lag1[comp], ref.rank_lag1[comp], LAG1_TOL))
    return checks


def _mean_scores(score_list: list[dict[str, ClassScore]]):
    if not score_list:
        return {}
    kinds = score_list[0].keys()
    averaged: dict[str, ClassScore] = {}
    for kind in kinds:
        values = [scores[kind] for scores in score_list if kind in scores]
        if not values:
            continue
        averaged[kind] = ClassScore(
            sum(row.mota for row in values) / len(values),
            sum(row.miss for row in values) / len(values),
            sum(row.mismatch for row in values) / len(values),
            sum(row.fp for row in values) / len(values),
        )
    return averaged


def _serialise_score(score: ClassScore):
    return {
        "mota": score.mota, "miss": score.miss,
        "mismatch": score.mismatch, "fp": score.fp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--tracker", default="build/tracker")
    parser.add_argument("--work", default=None)
    parser.add_argument("--segments", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--acceptance", required=True)
    args = parser.parse_args()
    evaluator = os.environ.get("EVALUATOR")
    if not evaluator:
        raise SystemExit("EVALUATOR is not set")
    work = args.work or tempfile.mkdtemp(prefix="acceptance_")
    library = default_library()
    segments = [load_segment(args.parquet_root, args.detections, name)
                for name in args.segments]
    casa_runs = []
    casa_frames = []
    casa_by_segment_scores: dict[str, dict[str, ClassScore]] = {}
    for data in segments:
        path = stage_variant(work, data, "casa", data.casa)
        csv_path = Path(work) / "casa" / f"{data.name}.csv"
        run_tracker(args.tracker, path, csv_path)
        casa_runs.append((data.name, csv_path))
        casa_frames.extend(segment_frames(
            args.parquet_root, args.detections, data.name))
    gt_root = args.parquet_root
    gt_map = {s.name: gt_root for s in segments}
    pooled_casa = score(evaluator, casa_runs, gt_map, work)
    for data in segments:
        csv_path = Path(work) / "casa" / f"{data.name}.csv"
        seg_scores = score(
            evaluator, [(data.name, csv_path)], {data.name: gt_root}, work)
        casa_by_segment_scores[data.name] = seg_scores
    for cls in TRACKED_OBJECT_CLASSES:
        tag = TYPE_NAMES[cls]
        if tag not in pooled_casa:
            continue
        values = [
            casa_by_segment_scores[name][tag].mota
            for name in casa_by_segment_scores
            if tag in casa_by_segment_scores[name]
        ]
        if not values:
            continue
        pooled = pooled_casa[tag].mota
        if pooled < min(values) - 1e-9 or pooled > max(values) + 1e-9:
            raise SystemExit("evaluator does not group by context_name")
    index_by_segment = {
        data.name: build_index(exclude_segment(library, data.name))
        for data in segments}
    seed_scores: dict[int, dict[str, ClassScore]] = {}
    seed_per_segment: dict[int, dict[str, dict[str, ClassScore]]] = {}
    emu_frames_by_seed: dict[int, list] = {}
    for seed in args.seeds:
        emu_runs = []
        all_emu_frames = []
        per_seg: dict[str, dict[str, ClassScore]] = {}
        for data in segments:
            frames = emulate_inputs(data.ground_truth, data.times, data.poses)
            index = index_by_segment[data.name]
            detections = emulate_donor(frames, data.name, seed, index)
            path = stage_variant(work, data, f"emu_{seed}", detections)
            csv_path = Path(work) / f"emu_{seed}" / f"{data.name}.csv"
            run_tracker(args.tracker, path, csv_path)
            emu_runs.append((data.name, csv_path))
            measure_frames = segment_frames(
                args.parquet_root, args.detections, data.name)
            emulated_measure = [
                frame._replace(detections=detections[frame.frame])
                for frame in measure_frames]
            all_emu_frames.extend(emulated_measure)
            per_seg[data.name] = score(
                evaluator, [(data.name, csv_path)], {data.name: gt_root}, work)
        seed_scores[seed] = score(
            evaluator, emu_runs, {s.name: gt_root for s in segments}, work)
        seed_per_segment[seed] = per_seg
        emu_frames_by_seed[seed] = all_emu_frames
    casa_stats = measure(casa_frames)
    mean_emu = _mean_scores(list(seed_scores.values()))
    checks: list[Check] = []
    for tag, ref in pooled_casa.items():
        if is_anomaly(ref):
            continue
        emu = mean_emu.get(tag)
        if emu is None or is_anomaly(emu):
            continue
        checks.append(compare_value(
            f"mota_{tag}", emu.mota, ref.mota, MOTA_TOL))
        cls = next(code for code, name in TYPE_NAMES.items() if name == tag)
        if cls in MISMATCH_TOL:
            checks.append(compare_value(
                f"mismatch_{tag}", emu.mismatch, ref.mismatch,
                MISMATCH_TOL[cls]))
    pooled_emu_by_seed = [
        measure(emu_frames_by_seed[seed]) for seed in args.seeds]
    for emu_stats in pooled_emu_by_seed:
        checks.extend(stats_checks(casa_stats, emu_stats))
    accepted = all(check.passed for check in checks if check.tolerance > 0)
    row_counts = {}
    for cls in TRACKED_OBJECT_CLASSES:
        for band in range(3):
            count = int(np.sum(
                (library.cls == cls) & (library.band == band)))
            row_counts[f"{TYPE_NAMES[cls]}_band{band}"] = count
    document = {
        "segments": args.segments,
        "seeds": args.seeds,
        "tolerances": {
            "mota": MOTA_TOL,
            "mismatch": {TYPE_NAMES[k]: v for k, v in MISMATCH_TOL.items()},
            "iou_pass": {TYPE_NAMES[k]: v for k, v in IOU_TOL.items()},
            "miss_rate": {TYPE_NAMES[k]: v for k, v in MISS_TOL.items()},
            "fp_survival": FP_SURVIVAL_TOL,
            "fp_rate_relative": FP_RATE_REL_TOL,
            "rank_lag1": LAG1_TOL,
        },
        "library_row_counts": row_counts,
        "casa": {k: _serialise_score(v) for k, v in pooled_casa.items()},
        "emulated": {
            str(seed): {k: _serialise_score(v) for k, v in scores.items()}
            for seed, scores in seed_scores.items()
        },
        "emulated_mean": {k: _serialise_score(v) for k, v in mean_emu.items()},
        "per_segment": {
            "casa": {
                name: {k: _serialise_score(v) for k, v in scores.items()}
                for name, scores in casa_by_segment_scores.items()
            },
            **{
                f"emu_{seed}": {
                    name: {k: _serialise_score(v) for k, v in scores.items()}
                    for name, scores in per_seg.items()
                }
                for seed, per_seg in seed_per_segment.items()
            },
        },
        "checks": [
            {"name": c.name, "value": c.value, "tolerance": c.tolerance,
             "passed": c.passed}
            for c in checks
        ],
        "accepted": accepted,
    }
    Path(args.acceptance).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
