"""CasA detector error measurement against Waymo labels."""

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from stage_segment import read_detections, read_poses, wrap_angle
from waymo_boxes import TRACKED_OBJECT_CLASSES, read_ground_truth_boxes

GATE_M = {1: 2.0, 2: 1.0, 4: 1.5}
IOU_THRESHOLDS = {1: 0.7, 2: 0.5, 4: 0.5}
COMPONENTS = (
    "along", "across", "z", "yaw", "length", "width", "height",
)
N_COMPONENTS = len(COMPONENTS)
_YAW_INDEX = COMPONENTS.index("yaw")
MIN_MISS_TRANSITIONS = 100
MIN_FLIP_PAIRS = 30
RANGE_EDGES = np.array([30.0, 50.0])
MAD_TO_SIGMA = 1.4826
_BOX_X, _BOX_Y, _BOX_L, _BOX_W, _BOX_H, _BOX_Z, _BOX_YAW = 0, 1, 3, 4, 5, 2, 6
_M_OFF = 5
_IOU_INDEX = _M_OFF + N_COMPONENTS
_IDENTITY_CORR = tuple(
    tuple(1.0 if i == j else 0.0 for j in range(N_COMPONENTS))
    for i in range(N_COMPONENTS)
)
_LEN_I, _WID_I, _Z_I, _H_I = 4, 5, 2, 6


class FrameInput(NamedTuple):
    segment: str
    frame: int
    world_from_vehicle: np.ndarray
    detections: tuple[np.ndarray, np.ndarray]
    labels: tuple[np.ndarray, np.ndarray, np.ndarray]
class BandStats(NamedTuple):
    object_class: int
    band: int
    matched: int
    miss_rate: float
    fp_per_frame: float
    along_med: float
    along_sig: float
    across_med: float
    across_sig: float
    yaw_sig: float
    flip_frac: float
    dlen_med: float
    dlen_sig: float
    dwid_med: float
    dwid_sig: float
    z_med: float
    z_sig: float
    height_med: float
    height_sig: float
    iou_pass: float
    p_miss_after_hit: float
    p_miss_after_miss: float
    percentiles: tuple[tuple[float, ...], ...]
class ClassModelStats(NamedTuple):
    object_class: int
    bands: tuple[BandStats, ...]
    rank_lag1: tuple[float, ...]
    rank_corr: tuple[tuple[float, ...], ...]
    p_flip_after_no_flip: float
    p_flip_after_flip: float
    flip_pairs: int
    lag1_pairs: int
    fp_per_frame: float
    fp_survival: float
    fp_range_deciles: tuple[float, ...]
    fp_size_m: tuple[float, float, float]
class MissPersistenceBand(NamedTuple):
    object_class: int
    band: int
    p_miss_after_miss: float
    p_miss_after_hit: float
    n_miss: int
    n_hit: int
class FalsePositiveClassStats(NamedTuple):
    object_class: int
    fp_survival: float
    fp_persist_n: int
    fp_count: int
    length_median: float
    width_median: float
    height_median: float
    range_deciles: tuple[float, ...]
class ErrorStats(NamedTuple):
    total_frames: int
    n_segments: int
    segments: tuple[str, ...]
    pooled_bands: tuple[BandStats, ...]
    miss_pooled: tuple[tuple[int, float, int], ...]
    miss_bands: tuple[MissPersistenceBand, ...]
    false_positives: tuple[FalsePositiveClassStats, ...]
    class_models: tuple[ClassModelStats, ...]


def band_of(ranges: np.ndarray) -> np.ndarray:
    return np.digitize(ranges, RANGE_EDGES, right=False)


def _range_bands(rows: Sequence) -> np.ndarray:
    if not rows:
        return np.array([], dtype=int)
    return band_of(np.array([row[1] for row in rows], dtype=np.float64))


def _oriented_polygons(boxes: np.ndarray):
    cos_t, sin_t = np.cos(boxes[:, _BOX_YAW]), np.sin(boxes[:, _BOX_YAW])
    half_l, half_w = boxes[:, _BOX_L] / 2, boxes[:, _BOX_W] / 2
    stacks = []
    for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        px = boxes[:, _BOX_X] + cos_t * half_l * dx - sin_t * half_w * dy
        py = boxes[:, _BOX_Y] + sin_t * half_l * dx + cos_t * half_w * dy
        stacks.append(np.stack([px, py], axis=-1))
    return shapely.polygons(np.stack(stacks, axis=1))


def iou3d(detections: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """BEV shapely intersection times z overlap, divided by union volume."""
    intersection_bev = shapely.area(
        shapely.intersection(_oriented_polygons(detections),
                             _oriented_polygons(labels)),
    )
    z_overlap = np.clip(
        np.minimum(
            detections[:, _BOX_Z] + detections[:, _BOX_H] / 2,
            labels[:, _BOX_Z] + labels[:, _BOX_H] / 2,
        ) - np.maximum(
            detections[:, _BOX_Z] - detections[:, _BOX_H] / 2,
            labels[:, _BOX_Z] - labels[:, _BOX_H] / 2,
        ),
        0, None,
    )
    volume_intersection = intersection_bev * z_overlap
    va = detections[:, _BOX_L] * detections[:, _BOX_W] * detections[:, _BOX_H]
    vb = labels[:, _BOX_L] * labels[:, _BOX_W] * labels[:, _BOX_H]
    return volume_intersection / (va + vb - volume_intersection)


def _consecutive_pairs(records) -> Iterator[tuple[tuple, tuple]]:
    by_key: dict[tuple, list] = {}
    for row in records:
        by_key.setdefault((row[3], row[0], row[2]), []).append(row)
    for recs in by_key.values():
        recs.sort(key=lambda item: item[4])
        for left, right in zip(recs, recs[1:], strict=False):
            if right[4] == left[4] + 1:
                yield left, right


def match_frame(frame: FrameInput, gate_scale: float):
    det_classes, det_boxes = frame.detections
    label_ids, label_classes, label_boxes = frame.labels
    matched, missed, false_pos = [], [], []
    pose = np.asarray(frame.world_from_vehicle, dtype=np.float64).reshape(4, 4)
    segment, frame_index = frame.segment, frame.frame
    for cls in TRACKED_OBJECT_CLASSES:
        detections = det_boxes[det_classes == cls]
        labels = label_boxes[label_classes == cls]
        ids = label_ids[label_classes == cls]
        pairs = []
        if len(detections) and len(labels):
            distance = np.hypot(
                detections[:, None, _BOX_X] - labels[None, :, _BOX_X],
                detections[:, None, _BOX_Y] - labels[None, :, _BOX_Y],
            )
            gate = GATE_M[cls] * gate_scale
            rows, cols = linear_sum_assignment(
                np.where(distance <= gate, distance, 1e6))
            kept = distance[rows, cols] <= gate
            pairs = list(zip(rows[kept], cols[kept], strict=True))
        used_det, used_lab = {r for r, _ in pairs}, {c for _, c in pairs}
        for row, col in pairs:
            heading = labels[col, _BOX_YAW]
            dx = detections[row, _BOX_X] - labels[col, _BOX_X]
            dy = detections[row, _BOX_Y] - labels[col, _BOX_Y]
            c, s = np.cos(heading), np.sin(heading)
            matched.append((
                cls, np.hypot(labels[col, _BOX_X], labels[col, _BOX_Y]),
                ids[col], segment, frame_index,
                dx * c + dy * s, -dx * s + dy * c,
                detections[row, _BOX_Z] - labels[col, _BOX_Z],
                wrap_angle(detections[row, _BOX_YAW] - heading),
                detections[row, _BOX_L] - labels[col, _BOX_L],
                detections[row, _BOX_W] - labels[col, _BOX_W],
                detections[row, _BOX_H] - labels[col, _BOX_H],
                float(iou3d(detections[row:row + 1], labels[col:col + 1])[0]),
            ))
        for col in range(len(labels)):
            if col not in used_lab:
                missed.append((
                    cls, np.hypot(labels[col, _BOX_X], labels[col, _BOX_Y]),
                    ids[col], segment, frame_index,
                ))
        for row in range(len(detections)):
            if row not in used_det:
                wx, wy = detections[row, _BOX_X], detections[row, _BOX_Y]
                world = pose @ np.array([wx, wy, 0.0, 1.0])
                false_pos.append((
                    cls, np.hypot(wx, wy), segment, frame_index,
                    float(world[0]), float(world[1]),
                    float(detections[row, _BOX_L]),
                    float(detections[row, _BOX_W]),
                    float(detections[row, _BOX_H]),
                ))
    return matched, missed, false_pos


def _mad_sigma(values: np.ndarray) -> float:
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def _is_flip(row) -> bool:
    return abs(row[_M_OFF + _YAW_INDEX]) > np.pi / 2


def _flip_persistence(records) -> np.ndarray:
    counts = np.zeros((2, 2), int)
    for left, right in _consecutive_pairs(records):
        counts[int(_is_flip(left)), int(_is_flip(right))] += 1
    return counts


def _clean_spearman(value: float, i: int, j: int) -> float:
    # spearmanr NaN = no measured correlation: off-diagonal 0, diagonal 1.
    if math.isnan(value):
        return 1.0 if i == j else 0.0
    return value


def _class_rank_stats(records):
    pairs: list[list[tuple[float, float]]] = [[] for _ in range(N_COMPONENTS)]
    count = 0
    for left, right in _consecutive_pairs(records):
        if _is_flip(left) or _is_flip(right):
            continue
        count += 1
        for comp in range(N_COMPONENTS):
            off = _M_OFF + comp
            pairs[comp].append((left[off], right[off]))
    lag1: list[float] = []
    for comp_pairs in pairs:
        if len(comp_pairs) < 3:
            lag1.append(0.0)
            continue
        first, second = zip(*comp_pairs, strict=True)
        rho = float(spearmanr(first, second)[0])
        lag1.append(0.0 if math.isnan(rho) else rho)
    rows = [row[_M_OFF:_M_OFF + N_COMPONENTS] for row in records
            if not _is_flip(row)]
    if len(rows) < 2:
        corr = _IDENTITY_CORR
    else:
        mat = np.asarray(spearmanr(np.array(rows)).statistic, dtype=np.float64)
        corr = tuple(
            tuple(_clean_spearman(float(mat[i, j]), i, j)
                  for j in range(N_COMPONENTS))
            for i in range(N_COMPONENTS)
        )
    return tuple(lag1), count, corr


def _miss_tables(matched, missed):
    state: dict[tuple, tuple[int, int]] = {}
    for row, band in zip(matched, _range_bands(matched), strict=True):
        state[(row[0], row[3], row[2], row[4])] = (0, int(band))
    for row, band in zip(missed, _range_bands(missed), strict=True):
        state[(row[0], row[3], row[2], row[4])] = (1, int(band))
    counts: dict[tuple[int, int], np.ndarray] = {}
    for (cls, seg, lid, frame), (status, band) in state.items():
        nxt = state.get((cls, seg, lid, frame + 1))
        if nxt is None:
            continue
        mat = counts.setdefault((cls, band), np.zeros((2, 2), int))
        mat[status, nxt[0]] += 1
    miss_pooled = []
    for cls in TRACKED_OBJECT_CLASSES:
        class_sum = sum(
            (m for (c, _), m in counts.items() if c == cls),
            np.zeros((2, 2), int),
        )
        if class_sum[1].sum():
            miss_pooled.append((
                cls, class_sum[1, 1] / class_sum[1].sum(),
                int(class_sum[1].sum()),
            ))
    bands = tuple(
        MissPersistenceBand(
            cls, band,
            m[1, 1] / m[1].sum() if m[1].sum() else float("nan"),
            m[0, 1] / m[0].sum() if m[0].sum() else float("nan"),
            int(m[1].sum()), int(m[0].sum()),
        )
        for (cls, band), m in sorted(counts.items())
    )
    return tuple(miss_pooled), bands


def _false_positive_stats(false_pos, frames, gate_scale):
    present = {(f.segment, f.frame) for f in frames}
    by_frame: dict[tuple, list[tuple[float, float]]] = {}
    for cls, _, segment, frame, wx, wy, *_ in false_pos:
        by_frame.setdefault((cls, segment, frame), []).append((wx, wy))
    persist: dict[int, list[int]] = {}
    for (cls, segment, frame), points in by_frame.items():
        if (segment, frame + 1) not in present:
            continue
        nxt = np.array(by_frame.get((cls, segment, frame + 1), []), np.float64)
        if nxt.ndim == 1:
            nxt = nxt.reshape(0, 2)
        acc = persist.setdefault(cls, [0, 0])
        gate = GATE_M[cls] * gate_scale
        for wx, wy in points:
            acc[1] += 1
            dxy = np.hypot(nxt[:, 0] - wx, nxt[:, 1] - wy) if len(nxt) else []
            if len(nxt) and np.min(dxy) <= gate:
                acc[0] += 1
    ranges: dict[int, list[float]] = {}
    sizes: dict[int, list] = {}
    for cls, det_range, *_ in false_pos:
        ranges.setdefault(cls, []).append(det_range)
    for cls, *_, length, width, height in false_pos:
        sizes.setdefault(cls, []).append((length, width, height))
    rows = []
    dec = np.arange(0, 101, 10)
    for cls in sorted(ranges):
        sz = np.array(sizes[cls])
        kept, total = persist.get(cls, [0, 0])
        n_rng = len(ranges[cls])
        rows.append(FalsePositiveClassStats(
            cls, kept / total if total else 0.0, total, n_rng,
            float(np.median(sz[:, 0])), float(np.median(sz[:, 1])),
            float(np.median(sz[:, 2])),
            tuple(float(v) for v in np.percentile(ranges[cls], dec)),
        ))
    return tuple(rows)


def _bands_for_class(cls, class_matched, totals, total_frames, miss_by, pool):
    errors = np.array([r[_M_OFF:_M_OFF + N_COMPONENTS] for r in class_matched])
    ious = np.array([r[_IOU_INDEX] for r in class_matched])
    eb = _range_bands(class_matched)
    iou_thr = IOU_THRESHOLDS[cls]
    bands = []
    for band in range(3):
        mask = (eb == band) if len(errors) else np.array([], bool)
        sel = errors[mask] if len(errors) else np.empty((0, 7))
        bi = ious[mask] if len(ious) else np.array([])
        miss, fp = totals.get((cls, band), [0, 0])
        pers = miss_by.get((cls, band))
        # No consecutive-frame data in this band: the chain is memoryless at
        # the band's measured miss rate (0 when the band has no labels).
        rate = miss / (miss + len(sel)) if miss + len(sel) else 0.0
        p_hit = pers.p_miss_after_hit if pers else rate
        p_miss = pers.p_miss_after_miss if pers else rate
        if pers and pers.n_miss < MIN_MISS_TRANSITIONS:
            p_miss = pool.get(cls, p_miss)
        d = (0.0,) * 4 + (float("nan"), 0.0) + (0.0,) * 8 + (0.0,)
        if len(sel):
            yaw = sel[:, _YAW_INDEX]
            ok = np.abs(yaw) <= np.pi / 2
            ys = _mad_sigma(yaw[ok]) if ok.any() else float("nan")
            meds = [float(np.median(sel[:, i])) for i in (0, 1)]
            sigs = [_mad_sigma(sel[:, i]) for i in (0, 1)]
            for idx in (_LEN_I, _WID_I, _Z_I, _H_I):
                meds.append(float(np.median(sel[:, idx])))
                sigs.append(_mad_sigma(sel[:, idx]))
            ip = float(np.mean(bi >= iou_thr)) if len(bi) else 0.0
            d = (meds[0], sigs[0], meds[1], sigs[1], ys,
                 float(np.mean(np.abs(yaw) > np.pi / 2)),
                 meds[2], sigs[2], meds[3], sigs[3],
                 meds[4], sigs[4], meds[5], sigs[5], ip)
        n = len(sel)
        bands.append(BandStats(
            cls, band, n, miss / (miss + n) if miss + n else 0.0,
            fp / total_frames if total_frames else 0.0, *d,
            p_hit, p_miss, _band_percentiles(sel),
        ))
    return bands


def _band_percentiles(errors: np.ndarray) -> tuple[tuple[float, ...], ...]:
    lv = np.arange(101)
    flip = np.abs(errors[:, _YAW_INDEX]) > np.pi / 2
    out = []
    for c in range(N_COMPONENTS):
        col = errors[:, c] if c != _YAW_INDEX else errors[~flip, c]
        out.append(tuple(float(v) for v in np.percentile(col, lv))
                     if len(col) else tuple(0.0 for _ in lv))
    return tuple(out)


def _class_models(matched, fpos, totals, n_fr, miss_bands, miss_pool, fpst):
    miss_by = {(r.object_class, r.band): r for r in miss_bands}
    pool = {c: p for c, p, _ in miss_pool}
    flip_global = np.zeros((2, 2), int)
    flip_by: dict[int, np.ndarray] = {}
    for cls in TRACKED_OBJECT_CLASSES:
        cm = [r for r in matched if r[0] == cls]
        if cm:
            flip_by[cls] = _flip_persistence(cm)
            flip_global += flip_by[cls]
    # No data means no measured effect: with no flipped pair anywhere the
    # pooled P(flip | flip) is 0 (the chain then never stays flipped).
    flip_pooled = (
        flip_global[1, 1] / flip_global[1].sum()
        if flip_global[1].sum() else 0.0
    )
    fp_by = {r.object_class: r for r in fpst}
    models = []
    for cls in TRACKED_OBJECT_CLASSES:
        cm = [r for r in matched if r[0] == cls]
        n_fp = sum(1 for r in fpos if r[0] == cls)
        if not cm and not n_fp:
            continue
        lag1, lag1_n, corr = _class_rank_stats(cm)
        flips = flip_by.get(cls, np.zeros((2, 2), int))
        flip_n = int(flips[1].sum())
        p_ff = flips[1, 1] / max(flip_n, 1)
        if flip_n < MIN_FLIP_PAIRS:
            p_ff = flip_pooled
        fp = fp_by.get(cls)
        models.append(ClassModelStats(
            cls, tuple(_bands_for_class(cls, cm, totals, n_fr, miss_by, pool)),
            lag1, corr,
            flips[0, 1] / flips[0].sum() if flips[0].sum() else 0.0,
            p_ff, flip_n, lag1_n, n_fp / n_fr if n_fr else 0.0,
            fp.fp_survival if fp else 0.0,
            fp.range_deciles if fp else (),
            (fp.length_median, fp.width_median, fp.height_median)
            if fp else (0.0, 0.0, 0.0),
        ))
    return tuple(models)


def measure(frames: Sequence[FrameInput], gate_scale=1.0) -> ErrorStats:
    matched, missed, fp = [], [], []
    for frame in frames:
        m, s, f = match_frame(frame, gate_scale)
        matched.extend(m)
        missed.extend(s)
        fp.extend(f)
    segs = tuple(sorted({frame.segment for frame in frames}))
    n_fr, n_seg = len(frames), len(segs)
    totals: dict[tuple[int, int], list[int]] = {}
    for row, band in zip(missed, _range_bands(missed), strict=True):
        totals.setdefault((row[0], int(band)), [0, 0])[0] += 1
    for row, band in zip(fp, _range_bands(fp), strict=True):
        totals.setdefault((row[0], int(band)), [0, 0])[1] += 1
    miss_pool, miss_bands = _miss_tables(matched, missed)
    fp_stats = _false_positive_stats(fp, frames, gate_scale)
    models = _class_models(
        matched, fp, totals, n_fr, miss_bands, miss_pool, fp_stats)
    pooled = tuple(b for m in models for b in m.bands if b.matched > 0)
    return ErrorStats(n_fr, n_seg, segs, pooled, miss_pool, miss_bands,
                      fp_stats, models)


def format_stats(stats: ErrorStats) -> str:
    hdr = (
        "class band matched miss_rate fp_per_frame along_med along_sig "
        "across_med across_sig yaw_sig flip_frac dlen_med dlen_sig "
        "dwid_med dwid_sig z_med z_sig height_med height_sig iou_pass")
    ff = "{:.4f}".format
    lines = [f"POOLED over {stats.n_segments} segments, "
             f"{stats.total_frames} frames", hdr]
    for r in stats.pooled_bands:
        lines.append(
            f"{r.object_class} {r.band} {r.matched} {ff(r.miss_rate)} "
            f"{ff(r.fp_per_frame)} {ff(r.along_med)} {ff(r.along_sig)} "
            f"{ff(r.across_med)} {ff(r.across_sig)} {ff(r.yaw_sig)} "
            f"{ff(r.flip_frac)} {ff(r.dlen_med)} {ff(r.dlen_sig)} "
            f"{ff(r.dwid_med)} {ff(r.dwid_sig)} {ff(r.z_med)} "
            f"{ff(r.z_sig)} {ff(r.height_med)} {ff(r.height_sig)} "
            f"{ff(r.iou_pass)}")
    for m in stats.class_models:
        lines += [
            f"rank_lag1 class {m.object_class}: {list(m.rank_lag1)} "
            f"pairs {m.lag1_pairs}",
            f"flip persistence class {m.object_class}: "
            f"P(flip next | flip) {m.p_flip_after_flip:.5f} (n "
            f"{m.flip_pairs}), P(flip next | no flip) "
            f"{m.p_flip_after_no_flip:.5f}",
        ]
    lines += [
        f"miss persistence class {c} pooled: P(miss next | miss) "
        f"{prob:.5f} (n {n})" for c, prob, n in stats.miss_pooled]
    lines += [
        f"miss persistence class {r.object_class} band {r.band}: "
        f"P(miss next | miss) {r.p_miss_after_miss:.5f} (n {r.n_miss}), "
        f"P(miss next | hit) {r.p_miss_after_hit:.5f} (n {r.n_hit})"
        for r in stats.miss_bands]
    for r in stats.false_positives:
        d = " ".join(f"{v:.2f}" for v in r.range_deciles)
        lines += [
            f"fp size median class {r.object_class}: length "
            f"{r.length_median:.3f} width {r.width_median:.3f} "
            f"height {r.height_median:.3f}",
            f"fp range deciles class {r.object_class} (n {r.fp_count}): {d}",
            f"fp persistence class {r.object_class}: P(same-class FP within "
            f"gate next frame) {r.fp_survival:.5f} (n {r.fp_persist_n})",
        ]
    return "\n".join(lines) + "\n"


def write_model(stats: ErrorStats, path: str) -> None:
    def round_value(value):
        if isinstance(value, float):
            # 1e-6 relative error is far below the measurement's sampling err.
            # NaN is not masked here: allow_nan=False below rejects it.
            return float(f"{value:.6g}")
        if isinstance(value, list):
            return [round_value(v) for v in value]
        if isinstance(value, dict):
            return {k: round_value(v) for k, v in value.items()}
        return value
    doc = {
        "source": {
            "tool": "stage/detector_error.py",
            "segments": list(stats.segments),
            "frames": stats.total_frames,
            "gate_m": {str(k): v for k, v in GATE_M.items()},
            "iou_thresholds": {str(k): v for k, v in IOU_THRESHOLDS.items()},
        },
        "components": list(COMPONENTS),
        "classes": {
            str(m.object_class): {
                "bands": [{
                    "matched": b.matched, "iou_pass": b.iou_pass,
                    "p_miss_after_hit": b.p_miss_after_hit,
                    "p_miss_after_miss": b.p_miss_after_miss,
                    "percentiles": [list(p) for p in b.percentiles],
                } for b in m.bands],
                "rank_lag1": list(m.rank_lag1),
                "rank_corr": [list(row) for row in m.rank_corr],
                "p_flip_after_no_flip": m.p_flip_after_no_flip,
                "p_flip_after_flip": m.p_flip_after_flip,
                "fp_per_frame": m.fp_per_frame,
                "fp_survival": m.fp_survival,
                "fp_range_deciles_m": list(m.fp_range_deciles),
                "fp_size_m": list(m.fp_size_m),
            }
            for m in stats.class_models
        },
    }
    # Serialize first, so a rejected value leaves no half-written file.
    text = json.dumps(round_value(doc), indent=2, allow_nan=False) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def frame_inputs(segment, times, poses, detections, ground_truth):
    """One FrameInput per timestamp that has scored labels."""
    rows: dict[int, list] = {}
    for lb in ground_truth:
        if (lb.object_class in TRACKED_OBJECT_CLASSES
                and lb.num_lidar_points_in_box):
            rows.setdefault(lb.capture_time_micros, []).append(lb)
    labels = {
        t: (np.array([x.laser_object_id for x in lbs]),
            np.array([x.object_class for x in lbs], np.int64),
            np.array([[*x.box] for x in lbs], dtype=np.float64))
        for t, lbs in rows.items()
    }
    out = []
    for frame_index, (time, (dc, db)) in enumerate(
            zip(times, detections, strict=True)):
        if time not in labels:
            continue
        lid, lc, lb = labels[time]
        out.append(FrameInput(
            segment, frame_index,
            np.asarray(poses[time], dtype=np.float64).reshape(4, 4),
            (dc, db), (lid, lc, lb),
        ))
    return out


def segment_frames(parquet_root, result_path, segment):
    poses = read_poses(parquet_root, segment)
    times = sorted(poses)
    detections = read_detections(result_path, segment)
    if len(detections) != len(times):
        raise SystemExit(
            f"{segment}: {len(detections)} detection frames, "
            f"{len(times)} poses")
    ground_truth = read_ground_truth_boxes(parquet_root, segment)
    return frame_inputs(segment, times, poses, detections, ground_truth)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--gate-scale", type=float, default=1.0)
    p.add_argument("--model-out", metavar="PATH", default=None)
    p.add_argument("segments", nargs="+")
    args = p.parse_args()
    frames = []
    for seg in args.segments:
        frames.extend(segment_frames(
            args.parquet_root, args.detections, seg))
    stats = measure(frames, args.gate_scale)
    if args.model_out:
        write_model(stats, args.model_out)
    sys.stdout.write(format_stats(stats))
