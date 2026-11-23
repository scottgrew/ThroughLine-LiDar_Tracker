"""Oracles for detector_error.measure, independent of its implementation."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
from detector_error import (
    COMPONENTS,
    FrameInput,
    band_of,
    iou3d,
    measure,
    write_model,
)

VEHICLE = 1
IDENTITY = np.eye(4)
EMPTY_LABELS = (
    np.zeros(0, dtype=np.int64),
    np.zeros(0, dtype=np.int64),
    np.zeros((0, 7)),
)
_BOX = (0, 1, 2, 3, 4, 5, 6)  # x, y, z, l, w, h, yaw


def _box(x, y, z, length, width, height, yaw):
    row = np.zeros((1, 7), dtype=np.float64)
    row[0, _BOX[0]] = x
    row[0, _BOX[1]] = y
    row[0, _BOX[2]] = z
    row[0, _BOX[3]] = length
    row[0, _BOX[4]] = width
    row[0, _BOX[5]] = height
    row[0, _BOX[6]] = yaw
    return row


def test_heading_frame_along_across_and_gated_matching():
    # Label at the origin, heading pi/2: along is +y, across is -x.
    label_boxes = np.array([
        [0.0, 0.0, 1.0, 4.0, 2.0, 1.5, math.pi / 2],
        [50.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
    ])
    label_ids = np.array([1, 2], dtype=np.int64)
    label_classes = np.array([VEHICLE, VEHICLE], dtype=np.int64)
    det_boxes = np.array([[0.0, 0.5, 1.0, 4.0, 2.0, 1.5, math.pi / 2]])
    det_classes = np.array([VEHICLE])
    frame = FrameInput(
        "seg", 0, IDENTITY, (det_classes, det_boxes),
        (label_ids, label_classes, label_boxes),
    )
    stats = measure([frame])
    vehicle_row = next(row for row in stats.pooled_bands
                       if row.object_class == VEHICLE and row.band == 0)
    assert math.isclose(vehicle_row.along_med, 0.5, abs_tol=1e-9)
    assert math.isclose(vehicle_row.across_med, 0.0, abs_tol=1e-9)
    assert vehicle_row.matched == 1


def test_range_band_edges_left_closed():
    ranges = np.array([0.0, 29.999, 30.0, 49.999, 50.0, 80.0])
    assert band_of(ranges).tolist() == [0, 0, 1, 1, 2, 2]


def test_flip_excluded_from_yaw_sigma():
    label_boxes = np.array([[10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0]])
    det_boxes = np.array([[10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 3.0]])
    frame = FrameInput(
        "seg", 0, IDENTITY,
        (np.array([VEHICLE]), det_boxes),
        (np.array([7], dtype=np.int64), np.array([VEHICLE]), label_boxes),
    )
    stats = measure([frame])
    row = stats.pooled_bands[0]
    assert row.flip_frac == 1.0
    assert math.isnan(row.yaw_sig)


def test_fp_persistence_respects_segment():
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
    det = (np.array([VEHICLE]), box.reshape(1, 7))
    frames = [
        FrameInput("a", 0, IDENTITY, det, EMPTY_LABELS),
        FrameInput("a", 1, IDENTITY, det, EMPTY_LABELS),
        FrameInput("b", 0, IDENTITY, det, EMPTY_LABELS),
        FrameInput("b", 1, IDENTITY, det, EMPTY_LABELS),
    ]
    stats = measure(frames)
    fp_row = stats.false_positives[0]
    assert fp_row.fp_survival == 1.0
    assert fp_row.fp_persist_n == 2


def test_iou_identical_boxes():
    label = _box(0.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0)
    assert math.isclose(float(iou3d(label, label)[0]), 1.0)


def test_iou_along_heading_offset():
    label = _box(0.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0)
    det = _box(2.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0)
    bev_overlap = 2.0 * 2.0
    volume = 4.0 * 2.0 * 1.5
    union = volume + volume - bev_overlap * 1.5
    expected = bev_overlap * 1.5 / union
    assert math.isclose(float(iou3d(det, label)[0]), expected, rel_tol=1e-6)
    assert math.isclose(expected, 1.0 / 3.0, rel_tol=1e-6)


def test_iou_vertical_offset():
    label = _box(0.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0)
    det = _box(0.0, 0.0, 1.75, 4.0, 2.0, 1.5, 0.0)
    z_overlap = 0.75
    footprint = 4.0 * 2.0
    volume = footprint * 1.5
    intersection = footprint * z_overlap
    expected = intersection / (2 * volume - intersection)
    assert math.isclose(float(iou3d(det, label)[0]), expected, rel_tol=1e-6)
    assert math.isclose(expected, 1.0 / 3.0, rel_tol=1e-6)


def test_iou_pi_rotation_same_box():
    label = _box(0.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0)
    det = _box(0.0, 0.0, 1.0, 4.0, 2.0, 1.5, math.pi)
    assert math.isclose(float(iou3d(det, label)[0]), 1.0, rel_tol=1e-6)


def test_percentiles_along_one_frame():
    labels = np.array([
        [10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [15.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [20.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
    ])
    # All three labels inside band 0, [0, 30) m.
    dets = np.array([
        [10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [15.5, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [19.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
    ])
    along = np.array([0.0, 0.5, -1.0])
    oracle = np.percentile(along, np.arange(101))
    frame = FrameInput(
        "seg", 0, IDENTITY,
        (np.array([VEHICLE, VEHICLE, VEHICLE]), dets),
        (np.array([1, 2, 3], dtype=np.int64),
         np.array([VEHICLE, VEHICLE, VEHICLE]), labels),
    )
    stats = measure([frame])
    model = stats.class_models[0]
    band0 = model.bands[0]
    got = np.array(band0.percentiles[0])
    assert band0.matched == 3
    assert np.allclose(got, oracle, rtol=0, atol=1e-12)


def test_write_model_schema_keys():
    labels = np.array([
        [15.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [18.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
    ])
    dets = labels.copy()
    frame = FrameInput(
        "s0", 0, IDENTITY,
        (np.array([VEHICLE, VEHICLE]), dets),
        (np.array([9, 10], dtype=np.int64),
         np.array([VEHICLE, VEHICLE]), labels),
    )
    stats = measure([frame])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        write_model(stats, str(path))
        doc = json.loads(path.read_text())
    assert set(doc) == {"source", "components", "classes"}
    source_keys = {
        "tool", "segments", "frames", "gate_m", "iou_thresholds",
    }
    assert set(doc["source"]) == source_keys
    assert doc["components"] == list(COMPONENTS)
    vehicle = doc["classes"]["1"]
    class_keys = {
        "bands", "rank_lag1", "rank_corr", "p_flip_after_no_flip",
        "p_flip_after_flip", "fp_per_frame", "fp_survival",
        "fp_range_deciles_m", "fp_size_m",
    }
    assert set(vehicle) == class_keys
    band = vehicle["bands"][0]
    assert set(band) == {
        "matched", "iou_pass", "p_miss_after_hit", "p_miss_after_miss",
        "percentiles",
    }
    assert len(band["percentiles"]) == 7
    for row in band["percentiles"]:
        assert len(row) == 101
    corr = np.array(vehicle["rank_corr"])
    assert corr.shape == (7, 7)
    assert np.allclose(corr, corr.T, atol=1e-12)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-12)


def test_constant_component_corr_and_model_has_no_nan():
    """No variation -> off-diagonal 0, diagonal 1, lag-1 0; model is JSON-safe."""
    labels = np.array([
        [10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [15.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [20.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [25.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0],
    ])
    dets = labels.copy()
    dets[:, 2] = 1.25
    frames = [
        FrameInput(
            "seg", t, IDENTITY,
            (np.array([VEHICLE]), dets[t:t + 1]),
            (np.array([t + 1], dtype=np.int64),
             np.array([VEHICLE]), labels[t:t + 1]),
        )
        for t in range(4)
    ]
    stats = measure(frames)
    model = stats.class_models[0]
    idx_z = COMPONENTS.index("z")
    corr = np.array(model.rank_corr)
    assert corr[idx_z, idx_z] == 1.0
    for j in range(len(COMPONENTS)):
        if j == idx_z:
            continue
        assert corr[idx_z, j] == 0.0
        assert corr[j, idx_z] == 0.0
    assert model.rank_lag1[idx_z] == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        write_model(stats, str(path))
        text = path.read_text()
        json.dumps(json.loads(text), allow_nan=False)