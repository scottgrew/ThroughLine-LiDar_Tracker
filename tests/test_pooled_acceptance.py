"""Oracle tests for the pooled acceptance driver helpers.

The evaluator text is the real two-line format of compute_tracking_metrics:
one LEVEL_1 and one LEVEL_2 line per class. Tolerance checks use values
exact in binary (0.25, 0.5) so the boundary is not a rounding accident.
"""

import numpy as np
from pooled_acceptance import (
    ClassScore,
    compare_value,
    is_anomaly,
    parse_level2,
)

EVALUATOR_TEXT = (
    "OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1: [MOTA 0.5] [MOTP 0.1] [Miss 0.3] "
    "[Mismatch 0.02] [FP 0.2]\n"
    "OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [MOTA 0.459946] [MOTP 0.150015] "
    "[Miss 0.314324] [Mismatch 0.0105227] [FP 0.215207]\n"
)


def test_parse_level2_reads_only_the_level_2_line():
    scores = parse_level2(EVALUATOR_TEXT)
    got = scores["VEHICLE"]
    assert set(scores) == {"VEHICLE"}
    assert np.allclose(
        [got.mota, got.miss, got.mismatch, got.fp],
        [0.459946, 0.314324, 0.0105227, 0.215207], rtol=0, atol=1e-12)


def test_all_zero_score_is_an_anomaly():
    assert is_anomaly(ClassScore(mota=0.0, miss=0.0, mismatch=0.0, fp=0.0))


def test_any_nonzero_value_is_not_an_anomaly():
    scores = [
        ClassScore(mota=0.459946, miss=0.314324, mismatch=0.01, fp=0.2),
        ClassScore(mota=0.0, miss=0.0, mismatch=0.0, fp=0.1),
        ClassScore(mota=-0.1, miss=0.0, mismatch=0.0, fp=0.0),
    ]
    assert [is_anomaly(score) for score in scores] == [False] * 3


def test_compare_value_passes_exactly_at_tolerance_either_side():
    above = compare_value("mota", 0.5, 0.25, 0.25)
    below = compare_value("mota", 0.25, 0.5, 0.25)
    assert [above.name, below.name] == ["mota", "mota"]
    assert [above.tolerance, below.tolerance] == [0.25, 0.25]
    assert [bool(above.passed), bool(below.passed)] == [True, True]


def test_compare_value_fails_just_above_tolerance_either_side():
    above = compare_value("mota", 0.5 + 1e-9, 0.25, 0.25)
    below = compare_value("mota", 0.25 - 1e-9, 0.5, 0.25)
    assert [bool(above.passed), bool(below.passed)] == [False, False]
