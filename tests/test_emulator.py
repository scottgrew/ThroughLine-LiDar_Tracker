"""Oracle tests for the copula emulator (T15c), written by the director from
the model's definition in .director/PLAN.md "T15c design", not from code.

Statistical checks allow 4 standard errors. For an autocorrelated 0/1 series
the standard error uses the effective sample size n (1 - lam) / (1 + lam).
"""

import math

import numpy as np
from emulate_detector import (
    COMPONENTS,
    BandModel,
    ClassModel,
    Objects,
    default_model,
    emulate,
    independent_model,
    linear_lag1,
    range_band,
    validate_model,
)
from scipy.stats import spearmanr

VEHICLE = 1
IDENTITY_POSE = np.eye(4)
TRUE_BOX = np.array([20.0, 0.0, 1.0, 4.5, 2.0, 1.6, 0.7])  # x y z l w h yaw
P = np.linspace(0.0, 1.0, 101)  # the percentile levels, as probabilities
ALONG, ACROSS, Z, YAW, LENGTH, WIDTH, HEIGHT = range(7)


def percentiles(**quantile_functions):
    """(7, 101) percentiles: each named component's quantile function at P,
    every other component 0 (no error)."""
    table = np.zeros((7, 101))
    for name, quantile in quantile_functions.items():
        table[COMPONENTS.index(name)] = quantile(P)
    return table


def model(table=None, miss=(0.0, 0.0), rank_lag1=None, rank_corr=None, flips=(0.0, 0.0), fp=(0.0, 0.0)):
    """A ClassModel with the same band model in every band."""
    band = BandModel(p_miss_after_hit=miss[0], p_miss_after_miss=miss[1],
                     percentiles=np.zeros((7, 101)) if table is None else table)
    return ClassModel(
        bands=(band,) * 3,
        rank_lag1=np.zeros(7) if rank_lag1 is None else np.asarray(rank_lag1, float),
        rank_corr=np.eye(7) if rank_corr is None else np.asarray(rank_corr, float),
        p_flip_after_no_flip=flips[0], p_flip_after_flip=flips[1],
        fp_per_frame=fp[0], fp_survival=fp[1],
        fp_range_deciles_m=tuple(float(v) for v in range(0, 101, 10)),
        fp_size_m=(4.0, 2.0, 1.5))


def one_vehicle(steps):
    objects = Objects(ids=np.array([7], np.uint64), classes=np.array([VEHICLE]), boxes=TRUE_BOX[None, :].copy())
    return ((objects, IDENTITY_POSE) for _ in range(steps))


def no_objects(steps):
    empty = Objects(ids=np.zeros(0, np.uint64), classes=np.zeros(0, np.int64), boxes=np.zeros((0, 7)))
    return ((empty, IDENTITY_POSE) for _ in range(steps))


def run(frames, class_model, seed=0):
    return emulate(frames, name="test", seed=seed, model={VEHICLE: class_model})


def along_errors(frames):
    heading = np.array([math.cos(TRUE_BOX[6]), math.sin(TRUE_BOX[6])])
    return np.array([(boxes[0, :2] - TRUE_BOX[:2]) @ heading for _, boxes in frames])


def test_linear_lag1_known_values():
    assert linear_lag1(0.0) == 0.0
    assert math.isclose(linear_lag1(0.5), 2 * math.sin(math.pi / 12), rel_tol=1e-12)
    assert math.isclose(linear_lag1(1.0), 1.0, rel_tol=1e-12)


def test_range_band_is_left_closed():
    assert range_band(np.array([0.0, 29.999, 30.0, 49.999, 50.0, 80.0])).tolist() == [0, 0, 1, 1, 2, 2]


def test_zero_errors_are_the_identity():
    for classes, boxes in run(one_vehicle(5), model()):
        assert classes.tolist() == [VEHICLE]
        assert np.array_equal(boxes, TRUE_BOX[None, :])


def test_each_component_lands_on_its_box_field():
    # A constant error per component (every percentile equal): along +1 m on
    # the heading, across +2 m left of it, z +3, yaw +0.1, length +0.4,
    # width +0.5, height +0.6.
    table = percentiles(along=lambda p: 1.0 + 0 * p, across=lambda p: 2.0 + 0 * p, z=lambda p: 3.0 + 0 * p,
                        yaw=lambda p: 0.1 + 0 * p, length=lambda p: 0.4 + 0 * p, width=lambda p: 0.5 + 0 * p,
                        height=lambda p: 0.6 + 0 * p)
    (_, boxes), = run(one_vehicle(1), model(table))
    c, s = math.cos(TRUE_BOX[6]), math.sin(TRUE_BOX[6])
    expected = TRUE_BOX + np.array([1.0 * c - 2.0 * s, 1.0 * s + 2.0 * c, 3.0, 0.4, 0.5, 0.6, 0.1])
    assert np.allclose(boxes[0], expected, atol=1e-12)


def test_marginal_quantiles_are_recovered():
    # A skewed marginal: along error = -ln(1 - p) - 1 (a shifted exponential),
    # independent over time (rank lag-1 0). The sample quantile at level q has
    # standard error sqrt(q (1 - q) / n) / f(Q(q)), f the exponential density.
    n = 20000
    quantile = lambda p: -np.log1p(-np.minimum(p, 1 - 1e-9)) - 1.0
    errors = along_errors(run(one_vehicle(n), model(percentiles(along=quantile))))
    for q in (0.05, 0.5, 0.95):
        density = 1.0 - q  # exponential density at its q quantile
        se = math.sqrt(q * (1 - q) / n) / density
        # Linear interpolation between 1 % percentiles adds at most the
        # quantile's curvature over a 1 % step: below 0.01 m up to q = 0.95.
        assert abs(np.quantile(errors, q) - quantile(np.array([q]))[0]) < 4 * se + 0.01


def test_rank_persistence_is_recovered():
    # Rank lag-1 0.9 on along. The Spearman estimate of a Gaussian AR(1)'s
    # lag-1 has standard error close to Pearson's, sqrt((1 - r^2) / n) with
    # r = linear_lag1(0.9); 1.5x allows for the rank transform.
    n, rank = 20000, 0.9
    errors = along_errors(run(one_vehicle(n), model(percentiles(along=lambda p: 2 * p - 1),
                                                    rank_lag1=[rank, 0, 0, 0, 0, 0, 0])))
    r = linear_lag1(rank)
    assert abs(spearmanr(errors[:-1], errors[1:])[0] - rank) < 1.5 * 4 * math.sqrt((1 - r**2) / n)


def test_cross_correlation_is_recovered():
    # Rank correlation 0.6 between along and length, no persistence.
    n = 20000
    corr = np.eye(7)
    corr[ALONG, LENGTH] = corr[LENGTH, ALONG] = 0.6
    table = percentiles(along=lambda p: 2 * p - 1, length=lambda p: 2 * p - 1)
    frames = run(one_vehicle(n), model(table, rank_corr=corr))
    along = along_errors(frames)
    length = np.array([boxes[0, 3] - TRUE_BOX[3] for _, boxes in frames])
    assert abs(spearmanr(along, length)[0] - 0.6) < 1.5 * 4 * math.sqrt((1 - linear_lag1(0.6) ** 2) / n)


def test_a_model_whose_innovation_is_not_a_covariance_is_rejected():
    # rho = (0.99, 0) with correlation 0.99 between the two components makes
    # C - D C D = [[0.0199, 0.99], [0.99, 1]], whose determinant is negative.
    corr = np.eye(7)
    corr[ALONG, ACROSS] = corr[ACROSS, ALONG] = 0.99
    bad = model(rank_lag1=[0.99, 0, 0, 0, 0, 0, 0], rank_corr=corr)
    try:
        validate_model({VEHICLE: bad})
    except ValueError as error:
        assert "class 1" in str(error)
    else:
        raise AssertionError("validate_model must reject a non-PSD innovation covariance")


def chain_check(states, p_on_after_off, p_on_after_on):
    states = np.asarray(states, dtype=np.int64)
    n = len(states)
    stationary = p_on_after_off / (p_on_after_off + 1 - p_on_after_on)
    lam = p_on_after_on - p_on_after_off
    n_eff = n * (1 - lam) / (1 + lam)
    assert abs(states.mean() - stationary) < 4 * math.sqrt(stationary * (1 - stationary) / n_eff)
    on = states[:-1] == 1
    assert abs(states[1:][on].mean() - p_on_after_on) < 4 * math.sqrt(p_on_after_on * (1 - p_on_after_on) / on.sum())


def test_miss_chain():
    frames = run(one_vehicle(30000), model(miss=(0.03, 0.85)))
    chain_check([len(boxes) == 0 for _, boxes in frames], 0.03, 0.85)


def test_flip_chain():
    frames = run(one_vehicle(30000), model(flips=(0.05, 0.6)))
    yaw_error = [abs(math.remainder(boxes[0, 6] - TRUE_BOX[6], 2 * math.pi)) for _, boxes in frames]
    assert all(e < 1e-9 or abs(e - math.pi) < 1e-9 for e in yaw_error)
    chain_check([e > math.pi / 2 for e in yaw_error], 0.05, 0.6)


def test_ghost_count_and_size():
    rate, survival, n = 3.0, 0.7, 20000
    frames = run(no_objects(n), model(fp=(rate, survival)))
    counts = np.array([len(boxes) for _, boxes in frames])
    assert abs(counts.mean() - rate) < 4 * math.sqrt(rate / (n * (1 - survival) / (1 + survival)))
    boxes = np.concatenate([b for _, b in frames if len(b)])
    assert np.allclose(boxes[:, 3:6], [4.0, 2.0, 1.5])
    assert np.allclose(boxes[:, 2], 0.75)  # vehicle-frame centre z = height / 2
    assert abs(np.median(np.hypot(boxes[:, 0], boxes[:, 1])) - 50.0) < 2.0


def test_same_seed_same_bytes_and_seed_matters():
    table = percentiles(along=lambda p: 2 * p - 1, yaw=lambda p: 0.2 * p - 0.1)
    m = model(table, miss=(0.05, 0.8), rank_lag1=[0.9] * 7, fp=(1.0, 0.5))
    first = run(one_vehicle(50), m, seed=0)
    again = run(one_vehicle(50), m, seed=0)
    other = run(one_vehicle(50), m, seed=1)
    assert all(np.array_equal(c1, c2) and np.array_equal(b1, b2) for (c1, b1), (c2, b2) in zip(first, again, strict=True))
    assert any(not np.array_equal(b1, b2) for (_, b1), (_, b2) in zip(first, other, strict=True))


def test_waymo_ids_above_int64_pass_through_staging():
    from stage_segment import emulate_inputs
    from waymo_boxes import Box, GroundTruthBox, stable_object_id

    name = next(f"object-{i}" for i in range(100) if stable_object_id(f"object-{i}") >= 2**63)
    box = Box(20.0, 0.0, 1.0, 4.5, 2.0, 1.6, 0.0)
    labels = [GroundTruthBox(t, name, VEHICLE, box, 100, None) for t in (0, 100_000)]
    poses = {t: IDENTITY_POSE.ravel().tolist() for t in (0, 100_000)}
    frames = emulate_inputs(labels, [0, 100_000], poses)
    assert frames[0][0].ids.tolist() == [stable_object_id(name)]
    assert len(emulate(frames, name="ids", seed=0, model={VEHICLE: model()})) == 2


def test_independent_model_removes_memory():
    # Class 1, band 1: the expected rates are a / (a + 1 - b) of the stored
    # numbers, written out here rather than taken from independent_model.
    model_in = default_model()
    vehicle = model_in[1]
    band = vehicle.bands[1]
    a, b = band.p_miss_after_hit, band.p_miss_after_miss
    flip_a, flip_b = vehicle.p_flip_after_no_flip, vehicle.p_flip_after_flip
    fp_survival_before = vehicle.fp_survival
    lag_before = vehicle.rank_lag1.copy()
    miss_rate = a / (a + 1 - b)
    flip_rate = flip_a / (flip_a + 1 - flip_b)

    result = independent_model(model_in)[1]

    assert np.array_equal(result.rank_lag1, np.zeros(7))
    band_out = result.bands[1]
    assert math.isclose(band_out.p_miss_after_hit, miss_rate, rel_tol=1e-12)
    assert math.isclose(band_out.p_miss_after_miss, miss_rate, rel_tol=1e-12)
    assert math.isclose(result.p_flip_after_no_flip, flip_rate, rel_tol=1e-12)
    assert math.isclose(result.p_flip_after_flip, flip_rate, rel_tol=1e-12)
    assert result.fp_survival == 0.0
    assert np.array_equal(model_in[1].rank_lag1, lag_before)
    assert model_in[1].bands[1].p_miss_after_hit == a
    assert model_in[1].bands[1].p_miss_after_miss == b
    assert model_in[1].fp_survival == fp_survival_before
