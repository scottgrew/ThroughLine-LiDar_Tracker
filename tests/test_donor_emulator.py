"""Oracle tests for the donor library and donor emulator, with expected values
worked out by hand, not from the implementation.

Statistical checks allow 4 standard errors (SE = sqrt(p (1 - p) / n) for a
share, the ghost bound of test_ghost_count_and_size for the ghost count).
Emulator tests build the DonorLibrary by hand and fix every seed. Vehicle
errors are read as detection x minus label x: boxes have yaw 0, so the x
shift is the along error.
"""

import dataclasses
import math
import tempfile
from pathlib import Path

import numpy as np
from detector_error import FrameInput
from donor_emulator import build_index, emulate_donor, next_row
from donor_library import (
    DonorLibrary,
    build_library,
    exclude_segment,
    read_library,
    validate_library,
    write_library,
)
from emulate_detector import BandModel, ClassModel, Objects

VEHICLE = 1
SLOT = 0  # class code 1 -> slot 0
IDENTITY = np.eye(4)
EDGE_TOL = 1e-6  # donor errors are stored as float32
LABEL_BOX = np.array([20.0, 0.0, 1.0, 4.5, 2.0, 1.6, 0.7])  # x y z l w h yaw


# ---------------------------------------------------------------- library


def car_frame(frame, label_rows, det_rows):
    """FrameInput for pose identity; rows are (id, box) and (box,)."""
    label_ids = np.array([r[0] for r in label_rows], dtype=np.int64)
    label_boxes = np.array([r[1] for r in label_rows]).reshape(-1, 7)
    det_boxes = np.array(det_rows).reshape(-1, 7)
    return FrameInput(
        "s", frame, IDENTITY,
        (np.full(len(det_boxes), VEHICLE), det_boxes),
        (label_ids, np.full(len(label_ids), VEHICLE), label_boxes),
    )


def box(x, y, yaw=0.0):
    return [x, y, 1.0, 4.0, 2.0, 1.5, yaw]


def case_a_frames():
    """Vehicle 1 at (10, 0) frames 0-3, detection offsets (0.1, 0.2),
    (0.3, 0), none, (-0.1, 0.1). Vehicle 2 at (40, 0), yaw 0.3, frames 0, 1,
    3, detection yaw = label + pi - 0.1 (a flip), no position offset."""
    flipped = 0.3 + math.pi - 0.1
    a_offsets = {0: (0.1, 0.2), 1: (0.3, 0.0), 3: (-0.1, 0.1)}
    frames = []
    for frame in range(4):
        labels = [(1, box(10.0, 0.0))]
        dets = []
        if frame in a_offsets:
            dx, dy = a_offsets[frame]
            dets.append(box(10.0 + dx, dy))
        if frame != 2:
            labels.append((2, box(40.0, 0.0, 0.3)))
            dets.append(box(40.0, 0.0, flipped))
        frames.append(car_frame(frame, labels, dets))
    return frames


def test_build_library_rows_and_flags():
    library = build_library(case_a_frames())
    # Sort order (segment, cls, str(id), frame): a0..a3, b0, b1, b3.
    assert library.segments == ("s",)
    assert library.segment.tolist() == [0] * 7
    assert library.cls.tolist() == [VEHICLE] * 7
    assert library.has_next.tolist() == [
        True, True, True, False, True, False, False]
    assert library.missed.tolist() == [
        False, False, True, False, False, False, False]
    assert library.band.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert library.flip.tolist() == [False] * 4 + [True] * 3
    # err in COMPONENTS order; yaw with the flip removed is -0.1 for b;
    # the missed row is all zeros.
    expected = np.zeros((7, 7))
    expected[0, :2] = (0.1, 0.2)
    expected[1, :2] = (0.3, 0.0)
    expected[3, :2] = (-0.1, 0.1)
    expected[4:, 3] = -0.1
    assert np.allclose(library.err, expected, atol=EDGE_TOL)


def test_scale_is_half_p16_p84_spread():
    # Along errors k/50 - 1 for k = 0..100 (one vehicle, 101 frames). The
    # 16th and 84th percentiles sit on integer positions 16 and 84:
    # -0.68 and +0.68, so scale = (0.68 + 0.68) / 2 = 0.68. Every other
    # component is constant 0 (spread < 1e-9) and every other (slot, band)
    # has no rows: all those scales are 1.0.
    frames = [
        car_frame(k, [(1, box(10.0, 0.0))], [box(10.0 + k / 50 - 1, 0.0)])
        for k in range(101)
    ]
    scale = build_library(frames).scale
    expected = np.ones((3, 3, 7))
    expected[SLOT, 0, 0] = 0.68
    assert scale.shape == (3, 3, 7)
    assert np.allclose(scale, expected, atol=1e-9)


# ------------------------------------------------- hand-built libraries


def chain(band, alongs=None, missed=None, flip=False, segment="s", errs=None):
    """Rows of one object over consecutive frames; every row but the last
    has a next frame. Each row is (segment, band, missed, flip, err, next)."""
    if errs is None:
        errs = [(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for a in alongs]
    count = len(errs)
    missed = [False] * count if missed is None else missed
    return [
        (segment, band, missed[i], flip, errs[i], i < count - 1)
        for i in range(count)
    ]


def make_library(*chains, fill=True):
    """DonorLibrary of vehicle chains. A (segment, band) with no chain gets
    a filler chain of zero errors appended after all given rows, so row
    numbers of the given chains are their order."""
    rows = [row for one in chains for row in one]
    names = sorted({row[0] for row in rows})
    present = {(row[0], row[1]) for row in rows}
    for name in names if fill else ():
        for band in range(3):
            if (name, band) not in present:
                rows += chain(band, [0.0, 0.0], segment=name)
    segment, band, missed, flip, err, has_next = zip(*rows, strict=True)
    return DonorLibrary(
        segments=tuple(names),
        segment=np.array([names.index(s) for s in segment], np.int16),
        cls=np.full(len(rows), VEHICLE, np.int8),
        band=np.array(band, np.int8),
        missed=np.array(missed, bool),
        flip=np.array(flip, bool),
        err=np.array(err, np.float32),
        has_next=np.array(has_next, bool),
        scale=np.ones((3, 3, 7)),
    )


def test_library_round_trip():
    library = make_library(
        chain(0, [0.1, 0.2, 0.3], missed=[False, True, False]),
        chain(1, [0.4, 0.5], flip=True),
        chain(2, [0.6, 0.7], segment="t"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "library.npz"
        write_library(library, path)
        loaded = read_library(path)
    assert loaded.segments == library.segments
    for field in dataclasses.fields(DonorLibrary):
        if field.name == "segments":
            continue
        got, want = getattr(loaded, field.name), getattr(library, field.name)
        assert got.dtype == want.dtype, field.name
        assert np.array_equal(got, want), field.name


def test_validate_names_class_and_band_without_donor_rows():
    # Vehicle rows in bands 0 and 1 only: band 2 has no donor rows.
    library = make_library(
        chain(0, [0.1, 0.2]), chain(1, [0.1, 0.2]), fill=False)
    try:
        validate_library(library)
    except ValueError as error:
        assert "class 1" in str(error) and "band 2" in str(error), error
    else:
        raise AssertionError("validate_library must reject class 1 band 2")


def test_exclude_segment_keeps_other_rows_and_has_next():
    library = make_library(
        chain(0, [0.1, 0.2], segment="s"),
        chain(0, [0.7, 0.8, 0.9], missed=[False, True, False], segment="t"),
    )
    kept = exclude_segment(library, "s")
    keep = np.array(library.segments)[library.segment] == "t"
    names = np.array(kept.segments)[kept.segment]
    assert names.tolist() == ["t"] * int(keep.sum())
    for field in ("cls", "band", "missed", "flip", "err", "has_next"):
        assert np.array_equal(
            getattr(kept, field), getattr(library, field)[keep]), field


# ------------------------------------------------------------- emulator


def ghostless_model(fp=(0.0, 0.0)):
    """A copula ClassModel with no object error (only ghosts may appear)."""
    band = BandModel(0.0, 0.0, np.zeros((7, 101)))
    return {VEHICLE: ClassModel(
        bands=(band,) * 3,
        rank_lag1=np.zeros(7),
        rank_corr=np.eye(7),
        p_flip_after_no_flip=0.0, p_flip_after_flip=0.0,
        fp_per_frame=fp[0], fp_survival=fp[1],
        fp_range_deciles_m=tuple(float(v) for v in range(0, 101, 10)),
        fp_size_m=(4.0, 2.0, 1.5),
    )}


def vehicles(xs, yaw=0.0):
    """One frame of static vehicles at (x, 0), ids 1..m, as Objects."""
    count = len(xs)
    boxes = np.tile([0.0, 0.0, 1.0, 4.5, 2.0, 1.6, yaw], (count, 1))
    boxes[:, 0] = xs
    return Objects(
        ids=np.arange(1, count + 1, dtype=np.uint64),
        classes=np.full(count, VEHICLE, np.int64),
        boxes=boxes,
    )


def nothing():
    return Objects(
        ids=np.zeros(0, np.uint64), classes=np.zeros(0, np.int64),
        boxes=np.zeros((0, 7)))


def run(library, frames, seed=0, model=None, neighbours=None):
    """emulate_donor over [(Objects, pose)] frames with a hand library."""
    args = () if neighbours is None else (neighbours,)
    index = build_index(library, *args)
    pairs = [(objects, IDENTITY) for objects in frames]
    return emulate_donor(
        pairs, "test", seed, index,
        ghostless_model() if model is None else model)


def along_series(out, x):
    """Per-frame along error of the single vehicle at x (nan if missed)."""
    return np.array([
        boxes[0, 0] - x if len(boxes) else np.nan for _, boxes in out])


def gap_to(values, allowed):
    """Distance from each value to the nearest allowed value."""
    return np.abs(
        np.asarray(values)[:, None] - np.asarray(allowed)[None, :]
    ).min(axis=1)


def tenths(values):
    """Index k - 1 for an error of k / 10."""
    return np.rint(np.asarray(values) * 10).astype(int) - 1


def test_history_chain_follows_donor_or_hops():
    # Band-0 history 0.1..0.5. From 0.1-0.4 the next row is the donor's
    # next (+0.1); 0.5 has no next, so it hops to the successor of a hit
    # row with a next (0.1..0.4): one of 0.2..0.5.
    library = make_library(chain(0, [0.1, 0.2, 0.3, 0.4, 0.5]))
    out = run(library, [vehicles([10.0])] * 200)
    errors = along_series(out, 10.0)
    steps = tenths(errors)
    allowed = np.zeros((5, 5), bool)
    allowed[np.arange(4), np.arange(1, 5)] = True
    allowed[4, 1:] = True
    assert errors.shape == (200,)
    assert np.allclose(errors, (steps + 1) / 10, atol=EDGE_TOL)
    assert allowed[steps[:-1], steps[1:]].all()


def test_hop_with_one_neighbour_takes_the_nearest_donor_state():
    # Rows 0: 0.10, 1: 0.50 | 2: 0.45, 3: 0.90. Hit rows with a next in
    # band: 0 and 2. Chain ends hop to another chain; eps 0.0 is the analog
    # hop only. Row 1 (0.50): nearest other-chain hit is row 2 (0.45), so
    # successor 3. Row 3 (0.90): nearest hit is row 2, same chain, dropped;
    # the next other-chain hit is row 0, so successor 1.
    library = make_library(chain(0, [0.10, 0.50]), chain(0, [0.45, 0.90]))
    index = build_index(library, neighbours=1, eps=0.0)
    rng = np.random.default_rng(0)
    hops = [next_row(index, SLOT, 0, row, rng) for row in (1, 3)]
    assert hops == [3, 1]


def test_hit_miss_miss_hit_history_hops_within_its_only_chain():
    # Rows 0 hit(0.1) -> 1 miss -> 2 miss -> 3 hit(0.2), one chain. Row 3 has
    # no next and no other chain exists, so it falls back to uniform over ANY
    # = rows 0, 1, 2, landing on successor 1, 2 or 3. The hit cadence is
    # therefore random, not every third frame.
    library = make_library(
        chain(0, [0.1, 0.0, 0.0, 0.2], missed=[False, True, True, False]))
    index = build_index(library)
    draws = [
        next_row(index, SLOT, 0, 3, np.random.default_rng(seed))
        for seed in range(30)
    ]
    assert np.isin(draws, [1, 2, 3]).all()


def test_band_change_hops_to_a_donor_of_the_new_band():
    # Band 0 history 0.1 -> 0.2 and band 1 history 0.15 -> 0.25. The object
    # is at 20 m (band 0) for frames 0-9, then 40 m (band 1). At frame 9
    # the band-0 chain sits on 0.2 (row with no next). The only band-1 hit
    # row with a next in band is 0.15, so frame 10 is its successor 0.25.
    library = make_library(chain(0, [0.1, 0.2]), chain(1, [0.15, 0.25]))
    frames = [vehicles([20.0])] * 10 + [vehicles([40.0])] * 10
    out = run(library, frames)
    errors = np.concatenate([
        along_series(out[:10], 20.0), along_series(out[10:], 40.0)])
    assert gap_to(errors[:10], [0.1, 0.2]).max() < EDGE_TOL
    assert gap_to(errors[10:], [0.15, 0.25]).max() < EDGE_TOL
    assert abs(errors[10] - 0.25) < EDGE_TOL


def test_missed_without_missed_candidates_falls_back_to_any_row():
    # Rows 0 hit 0.1, 1 hit 0.2, 2 missed with no next. MISSED candidates
    # (missed row with a next in band): none. Prev row 2 is missed, so the
    # draw is uniform over ANY = rows 0, 1 ("a row in ANY"), and
    # the hop lands on successor 1 or 2.
    library = make_library(
        chain(0, [0.1, 0.2, 0.0], missed=[False, False, True]))
    index = build_index(library)
    draws = [
        next_row(index, SLOT, 0, 2, np.random.default_rng(seed))
        for seed in range(30)
    ]
    assert np.isin(draws, [1, 2]).all()


def test_start_row_is_uniform_and_ids_absent_a_frame_restart():
    # Band 2 has rows 0.1, 0.2 (history 1) and 0.3, 0.4 (history 2): every
    # start is uniform over the four rows, share 0.25 each. 2000 ids give
    # SE = sqrt(0.25 * 0.75 / 2000); the bound is 4 SE. Frame 1 is empty
    # (ids absent), so frame 2 must restart uniformly, not continue.
    library = make_library(chain(2, [0.1, 0.2]), chain(2, [0.3, 0.4]))
    count = 2000
    full = vehicles([60.0] * count)
    out = run(library, [full, nothing(), full])
    bound = 4 * math.sqrt(0.25 * 0.75 / count)
    shares = np.array([
        np.bincount(tenths(boxes[:, 0] - 60.0), minlength=4) / count
        for boxes in (out[0][1], out[2][1])
    ])
    assert out[1][0].dtype == np.int64
    assert out[1][0].shape == (0,) and out[1][1].shape == (0, 7)
    assert np.abs(shares - 0.25).max() < bound


def test_range_bands_are_left_closed():
    # Band errors: band 0 -> 0.1, band 1 -> 0.2, band 2 -> 0.3. Ranges
    # 29.999 | 30.0 | 49.999 | 50.0 land in bands 0 | 1 | 1 | 2.
    library = make_library(
        chain(0, [0.1, 0.1]), chain(1, [0.2, 0.2]), chain(2, [0.3, 0.3]))
    xs = np.array([29.999, 30.0, 49.999, 50.0])
    out = run(library, [vehicles(xs)] * 10)
    errors = np.array([boxes[:, 0] - xs for _, boxes in out])
    expected = np.tile([0.1, 0.2, 0.2, 0.3], (10, 1))
    assert np.allclose(errors, expected, atol=EDGE_TOL)


def library_with_error(err, flip=False):
    return make_library(chain(0, errs=[err, err], flip=flip))


def one_box(library):
    (_, boxes), = run(library, [vehicles([20.0], yaw=0.7)])
    return boxes[0]


def test_donor_error_lands_on_box_fields():
    # Error along 1, across 2, z 3, yaw 0.1, length 0.4, width 0.5,
    # height 0.6 on a yaw-0.7 box at (20, 0): x shifts by 1 cos - 2 sin,
    # y by 1 sin + 2 cos (heading frame to vehicle frame).
    c, s = math.cos(0.7), math.sin(0.7)
    library = library_with_error((1.0, 2.0, 3.0, 0.1, 0.4, 0.5, 0.6))
    expected = LABEL_BOX + np.array(
        [1.0 * c - 2.0 * s, 1.0 * s + 2.0 * c, 3.0, 0.4, 0.5, 0.6, 0.1])
    assert np.allclose(one_box(library), expected, atol=EDGE_TOL)


def test_flip_adds_pi_and_wraps():
    # Flip adds pi to the stored yaw error 0.1: 0.7 + 0.1 + pi = 0.8 + pi,
    # which wraps to 0.8 - pi (about -2.3416).
    library = library_with_error((0.0,) * 3 + (0.1,) + (0.0,) * 3, flip=True)
    expected = LABEL_BOX.copy()
    expected[6] = 0.8 - math.pi
    assert np.allclose(one_box(library), expected, atol=EDGE_TOL)


def test_negative_length_error_floors_at_a_tenth():
    # 4.5 - 10 < 0.1, so length is 0.1; the other fields keep the label.
    library = library_with_error((0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0))
    expected = LABEL_BOX.copy()
    expected[3] = 0.1
    assert np.allclose(one_box(library), expected, atol=EDGE_TOL)


def test_hop_never_returns_to_its_own_chain():
    # Chain A rows 0-2 (0.1-0.3), chain B rows 3-4 (0.4, 0.5). From the end
    # of A (row 2) the only successor in B is row 4; from the end of B (row
    # 4) the successors in A are rows 1 and 2. eps 0.0 is analog only, eps
    # 1.0 uniform only.
    library = make_library(chain(0, [0.1, 0.2, 0.3]), chain(0, [0.4, 0.5]))
    for eps in (0.0, 1.0):
        index = build_index(library, eps=eps)
        for seed in range(30):
            rng = np.random.default_rng(seed)
            assert next_row(index, SLOT, 0, 2, rng) == 4
            assert next_row(index, SLOT, 0, 4, rng) in (1, 2)


def test_eps_bounds():
    library = make_library(chain(0, [0.1, 0.2]))
    for eps in (-0.1, 1.1):
        try:
            build_index(library, eps=eps)
        except ValueError:
            continue
        raise AssertionError(f"eps {eps} must raise")


def test_walk_covers_every_successor_row():
    # 10 chains of 3 band-0 rows. Rows 1 and 2 of each chain are successors
    # of a continuable row; 3000 steps from a uniform start must visit all.
    chains = [
        chain(0, [0.01 * (3 * c + j + 1) for j in range(3)])
        for c in range(10)
    ]
    library = make_library(*chains)
    successors = [
        row + 1 for row in range(len(library.band) - 1)
        if library.has_next[row] and library.band[row] == 0
    ]
    index = build_index(library)
    rng = np.random.default_rng(0)
    row = None
    visited = []
    for _ in range(3000):
        row = next_row(index, SLOT, 0, row, rng)
        visited.append(row)
    assert np.isin(successors, visited).all()


def test_same_seed_same_bytes_and_seed_matters():
    library = make_library(
        chain(0, [0.1, 0.2, 0.3], missed=[False, True, False]),
        chain(0, [0.4, 0.5, 0.6]))
    model = ghostless_model(fp=(1.0, 0.5))
    frames = [vehicles([10.0])] * 60

    def flat(seed):
        out = run(library, frames, seed=seed, model=model)
        return np.concatenate([boxes.ravel() for _, boxes in out])

    assert np.array_equal(flat(0), flat(0))
    assert not np.array_equal(flat(0), flat(1))


def test_excluded_segment_is_never_emitted():
    # Segment s donates 0.1, 0.2; segment t donates 0.7, 0.8 (and zero
    # filler errors in the bands it lacks, unused at 10 m). Excluding s,
    # a 10 m object emits only t's errors, whatever the seed.
    library = make_library(
        chain(0, [0.1, 0.2], segment="s"),
        chain(0, [0.7, 0.8], segment="t"))
    held_out = exclude_segment(library, "s")
    out = run(held_out, [vehicles([10.0])] * 200)
    errors = along_series(out, 10.0)
    assert gap_to(errors, [0.7, 0.8]).max() < EDGE_TOL


def test_excluding_every_row_raises():
    library = make_library(chain(0, [0.1, 0.2]))
    try:
        build_index(exclude_segment(library, "s"))
    except ValueError:
        return
    raise AssertionError("an index over an empty library must raise")


def test_ghost_count_on_empty_frames():
    # Same bound as test_ghost_count_and_size: rate 3.0, survival 0.7,
    # effective n = frames (1 - survival) / (1 + survival).
    rate, survival, frames = 3.0, 0.7, 20000
    library = make_library(chain(0, [0.1, 0.2]))
    out = run(
        library, [nothing()] * frames, model=ghostless_model((rate, survival)))
    counts = np.array([len(boxes) for _, boxes in out])
    bound = 4 * math.sqrt(rate / (frames * (1 - survival) / (1 + survival)))
    assert abs(counts.mean() - rate) < bound
