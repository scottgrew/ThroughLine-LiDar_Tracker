"""Tests for marks.py, run as plain functions by the Makefile test target."""

from marks import (
    Breakdown,
    FrameDetails,
    Totals,
    check_totals,
    mismatch_events,
    parse_details,
)


def _frame(timestamp, tp=(), fn=(), fp=()):
    return FrameDetails(timestamp, tuple(tp), tuple(fn), tuple(fp))


def test_eviction_counts_twice():
    # Arrange
    frames = [
        _frame(0, tp=[("o1", "h1"), ("o2", "h2")]),
        _frame(1, tp=[("o1", "h2")]),
    ]

    # Act
    events = mismatch_events(frames)

    # Assert
    assert events == [[], ["o1", "o1"]]


def test_evicted_gt_in_frame_is_a_miss_not_a_switch():
    # Arrange
    frames = [
        _frame(0, tp=[("o1", "h1"), ("o2", "h2")]),
        _frame(1, tp=[("o1", "h2")], fn=["o2"]),
    ]

    # Act
    events = mismatch_events(frames)

    # Assert
    assert events == [[], ["o1"]]


def test_continued_pair_is_not_a_switch():
    # Arrange
    frames = [_frame(0, tp=[("o1", "h1")]), _frame(1, tp=[("o1", "h1")])]

    # Act
    events = mismatch_events(frames)

    # Assert
    assert events == [[], []]


def test_same_track_after_a_gap_is_not_a_switch():
    # Arrange
    frames = [
        _frame(0, tp=[("o1", "h1")]),
        _frame(1, fn=["o1"], fp=["h1"]),
        _frame(2, tp=[("o1", "h1")]),
    ]

    # Act
    events = mismatch_events(frames)

    # Assert
    assert events == [[], [], []]


def test_new_track_after_a_gap_is_a_switch():
    # Arrange
    frames = [
        _frame(0, tp=[("o1", "h1")]),
        _frame(1, fn=["o1"]),
        _frame(2, tp=[("o1", "h3")]),
    ]

    # Act
    events = mismatch_events(frames)

    # Assert
    assert events == [[], [], ["o1"]]


def test_parse_round_trip():
    # Arrange
    lines = [
        "T\tVEHICLE\t2\t1\t0\t2\t4\n",
        "F\tVEHICLE\t100\tgt1=p1\tgt2\tp2\n",
        "F\tVEHICLE\t200\tgt1=p1\t\t\n",
        "F\tVEHICLE\t300\t\tgt2\t\n",
    ]
    expected = {
        "VEHICLE": Breakdown(
            Totals(misses=2, fps=1, mismatches=0, matches=2, objects=4),
            (
                FrameDetails(100, (("gt1", "p1"),), ("gt2",), ("p2",)),
                FrameDetails(200, (("gt1", "p1"),), (), ()),
                FrameDetails(300, (), ("gt2",), ()),
            ),
        ),
    }

    # Act
    parsed = parse_details(lines)

    # Assert
    assert parsed == expected


def test_parse_rejects():
    # Arrange
    bad_inputs = [
        ["X\tVEHICLE\t0"],
        ["T\tVEHICLE\t1\t1"],
        ["T\tVEHICLE\t0\t0\t0\t0\t0", "F\tVEHICLE\t1\tgt1\t\t"],
        ["T\tVEHICLE\t0\t0\t0\t0\t0", "F\tVEHICLE\t1\ta=b=c\t\t"],
        ["F\tVEHICLE\t1\t\t\t"],
        [
            "T\tVEHICLE\t0\t0\t0\t0\t0",
            "F\tVEHICLE\t2\t\t\t",
            "F\tVEHICLE\t2\t\t\t",
        ],
    ]

    # Act and assert
    for lines in bad_inputs:
        try:
            parse_details(lines)
        except ValueError:
            continue
        else:
            raise AssertionError(f"parse_details accepted {lines!r}")


def test_check_totals_rejects_an_off_by_one():
    # Arrange
    frames = (
        _frame(0, tp=[("o1", "h1"), ("o2", "h2")]),
        _frame(1, tp=[("o1", "h2")]),
    )
    correct = Breakdown(Totals(0, 0, 2, 3, 3), frames)
    off_by_one = Breakdown(Totals(0, 0, 3, 3, 3), frames)

    # Act
    check_totals("VEHICLE", correct)

    # Assert
    try:
        check_totals("VEHICLE", off_by_one)
    except ValueError as error:
        assert "mismatches" in str(error)
    else:
        raise AssertionError("check_totals accepted an off-by-one mismatch")
