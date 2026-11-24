"""Turns the per-frame matches printed by compute_tracking_details_main into
marks for the tracking view.

The binary prints, for each breakdown, its totals and the matched, missed and
false-positive ids of every frame. parse_details reads that output. Waymo's
evaluator does not print ID switches, so mismatch_events replays its mismatch
rule over the frames, and check_totals confirms the replay reproduces the
binary's totals. main writes one CSV row per label, miss, false positive and
ID switch, carrying the box of the object that the mark sits on. A label is
a ground-truth object the frame matched or missed, so the view can show
every object a mark refers to.
"""

import argparse
import csv
from collections.abc import Iterable, Sequence
from typing import NamedTuple

MARK_COLUMNS = (
    "frame_timestamp_micros",
    "kind",
    "object_type",
    "object_id",
    "center_x",
    "center_y",
    "center_z",
    "length",
    "width",
    "height",
    "yaw",
)
TOTAL_TAG = "T"
FRAME_TAG = "F"
TOTAL_FIELD_COUNT = 7
FRAME_FIELD_COUNT = 6


class Totals(NamedTuple):
    misses: int
    fps: int
    mismatches: int
    matches: int
    objects: int


class FrameDetails(NamedTuple):
    timestamp: int
    tp: tuple[tuple[str, str], ...]
    fn: tuple[str, ...]
    fp: tuple[str, ...]


class Breakdown(NamedTuple):
    totals: Totals
    frames: tuple[FrameDetails, ...]


def _line_error(number: int, message: str) -> ValueError:
    return ValueError(f"line {number}: {message}")


def _integer(text: str, number: int) -> int:
    try:
        return int(text)
    except ValueError:
        raise _line_error(number, f"{text!r} is not an integer") from None


def _ids(field: str) -> tuple[str, ...]:
    return tuple(field.split(" ")) if field else ()


def _pair(entry: str, number: int) -> tuple[str, str]:
    parts = entry.split("=")
    if len(parts) != 2:
        raise _line_error(number, f"tp entry {entry!r} is not gt=pred")
    return (parts[0], parts[1])


def _total_line(fields: list[str], number: int) -> tuple[str, Totals]:
    if len(fields) != TOTAL_FIELD_COUNT:
        raise _line_error(number, "a T line needs 7 fields")
    totals = Totals(*(_integer(text, number) for text in fields[2:]))
    return fields[1], totals


def _frame_line(fields: list[str], number: int, prior: list[FrameDetails]):
    if len(fields) != FRAME_FIELD_COUNT:
        raise _line_error(number, "an F line needs 6 fields")
    timestamp = _integer(fields[2], number)
    if prior and timestamp <= prior[-1].timestamp:
        raise _line_error(number, f"timestamp {timestamp} is not ascending")
    tp = tuple(_pair(entry, number) for entry in fields[3].split(" ")
               if entry)
    return fields[1], FrameDetails(timestamp, tp, _ids(fields[4]),
                                   _ids(fields[5]))


def parse_details(lines: Iterable[str]) -> dict[str, Breakdown]:
    """Returns the breakdowns in the binary's output, keyed by name.

    A T line must come before the F lines of its breakdown, and the F lines
    of one breakdown must have strictly ascending timestamps.
    """
    totals: dict[str, Totals] = {}
    frames: dict[str, list[FrameDetails]] = {}
    for number, line in enumerate(lines, start=1):
        fields = line.rstrip("\n").split("\t")
        if fields[0] == TOTAL_TAG:
            name, breakdown_totals = _total_line(fields, number)
            if name in totals:
                raise _line_error(number, f"{name} has two T lines")
            totals[name] = breakdown_totals
            frames[name] = []
        elif fields[0] == FRAME_TAG:
            if len(fields) < 2 or fields[1] not in totals:
                raise _line_error(number, "an F line has no T line before it")
            name, frame = _frame_line(fields, number, frames[fields[1]])
            frames[name].append(frame)
        else:
            raise _line_error(number, f"unknown record tag {fields[0]!r}")
    return {
        name: Breakdown(totals[name], tuple(frames[name])) for name in totals
    }


def mismatch_events(frames: Sequence[FrameDetails]) -> list[list[str]]:
    """Returns, per frame, the gt ids that Waymo's evaluator charges an ID
    switch to in that frame.

    This replays MOT::Eval and MOT::ComputeNumMismatches. gt_to_pred and
    pred_to_gt hold the mapping kept from the previous frame.
    """
    gt_to_pred: dict[str, str] = {}
    pred_to_gt: dict[str, str] = {}
    events: list[list[str]] = []
    for frame in frames:
        new = dict(frame.tp)
        continued = {
            gt for gt, pred in frame.tp if gt_to_pred.get(gt) == pred
        }
        present = ({gt for gt, _ in frame.tp} | set(frame.fn)) - continued
        frame_events: list[str] = []
        for gt, pred in frame.tp:
            if gt in gt_to_pred and gt_to_pred[gt] != pred:
                frame_events.append(gt)
            # Waymo's Mismatch_GT_Eviction: t0 o1->h1, o2->h2; t1 o1->h2
            # counts 2. o1 moves from h1 to h2, and the prediction h2 is
            # taken from o2, which is absent from t1's matches, so the
            # eviction is a second mismatch charged to o1. An o2 still in
            # the frame as a miss is not charged (Missmatch_Miss_GT_Eviction).
            if (pred in pred_to_gt and pred_to_gt[pred] not in present
                    and pred_to_gt[pred] not in new):
                frame_events.append(gt)
        events.append(frame_events)
        evicted = {
            pred_to_gt[pred] for pred in new.values()
            if pred in pred_to_gt and pred_to_gt[pred] not in new
        }
        for gt in evicted:
            del gt_to_pred[gt]
        gt_to_pred.update(new)
        pred_to_gt = {pred: gt for gt, pred in gt_to_pred.items()}
    return events


def check_totals(name: str, breakdown: Breakdown) -> None:
    """Raises ValueError unless the replay of the frames reproduces the totals
    that the binary printed for the breakdown."""
    frames = breakdown.frames
    replayed = {
        "misses": sum(len(frame.fn) for frame in frames),
        "fps": sum(len(frame.fp) for frame in frames),
        "mismatches": sum(len(events) for events in mismatch_events(frames)),
        "matches": sum(len(frame.tp) for frame in frames),
    }
    for field, value in replayed.items():
        printed = getattr(breakdown.totals, field)
        if value != printed:
            raise ValueError(
                f"{name}: {field} replays as {value} but the binary "
                f"printed {printed}")


def _mark_row(kind: str, timestamp: int, object_id: str, boxes: dict) -> tuple:
    waymo_object = boxes[(timestamp, object_id)]
    box = waymo_object.object.box
    return (timestamp, kind, waymo_object.object.type, object_id,
            box.center_x, box.center_y, box.center_z, box.length, box.width,
            box.height, box.heading)


def mark_rows(breakdowns: dict, gt_boxes: dict, pred_boxes: dict) -> list:
    """Returns one row per label, miss, false positive and ID switch, sorted
    by timestamp, kind and id.

    The boxes are Waymo's metrics_pb2.Object protos keyed by (timestamp,
    object id). Every breakdown's totals are checked before any row is built.
    """
    for name, breakdown in breakdowns.items():
        check_totals(name, breakdown)
    rows: list = []
    for breakdown in breakdowns.values():
        frames = breakdown.frames
        switches = mismatch_events(frames)
        for frame, switched in zip(frames, switches, strict=True):
            rows += [_mark_row("label", frame.timestamp, gt_id, gt_boxes)
                     for gt_id, _ in frame.tp]
            rows += [_mark_row("label", frame.timestamp, gt_id, gt_boxes)
                     for gt_id in frame.fn]
            rows += [_mark_row("miss", frame.timestamp, gt_id, gt_boxes)
                     for gt_id in frame.fn]
            rows += [_mark_row("false_positive", frame.timestamp, pred_id,
                               pred_boxes) for pred_id in frame.fp]
            rows += [_mark_row("id_switch", frame.timestamp, gt_id, gt_boxes)
                     for gt_id in switched]
    rows.sort(key=lambda row: (row[0], row[1], row[3]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", required=True)
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    # The protobuf and pandas imports stay here so that the tests can import
    # this module without them.
    import to_waymo_objects

    with open(arguments.details, encoding="utf-8") as stream:
        breakdowns = parse_details(stream)
    ground_truth = to_waymo_objects.build_ground_truth_objects(
        arguments.parquet_root, arguments.segment)
    predictions = to_waymo_objects.build_prediction_objects(
        arguments.tracks, arguments.segment)
    gt_boxes = {(o.frame_timestamp_micros, o.object.id): o
                for o in ground_truth.objects}
    pred_boxes = {(o.frame_timestamp_micros, o.object.id): o
                  for o in predictions.objects}

    rows = mark_rows(breakdowns, gt_boxes, pred_boxes)
    with open(arguments.out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(MARK_COLUMNS)
        writer.writerows(rows)

    for name, breakdown in breakdowns.items():
        totals = breakdown.totals
        print(f"verified {name}: misses {totals.misses} fps {totals.fps} "
              f"mismatches {totals.mismatches} matches {totals.matches}")
    print(f"wrote {len(rows)} marks to {arguments.out}")


if __name__ == "__main__":
    main()
