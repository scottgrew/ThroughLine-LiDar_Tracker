"""WOMD scenarios whose non-ego history is replaced by tracker output."""

import argparse
import csv
import math
import os
import struct
import subprocess
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from stage_womd import (
    WOMD_TO_TRACKER_CLASS,
    read_scenarios,
    stage_scenario,
    timeline,
)
from tfrecord.writer import TFRecordWriter
from womd_protos import scenario_pb2

NOW_STEP = 10
GATE = 2.0
CLASS_MISMATCH_COST = 1e9


class TrackRow(NamedTuple):
    step: int
    track_id: int
    tracker_class: int
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    speed: float


Rows = list[TrackRow]
Matches = dict[int, int]
Poses = dict[int, np.ndarray]
Steps = dict[int, int]


def wrap_angle(angle):
    return math.remainder(angle, 2.0 * math.pi)


def rows_from_csv(path, poses: Poses, steps: Steps) -> Rows:
    """Tracker CSV rows, boxes moved from the vehicle frame to the world."""
    rows = []
    with open(path, newline="") as file:
        for record in csv.DictReader(file):
            capture_time = int(record["frame_timestamp_micros"])
            if capture_time not in poses:
                raise KeyError(f"no pose for capture time {capture_time}")
            pose = poses[capture_time]
            local = np.array([
                float(record["center_x"]),
                float(record["center_y"]),
                float(record["center_z"]),
                1.0,
            ])
            x, y, z, _ = pose @ local
            vehicle_yaw = math.atan2(pose[1, 0], pose[0, 0])
            rows.append(TrackRow(
                step=steps[capture_time],
                track_id=int(record["track_id"]),
                tracker_class=int(record["object_class"]),
                x=float(x),
                y=float(y),
                z=float(z),
                length=float(record["length"]),
                width=float(record["width"]),
                height=float(record["height"]),
                yaw=wrap_angle(float(record["yaw"]) + vehicle_yaw),
                speed=float(record["speed"]),
            ))
    return rows


def eligible_agents(scenario, step: int) -> dict[int, int]:
    """WOMD track index -> tracker class, for non-SDC agents valid at step."""
    return {
        index: WOMD_TO_TRACKER_CLASS[track.object_type]
        for index, track in enumerate(scenario.tracks)
        if index != scenario.sdc_track_index
        and track.states[step].valid
        and track.object_type in WOMD_TO_TRACKER_CLASS
    }


def match_at_step(rows: Rows, scenario, step: int, gate_m: float) -> Matches:
    """Tracker track id -> WOMD track index, by centre distance at step."""
    step_rows = [row for row in rows if row.step == step]
    agents = eligible_agents(scenario, step)
    if not step_rows or not agents:
        return {}
    agent_indices = list(agents)
    agent_xy = np.array([
        [
            scenario.tracks[index].states[step].center_x,
            scenario.tracks[index].states[step].center_y,
        ]
        for index in agent_indices
    ])
    row_xy = np.array([[row.x, row.y] for row in step_rows])
    distances = np.linalg.norm(
        row_xy[:, None, :] - agent_xy[None, :, :], axis=2)
    row_classes = np.array([row.tracker_class for row in step_rows])
    agent_classes = np.array([agents[index] for index in agent_indices])
    mismatched = row_classes[:, None] != agent_classes[None, :]
    costs = np.where(mismatched, CLASS_MISMATCH_COST, distances)
    row_ids, agent_ids = linear_sum_assignment(costs)
    return {
        step_rows[row_id].track_id: agent_indices[agent_id]
        for row_id, agent_id in zip(row_ids, agent_ids, strict=True)
        if costs[row_id, agent_id] <= gate_m
    }


def set_state_from_row(state, row: TrackRow) -> None:
    state.valid = True
    state.center_x = row.x
    state.center_y = row.y
    state.center_z = row.z
    state.length = row.length
    state.width = row.width
    state.height = row.height
    state.heading = row.yaw
    state.velocity_x = row.speed * math.cos(row.yaw)
    state.velocity_y = row.speed * math.sin(row.yaw)


def tracked_scenario(scenario, rows: Rows, current_step=NOW_STEP, gate_m=GATE):
    """Copy of scenario; non-SDC steps 0..current_step come from rows."""
    matches = match_at_step(rows, scenario, current_step, gate_m)
    tracked = scenario_pb2.Scenario()
    tracked.CopyFrom(scenario)
    for index, track in enumerate(tracked.tracks):
        if index == scenario.sdc_track_index:
            continue
        for step in range(current_step + 1):
            track.states[step].valid = False
    for row in rows:
        index = matches.get(row.track_id)
        if index is not None and row.step <= current_step:
            set_state_from_row(tracked.tracks[index].states[row.step], row)
    return tracked


def write_records(path, payloads: Iterable[bytes]) -> None:
    """TFRecord framing: length, masked CRC of length, payload, masked CRC."""
    with open(path, "wb") as file:
        for payload in payloads:
            length = struct.pack("<Q", len(payload))
            file.write(length)
            file.write(TFRecordWriter.masked_crc(length))
            file.write(payload)
            file.write(TFRecordWriter.masked_crc(payload))


def main() -> None:
    from donor_emulator import build_index
    from donor_library import default_library

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ids", default=None)
    arguments = parser.parse_args()
    wanted = None
    if arguments.ids is not None:
        with open(arguments.ids) as file:
            wanted = {line.strip() for line in file if line.strip()}
    index = build_index(default_library())
    payloads = []
    matched_total = 0
    unmatched_total = 0
    for scenario in read_scenarios(arguments.shard):
        if wanted is not None and scenario.scenario_id not in wanted:
            continue
        log_path = stage_scenario(
            scenario, arguments.work, arguments.seed, index)
        csv_path = os.path.join(
            arguments.work, f"{scenario.scenario_id}.csv")
        subprocess.run(
            [arguments.tracker, "--segment", log_path, "--export", csv_path],
            check=True, stdout=subprocess.DEVNULL)
        times, poses = timeline(scenario)
        steps = {capture_time: step for step, capture_time in enumerate(times)}
        rows = rows_from_csv(csv_path, poses, steps)
        payloads.append(tracked_scenario(scenario, rows).SerializeToString())
        matched = match_at_step(rows, scenario, NOW_STEP, GATE)
        matched_total += len(matched)
        unmatched_total += (
            len(eligible_agents(scenario, NOW_STEP)) - len(matched))
    write_records(arguments.out, payloads)
    print(f"wrote {len(payloads)} scenarios; step {NOW_STEP} agents "
          f"matched {matched_total}, unmatched {unmatched_total}")


if __name__ == "__main__":
    main()
