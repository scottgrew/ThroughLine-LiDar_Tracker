"""Stage WOMD scenarios into tracker MCAPs with emulated CasA detections."""

import argparse
import math
import os

import numpy as np
import tfrecord.reader as tfrecord_reader
from donor_emulator import emulate_donor
from emulate_detector import Objects
from stage_segment import write_log
from womd_protos import scenario_pb2

WOMD_TO_TRACKER_CLASS = {1: 1, 2: 2, 3: 4}

STEPS = 91


def read_scenarios(path):
    """Yields Scenario messages from a WOMD TFRecord shard."""
    for record in tfrecord_reader.tfrecord_iterator(path):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(record.tobytes())
        yield scenario


def world_from_vehicle(sdc_state):
    """(4, 4) pose of the SDC box: z-rotation by heading, translation
    (center_x, center_y, center_z - height / 2)."""
    heading = sdc_state.heading
    cosine = math.cos(heading)
    sine = math.sin(heading)
    center_z = sdc_state.center_z - sdc_state.height / 2.0
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = cosine
    pose[0, 1] = -sine
    pose[1, 0] = sine
    pose[1, 1] = cosine
    pose[0, 3] = sdc_state.center_x
    pose[1, 3] = sdc_state.center_y
    pose[2, 3] = center_z
    return pose


def sdc_pose(scenario, step):
    """(4, 4) world_from_vehicle of the SDC at one step."""
    sdc_state = scenario.tracks[scenario.sdc_track_index].states[step]
    if not sdc_state.valid:
        raise ValueError(
            f"{scenario.scenario_id}: SDC invalid at step {step}")
    return world_from_vehicle(sdc_state)


def timeline(scenario):
    """Returns (capture time per step, {capture time: world_from_vehicle})."""
    times = [
        round(scenario.timestamps_seconds[step] * 1_000_000)
        for step in range(STEPS)
    ]
    poses = {
        capture_time: sdc_pose(scenario, step)
        for step, capture_time in enumerate(times)
    }
    return times, poses


def objects_at_step(scenario, step):
    """Returns labelled Objects and world_from_vehicle at one step."""
    sdc_index = scenario.sdc_track_index
    sdc_heading = scenario.tracks[sdc_index].states[step].heading
    pose = sdc_pose(scenario, step)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    ids: list = []
    classes: list = []
    boxes: list = []
    for index, track in enumerate(scenario.tracks):
        if index == sdc_index:
            continue
        state = track.states[step]
        if not state.valid:
            continue
        object_type = track.object_type
        if object_type not in WOMD_TO_TRACKER_CLASS:
            continue
        tracker_class = WOMD_TO_TRACKER_CLASS[object_type]
        world = np.array(
            [state.center_x, state.center_y, state.center_z],
            dtype=np.float64,
        )
        local = rotation.T @ (world - translation)
        yaw = math.remainder(state.heading - sdc_heading, 2.0 * math.pi)
        box = np.array(
            [
                local[0], local[1], local[2], state.length, state.width,
                state.height, yaw,
            ],
            dtype=np.float64,
        )
        ids.append(np.uint64(track.id))
        classes.append(tracker_class)
        boxes.append(box)
    if not ids:
        objects = Objects(
            ids=np.zeros(0, np.uint64),
            classes=np.zeros(0, np.int64),
            boxes=np.zeros((0, 7)),
        )
    else:
        objects = Objects(
            ids=np.array(ids, dtype=np.uint64),
            classes=np.array(classes, dtype=np.int64),
            boxes=np.stack(boxes),
        )
    return objects, pose


def labels_by_time(times, frames):
    """Returns {capture_time_micros: (classes, boxes)} from frame Objects."""
    labels = {}
    for capture_time, (objects, _) in zip(times, frames, strict=True):
        if len(objects.ids) == 0:
            continue
        labels[capture_time] = (objects.classes, objects.boxes)
    return labels


def stage_scenario(scenario, out_dir, seed, index):
    """Writes out_dir/<scenario_id>.mcap and returns its path."""
    scenario_id = scenario.scenario_id
    frames = [objects_at_step(scenario, step) for step in range(STEPS)]
    times, poses = timeline(scenario)
    poses_by_time = {
        capture_time: pose.reshape(-1).tolist()
        for capture_time, pose in poses.items()
    }
    detections = emulate_donor(frames, scenario_id, seed, index)
    labels = labels_by_time(times, frames)
    out_path = os.path.join(out_dir, f"{scenario_id}.mcap")
    os.makedirs(out_dir, exist_ok=True)
    write_log(
        out_path, scenario_id, times, poses_by_time, None,
        detections, labels)
    return out_path


def main():
    from donor_emulator import build_index
    from donor_library import default_library

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()
    index = build_index(default_library())
    for count, scenario in enumerate(read_scenarios(arguments.shard), 1):
        stage_scenario(scenario, arguments.out, arguments.seed, index)
        if arguments.limit is not None and count >= arguments.limit:
            break


if __name__ == "__main__":
    main()
