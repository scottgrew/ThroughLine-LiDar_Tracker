"""Oracle tests for tracked scenarios; expected values are hand-written."""

import math
import os
import tempfile

import numpy as np
from tfrecord.reader import tfrecord_iterator
from tracked_scenarios import (
    TrackRow,
    rows_from_csv,
    tracked_scenario,
    write_records,
)
from womd_protos import scenario_pb2

TOL = 1e-5
VEHICLE = scenario_pb2.Track.TYPE_VEHICLE
PEDESTRIAN = scenario_pb2.Track.TYPE_PEDESTRIAN
CSV_HEADER = (
    "frame_timestamp_micros,track_id,object_class,center_x,center_y,"
    "center_z,length,width,height,yaw,speed,yaw_rate\n"
)


def _state(x, y, valid=True):
    return scenario_pb2.ObjectState(
        center_x=x, center_y=y, center_z=0.5, length=4.5, width=2.0,
        height=1.6, heading=0.0, valid=valid,
    )


def _scenario():
    """Steps 0-2 history, step 3 future. SDC (id 0) valid at the origin. A
    vehicle (id 7) valid at x = 10 + step. A pedestrian (id 9) valid only at
    step 2, at (20, 5)."""
    invalid = _state(0.0, 0.0, valid=False)
    scenario = scenario_pb2.Scenario(
        scenario_id="tracked-oracle",
        sdc_track_index=0,
        timestamps_seconds=[0.0, 0.1, 0.2, 0.3],
    )
    scenario.tracks.extend([
        scenario_pb2.Track(
            id=0, object_type=VEHICLE,
            states=[_state(0.0, 0.0) for _ in range(4)]),
        scenario_pb2.Track(
            id=7, object_type=VEHICLE,
            states=[_state(10.0 + step, 0.0) for step in range(4)]),
        scenario_pb2.Track(
            id=9, object_type=PEDESTRIAN,
            states=[invalid, invalid, _state(20.0, 5.0), invalid]),
    ])
    return scenario


def _row(step, track_id, tracker_class, x, y):
    """Tracker row with yaw pi/3 and speed 4 m/s: velocity (2, 2 sqrt 3)."""
    return TrackRow(
        step=step, track_id=track_id, tracker_class=tracker_class, x=x, y=y,
        z=0.5, length=4.5, width=2.0, height=1.6, yaw=math.pi / 3, speed=4.0,
    )


def test_tracked_scenario_replaces_history_only():
    scenario = _scenario()
    original = scenario_pb2.Scenario()
    original.CopyFrom(scenario)
    rows = [
        _row(1, 7, 1, 11.3, 0.0),
        _row(2, 7, 1, 12.4, 0.0),
        _row(2, 9, 2, 50.0, 5.0),
    ]

    tracked = tracked_scenario(scenario, rows, current_step=2, gate_m=2.0)

    vehicle = tracked.tracks[1].states
    assert not vehicle[0].valid
    assert vehicle[1].valid and vehicle[2].valid
    assert math.isclose(vehicle[1].center_x, 11.3, abs_tol=TOL)
    assert math.isclose(vehicle[2].center_x, 12.4, abs_tol=TOL)
    assert math.isclose(vehicle[2].heading, math.pi / 3, abs_tol=TOL)
    assert math.isclose(vehicle[2].velocity_x, 2.0, abs_tol=TOL)
    assert math.isclose(vehicle[2].velocity_y, 2.0 * math.sqrt(3.0),
                        abs_tol=TOL)
    assert vehicle[3] == original.tracks[1].states[3]
    assert not any(state.valid for state in tracked.tracks[2].states[:3])
    assert tracked.tracks[2].states[3] == original.tracks[2].states[3]
    assert tracked.tracks[0] == original.tracks[0]
    assert scenario == original


def test_rows_from_csv_moves_boxes_to_world():
    csv_text = (
        CSV_HEADER
        + "1000000,7,1,2.0,1.0,0.4,4.5,2.0,1.6,0.25,3.0,0.0\n"
        + "1100000,9,2,3.0,-1.0,0.2,0.8,0.6,1.7,0.0,1.2,0.0\n"
    )
    rotated_quarter_turn = np.array([
        [0.0, -1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 6.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    poses = {1000000: rotated_quarter_turn, 1100000: np.eye(4)}
    steps = {1000000: 0, 1100000: 1}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tracks.csv")
        with open(path, "w") as file:
            file.write(csv_text)
        first, second = rows_from_csv(path, poses, steps)

    assert (first.step, first.track_id, first.tracker_class) == (0, 7, 1)
    assert math.isclose(first.x, 4.0, abs_tol=TOL)
    assert math.isclose(first.y, 8.0, abs_tol=TOL)
    assert math.isclose(first.z, 0.9, abs_tol=TOL)
    assert math.isclose(first.yaw, 1.8207963, abs_tol=TOL)
    assert math.isclose(first.speed, 3.0, abs_tol=TOL)
    assert (second.step, second.track_id, second.tracker_class) == (1, 9, 2)
    assert math.isclose(second.x, 3.0, abs_tol=TOL)
    assert math.isclose(second.y, -1.0, abs_tol=TOL)
    assert math.isclose(second.z, 0.2, abs_tol=TOL)
    assert math.isclose(second.yaw, 0.0, abs_tol=TOL)


def test_rows_from_csv_raises_on_time_without_pose():
    csv_text = CSV_HEADER + "1000000,7,1,2.0,1.0,0.4,4.5,2.0,1.6,0.25,3.0,0.0\n"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tracks.csv")
        with open(path, "w") as file:
            file.write(csv_text)
        try:
            rows_from_csv(path, {}, {1000000: 0})
        except KeyError as error:
            assert "1000000" in str(error)
        else:
            raise AssertionError("a row time without a pose must raise")


def test_write_records_round_trip():
    payloads = [b"first payload", b"second payload"]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "shard.tfrecord")
        write_records(path, payloads)
        read_back = [record.tobytes() for record in tfrecord_iterator(path)]
    assert read_back == payloads
