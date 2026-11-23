"""Oracle tests for WOMD staging; conventions stated here, not read from code."""

import math

import numpy as np
from stage_womd import WOMD_TO_TRACKER_CLASS, objects_at_step, world_from_vehicle
from womd_protos import scenario_pb2

STEP = 3


def _state(x, y, z, heading, height=1.6, length=4.5, width=2.0, valid=True):
    return scenario_pb2.ObjectState(
        center_x=x, center_y=y, center_z=z,
        length=length, width=width, height=height, heading=heading,
        valid=valid,
    )


def _track(track_id, object_type, states):
    return scenario_pb2.Track(id=track_id, object_type=object_type, states=states)


def _scenario(sdc_valid=True):
    """At step STEP: the SDC at (10, 5, 1.0), height 1.6, heading pi/2; a
    vehicle at (10, 15, 1.0) heading pi/2; a cyclist; an "other" object; a
    pedestrian invalid at the step. Every track is invalid before STEP."""
    pad = [_state(0.0, 0.0, 0.0, 0.0, valid=False) for _ in range(STEP)]
    at_step = {
        0: _state(10.0, 5.0, 1.0, math.pi / 2, height=1.6, valid=sdc_valid),
        1: _state(10.0, 15.0, 1.0, math.pi / 2),
        2: _state(20.0, 5.0, 1.0, 0.0, length=1.8, width=0.6),
        3: _state(30.0, 5.0, 1.0, 0.0),
        4: _state(40.0, 5.0, 1.0, 0.0, valid=False),
    }
    types = {
        0: scenario_pb2.Track.TYPE_VEHICLE,
        1: scenario_pb2.Track.TYPE_VEHICLE,
        2: scenario_pb2.Track.TYPE_CYCLIST,
        3: scenario_pb2.Track.TYPE_OTHER,
        4: scenario_pb2.Track.TYPE_PEDESTRIAN,
    }
    scenario = scenario_pb2.Scenario(
        scenario_id="oracle-scenario",
        sdc_track_index=0,
        timestamps_seconds=[0.1 * index for index in range(STEP + 1)],
    )
    scenario.tracks.extend(
        _track(track_id, types[track_id], pad + [at_step[track_id]])
        for track_id in at_step
    )
    return scenario


def test_objects_at_step_oracles():
    """The convention, written out: the vehicle frame's origin is the SDC
    box's centre lowered by half its height, x along its heading."""
    objects, _pose = objects_at_step(_scenario(), STEP)
    assert sorted(objects.ids.tolist()) == [1, 2]  # SDC, other, invalid out
    assert objects.ids.dtype == np.uint64
    # WOMD stores heading as float32 (pi/2 is off by 4e-8), so positions and
    # angles agree to about 1e-6, not to float64 rounding.
    tol = 1e-6
    vehicle = int(np.where(objects.ids == 1)[0][0])
    cyclist = int(np.where(objects.ids == 2)[0][0])
    # Origin (10, 5, 1.0 - 0.8 = 0.2). The vehicle is 10 m ahead along the
    # SDC's heading (north), at the same height above the ground: z 0.8.
    assert np.allclose(objects.boxes[vehicle, :3], [10.0, 0.0, 0.8], atol=tol)
    assert math.isclose(objects.boxes[vehicle, 6], 0.0, abs_tol=tol)
    # The cyclist is 10 m east of the SDC: 10 m to its right, heading
    # east = -pi/2 relative to a north-facing vehicle.
    assert np.allclose(objects.boxes[cyclist, :3], [0.0, -10.0, 0.8], atol=tol)
    assert math.isclose(objects.boxes[cyclist, 6], -math.pi / 2, abs_tol=tol)
    assert int(objects.classes[cyclist]) == 4


def test_world_from_vehicle_maps_origin_and_x_axis():
    state = _state(7.0, -2.0, 3.0, math.pi / 4, height=2.0)
    pose = world_from_vehicle(state)
    origin = pose[:3, 3]
    expected = np.array([7.0, -2.0, 3.0 - 1.0])
    assert np.allclose(origin, expected)
    x_axis = pose[:3, 0]
    expected_axis = np.array([
        math.cos(state.heading),
        math.sin(state.heading),
        0.0,
    ])
    assert np.allclose(x_axis, expected_axis)


def test_invalid_sdc_raises():
    scenario = _scenario(sdc_valid=False)
    try:
        objects_at_step(scenario, STEP)
    except ValueError as error:
        assert "oracle-scenario" in str(error) and f"step {STEP}" in str(error)
    else:
        raise AssertionError("an invalid SDC state must raise ValueError")
