"""Stateful CasA-style detector error applied to labelled WOMD boxes."""

import functools
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.special import ndtr

COMPONENTS = ("along", "across", "z", "yaw", "length", "width", "height")
RANGE_EDGES_M = (30.0, 50.0)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "casa_error_model.json"
P_LEVELS = np.linspace(0.0, 1.0, 101)
_MIN_EIGEN = -1e-9
# Two layouts that must not be mixed: the error vector follows COMPONENTS,
# the box is x, y, z, length, width, height, yaw (waymo_boxes.Box).
_ALONG, _ACROSS, _Z, _YAW, _LENGTH, _WIDTH, _HEIGHT = range(7)
_BOX_X, _BOX_Y, _BOX_Z, _BOX_LENGTH, _BOX_WIDTH, _BOX_HEIGHT, _BOX_YAW = (
    range(7))


@dataclass(frozen=True)
class BandModel:
    """Per-range-band miss rates and empirical error percentiles."""

    p_miss_after_hit: float
    p_miss_after_miss: float
    percentiles: np.ndarray


@dataclass(frozen=True)
class ClassModel:
    """Measured error model for one tracked object class."""

    bands: tuple[BandModel, BandModel, BandModel]
    rank_lag1: np.ndarray
    rank_corr: np.ndarray
    p_flip_after_no_flip: float
    p_flip_after_flip: float
    fp_per_frame: float
    fp_survival: float
    fp_range_deciles_m: tuple[float, ...]
    fp_size_m: tuple[float, float, float]


def linear_lag1(rank_lag1):
    """Pearson lag-1 / correlation from Spearman rank; scalar or array."""
    return 2.0 * np.sin(np.asarray(rank_lag1, dtype=float) * math.pi / 6.0)


def range_band(ranges: np.ndarray) -> np.ndarray:
    """Left-closed range band index: [0,30), [30,50), [50, inf) metres."""
    return np.digitize(ranges, RANGE_EDGES_M, right=False)


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Return A with A @ A.T = matrix for symmetric PSD matrix."""
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) < _MIN_EIGEN:
        raise ValueError("matrix has negative eigenvalue")
    return vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))


def _copula_mats(class_model: ClassModel):
    """Return (A0, A1, D) for the class Gaussian copula."""
    corr = np.asarray(class_model.rank_corr, dtype=float)
    lag = np.asarray(class_model.rank_lag1, dtype=float)
    c_matrix = linear_lag1(corr)
    d_diag = linear_lag1(lag)
    d_matrix = np.diag(d_diag)
    s_matrix = c_matrix - d_matrix @ c_matrix @ d_matrix
    a0 = _sqrt_psd(c_matrix)
    a1 = _sqrt_psd(s_matrix)
    return a0, a1, d_matrix


def _in_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_model(model: dict[int, ClassModel]) -> None:
    """Raise ValueError naming class code on any model violation."""
    for code, class_model in model.items():
        tag = f"class {code}"
        lag = np.asarray(class_model.rank_lag1, dtype=float)
        corr = np.asarray(class_model.rank_corr, dtype=float)
        if lag.shape != (7,):
            raise ValueError(f"{tag}: rank_lag1 must have shape (7,)")
        if corr.shape != (7, 7):
            raise ValueError(f"{tag}: rank_corr must have shape (7, 7)")
        if not np.allclose(corr, corr.T, atol=1e-12):
            raise ValueError(f"{tag}: rank_corr must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-12):
            raise ValueError(f"{tag}: rank_corr must have unit diagonal")
        for band in class_model.bands:
            _in_unit_interval(f"{tag} p_miss_after_hit", band.p_miss_after_hit)
            _in_unit_interval(
                f"{tag} p_miss_after_miss", band.p_miss_after_miss)
            pct = np.asarray(band.percentiles, dtype=float)
            if pct.shape != (7, 101):
                raise ValueError(f"{tag}: percentiles must have shape (7, 101)")
            if not np.all(np.diff(pct, axis=1) >= -1e-12):
                raise ValueError(f"{tag}: percentiles must be non-decreasing")
        _in_unit_interval(f"{tag} p_flip_after_no_flip",
                          class_model.p_flip_after_no_flip)
        _in_unit_interval(f"{tag} p_flip_after_flip",
                          class_model.p_flip_after_flip)
        _in_unit_interval(f"{tag} fp_survival", class_model.fp_survival)
        try:
            _copula_mats(class_model)
        except ValueError as error:
            raise ValueError(f"{tag}: {error}") from error


def load_model(path: Path) -> dict[int, ClassModel]:
    """Load the JSON error model and validate it."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    model: dict[int, ClassModel] = {}
    for key, raw in payload["classes"].items():
        bands = tuple(
            BandModel(
                p_miss_after_hit=float(b["p_miss_after_hit"]),
                p_miss_after_miss=float(b["p_miss_after_miss"]),
                percentiles=np.asarray(b["percentiles"], dtype=float),
            )
            for b in raw["bands"]
        )
        model[int(key)] = ClassModel(
            bands=bands,
            rank_lag1=np.asarray(raw["rank_lag1"], dtype=float),
            rank_corr=np.asarray(raw["rank_corr"], dtype=float),
            p_flip_after_no_flip=float(raw["p_flip_after_no_flip"]),
            p_flip_after_flip=float(raw["p_flip_after_flip"]),
            fp_per_frame=float(raw["fp_per_frame"]),
            fp_survival=float(raw["fp_survival"]),
            fp_range_deciles_m=tuple(
                float(v) for v in raw["fp_range_deciles_m"]),
            fp_size_m=tuple(float(v) for v in raw["fp_size_m"]),
        )
    validate_model(model)
    return model


@functools.cache
def default_model() -> dict[int, ClassModel]:
    return load_model(DEFAULT_MODEL_PATH)


class Objects(NamedTuple):
    """One frame of labelled boxes to emulate detections from."""

    ids: np.ndarray
    classes: np.ndarray
    boxes: np.ndarray


class _ObjectState:
    __slots__ = ("z", "missed", "flipped")

    def __init__(self):
        self.z = np.zeros(7)
        self.missed = False
        self.flipped = False


class _GhostState:
    __slots__ = (
        "object_class", "world_x", "world_y", "world_yaw", "length", "width",
        "height", "z",
    )

    def __init__(self, cls, wx, wy, wyaw, length, width, height, z):
        self.object_class = cls
        self.world_x = wx
        self.world_y = wy
        self.world_yaw = wyaw
        self.length = length
        self.width = width
        self.height = height
        self.z = z


class EmulatorState:
    """Per-object copula state, miss and flip chains, and persistent ghosts."""

    def __init__(self):
        self.objects: dict[int, _ObjectState] = {}
        self.ghosts: list[_GhostState] = []
        self.first_frame = True
        self._mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _class_mats(state: EmulatorState, code: int, spec: ClassModel):
    cached = state._mats.get(code)
    if cached is None:
        cached = _copula_mats(spec)
        state._mats[code] = cached
    return cached


def _ego_yaw(world_from_vehicle: np.ndarray) -> float:
    return math.atan2(world_from_vehicle[1, 0], world_from_vehicle[0, 0])


def _stationary_on_prob(on_after_off: float, on_after_on: float) -> float:
    return on_after_off / (on_after_off + 1.0 - on_after_on)


def _memoryless_band(band: BandModel) -> BandModel:
    rate = _stationary_on_prob(band.p_miss_after_hit, band.p_miss_after_miss)
    return replace(band, p_miss_after_hit=rate, p_miss_after_miss=rate)


def _without_memory(class_model: ClassModel) -> ClassModel:
    flip_rate = _stationary_on_prob(
        class_model.p_flip_after_no_flip, class_model.p_flip_after_flip)
    return replace(
        class_model,
        bands=tuple(_memoryless_band(band) for band in class_model.bands),
        rank_lag1=np.zeros_like(class_model.rank_lag1),
        p_flip_after_no_flip=flip_rate,
        p_flip_after_flip=flip_rate,
        fp_survival=0.0,
    )


def independent_model(model: dict[int, ClassModel]) -> dict[int, ClassModel]:
    """Return a copy of model with the same marginal error rates and no
    frame-to-frame memory: rank_lag1 zero, miss and flip chains at their
    stationary rates, and no ghost survival. This is the independent-noise
    arm of the tracking view; the input model is not changed.
    """
    return {
        code: _without_memory(class_model)
        for code, class_model in model.items()
    }


def _copula_errors(z: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    uniforms = ndtr(z)
    return np.array([
        np.interp(uniforms[k], P_LEVELS, percentiles[k]) for k in range(7)
    ])


def box_band(box: np.ndarray) -> int:
    return int(range_band(np.array([math.hypot(box[0], box[1])]))[0])


def apply_errors(box: np.ndarray, errors: np.ndarray, flipped: bool):
    """Apply a 7-component error in vehicle-frame box coordinates."""
    yaw = box[_BOX_YAW]
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    out = box.copy()
    along, across = errors[_ALONG], errors[_ACROSS]
    out[_BOX_X] += cosine * along - sine * across
    out[_BOX_Y] += sine * along + cosine * across
    out[_BOX_Z] += errors[_Z]
    yaw_delta = errors[_YAW]
    if flipped:
        yaw_delta += math.pi
    out[_BOX_YAW] = math.remainder(yaw + yaw_delta, 2.0 * math.pi)
    out[_BOX_LENGTH] = max(0.1, box[_BOX_LENGTH] + errors[_LENGTH])
    out[_BOX_WIDTH] = max(0.1, box[_BOX_WIDTH] + errors[_WIDTH])
    out[_BOX_HEIGHT] = max(0.1, box[_BOX_HEIGHT] + errors[_HEIGHT])
    return out


def make_rng(name: str, seed: int) -> np.random.Generator:
    """Deterministic PCG64 stream from sha256 of name:seed."""
    digest = hashlib.sha256(f"{name}:{seed}".encode()).digest()
    seed_int = int.from_bytes(digest[:8], "little")
    return np.random.Generator(np.random.PCG64(seed_int))


def _noisy_box(box: np.ndarray, band: BandModel, z: np.ndarray, flipped: bool):
    errors = _copula_errors(z, band.percentiles)
    return apply_errors(box, errors, flipped)


def _vbox_from_world(wx, wy, wyaw, length, width, height, rot, trans, ego_yaw):
    world = np.array([wx, wy, 0.0])
    local = rot.T @ (world - trans)
    return np.array(
        [
            local[0], local[1], height / 2.0,
            length, width, height,
            math.remainder(wyaw - ego_yaw, 2.0 * math.pi),
        ],
        dtype=np.float64,
    )


def _birth_ghost(object_class, spec, state, vehicle_xy, rng):
    deciles = spec.fp_range_deciles_m
    sample = rng.uniform(0.0, 1.0)
    radius = float(np.interp(sample * 10.0, np.arange(11), deciles))
    bearing = rng.uniform(-math.pi, math.pi)
    world_x = vehicle_xy[0] + radius * math.cos(bearing)
    world_y = vehicle_xy[1] + radius * math.sin(bearing)
    world_yaw = rng.uniform(-math.pi, math.pi)
    length, width, height = spec.fp_size_m
    a0, _, _ = _class_mats(state, object_class, spec)
    z = a0 @ rng.standard_normal(7)
    return _GhostState(
        object_class, world_x, world_y, world_yaw,
        length, width, height, z,
    )


def emulate_frame(objects, world_from_vehicle, state, rng, model):
    """Noisy detections for one labelled frame; advance ghost state.

    objects: Objects; world_from_vehicle: (4, 4); state: EmulatorState;
    rng: np.random.Generator; model: Mapping[int, ClassModel]. Returns
    classes (m,) int64 and boxes (m, 7) in the vehicle frame.
    """
    current_ids = {int(value) for value in objects.ids}
    for object_id in list(state.objects):
        if object_id not in current_ids:
            del state.objects[object_id]

    classes_out: list[int] = []
    boxes_out: list[np.ndarray] = []
    pose = np.asarray(world_from_vehicle, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    vehicle_world = (float(translation[0]), float(translation[1]))
    ego_yaw = _ego_yaw(pose)

    for index, object_id in enumerate(objects.ids):
        object_class = int(objects.classes[index])
        if object_class not in model:
            continue
        spec = model[object_class]
        box = objects.boxes[index]
        band_index = box_band(box)
        band = spec.bands[band_index]
        a0, a1, d_mat = _class_mats(state, object_class, spec)
        previous = state.objects.get(int(object_id))
        if previous is None:
            state_obj = _ObjectState()
            state_obj.z = a0 @ rng.standard_normal(7)
            miss_rate = _stationary_on_prob(
                band.p_miss_after_hit, band.p_miss_after_miss)
            state_obj.missed = rng.random() < miss_rate
            flip_rate = _stationary_on_prob(
                spec.p_flip_after_no_flip, spec.p_flip_after_flip)
            state_obj.flipped = rng.random() < flip_rate
            state.objects[int(object_id)] = state_obj
        else:
            previous.z = d_mat @ previous.z + a1 @ rng.standard_normal(7)
            probability = (
                band.p_miss_after_miss if previous.missed
                else band.p_miss_after_hit
            )
            previous.missed = rng.random() < probability
            flip_probability = (
                spec.p_flip_after_flip if previous.flipped
                else spec.p_flip_after_no_flip
            )
            previous.flipped = rng.random() < flip_probability
            state_obj = previous

        if state_obj.missed:
            continue
        classes_out.append(object_class)
        boxes_out.append(_noisy_box(box, band, state_obj.z, state_obj.flipped))

    sorted_classes = sorted(model)
    first = state.first_frame
    if first:
        state.first_frame = False
    else:
        surviving: list[_GhostState] = []
        for ghost in state.ghosts:
            if rng.random() < model[ghost.object_class].fp_survival:
                surviving.append(ghost)
        for ghost in surviving:
            spec = model[ghost.object_class]
            _, a1, d_mat = _class_mats(state, ghost.object_class, spec)
            ghost.z = d_mat @ ghost.z + a1 @ rng.standard_normal(7)
        state.ghosts = surviving

    for object_class in sorted_classes:
        spec = model[object_class]
        rate = spec.fp_per_frame if first else (
            spec.fp_per_frame * (1.0 - spec.fp_survival)
        )
        count = rng.poisson(rate)
        for _ in range(count):
            ghost = _birth_ghost(
                object_class, spec, state, vehicle_world, rng)
            state.ghosts.append(ghost)

    for ghost in state.ghosts:
        spec = model[ghost.object_class]
        box = _vbox_from_world(
            ghost.world_x, ghost.world_y, ghost.world_yaw,
            ghost.length, ghost.width, ghost.height,
            rotation, translation, ego_yaw,
        )
        band = spec.bands[box_band(box)]
        classes_out.append(ghost.object_class)
        boxes_out.append(_noisy_box(box, band, ghost.z, False))

    if not boxes_out:
        return np.zeros(0, np.int64), np.zeros((0, 7))
    return np.array(classes_out, np.int64), np.stack(boxes_out)


def emulate(frames, name, seed, model=None):
    """Emulate detections over ordered frames with a deterministic RNG.

    frames: iterable of (Objects, world_from_vehicle (4, 4)); name and seed
    pick the stream (sha256 of "name:seed"). model: optional error model dict.
    """
    if model is None:
        model = default_model()
    rng = make_rng(name, seed)
    state = EmulatorState()
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for objects, world_from_vehicle in frames:
        output.append(
            emulate_frame(objects, world_from_vehicle, state, rng, model))
    return output
