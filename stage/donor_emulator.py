"""Donor-replay detector emulation with analog hops between library rows."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from donor_library import CLASS_SLOTS, DonorLibrary, validate_library
from emulate_detector import Objects, apply_errors, box_band, emulate, make_rng
from scipy.spatial import cKDTree

HOP_NEIGHBOURS = 8
# Chance a chain end jumps uniformly to another chain instead of the analog
# hop: a nearest-neighbour walk alone traps itself in short cycles.
HOP_EPS = 0.2
# Extra nearest rows queried so same-chain rows can be dropped.
_HOP_QUERY_SLACK = 32
FLIP_PENALTY = 8.0
_HIT, _MISSED, _ANY = 0, 1, 2


class Candidates(NamedTuple):
    rows: np.ndarray
    tree: cKDTree | None


@dataclass(frozen=True)
class DonorIndex:
    library: DonorLibrary
    features: np.ndarray
    neighbours: int
    starts: dict[tuple[int, int], np.ndarray]
    candidates: dict[tuple[int, int, int], Candidates]
    chain: np.ndarray
    eps: float


def _class_in_library(library: DonorLibrary, cls: int) -> bool:
    return cls in CLASS_SLOTS and bool(np.any(library.cls == cls))


def _slot_mask(library: DonorLibrary, slot: int, band: int) -> np.ndarray:
    codes = [code for code, owner in CLASS_SLOTS.items() if owner == slot]
    return np.isin(library.cls, codes) & (library.band == band)


def build_index(library: DonorLibrary, neighbours=HOP_NEIGHBOURS, eps=HOP_EPS):
    validate_library(library)
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"hop eps must be in [0, 1], got {eps}")
    codes, code_of_row = np.unique(library.cls, return_inverse=True)
    row_slot = np.array([CLASS_SLOTS[int(code)] for code in codes])[code_of_row]
    scale = library.scale[row_slot, library.band.astype(np.int64)]
    features = np.column_stack(
        [library.err / scale, FLIP_PENALTY * library.flip.astype(np.float64)])
    # Row j continues in its band when row j + 1 is the same object's next
    # frame in the same band; the last row has no successor.
    next_band = np.append(library.band[1:], -1)
    starts: dict[tuple[int, int], np.ndarray] = {}
    candidates: dict[tuple[int, int, int], Candidates] = {}
    for slot in range(3):
        for band in range(3):
            mask = _slot_mask(library, slot, band)
            starts[(slot, band)] = np.where(mask)[0].astype(np.int64)
            continuable = mask & library.has_next & (next_band == band)
            for kind, missed_only in (
                    (_HIT, False), (_MISSED, True), (_ANY, None)):
                if kind == _ANY:
                    rows = np.where(continuable)[0].astype(np.int64)
                elif missed_only:
                    sel = continuable & library.missed
                    rows = np.where(sel)[0].astype(np.int64)
                else:
                    sel = continuable & ~library.missed
                    rows = np.where(sel)[0].astype(np.int64)
                tree = None
                if kind == _HIT and len(rows):
                    tree = cKDTree(features[rows])
                candidates[(slot, band, kind)] = Candidates(rows, tree)
    # Row j starts a new chain when row j - 1 has no successor.
    chain = np.cumsum(np.r_[True, ~library.has_next[:-1]]).astype(np.int64)
    return DonorIndex(
        library, features, neighbours, starts, candidates, chain, eps)


def _pick_uniform(rows: np.ndarray, rng: np.random.Generator) -> int:
    if len(rows) == 0:
        raise ValueError("donor candidate set is empty")
    return int(rows[rng.integers(len(rows))])


def _other_chain(index: DonorIndex, rows: np.ndarray, row: int) -> np.ndarray:
    return rows[index.chain[rows] != index.chain[row]]


def _analog_hop(index, slot, band, row, rng):
    """Successor of a uniform pick among the nearest other-chain hit rows."""
    hit = index.candidates[(slot, band, _HIT)]
    if len(hit.rows) == 0:
        return None
    query_k = min(index.neighbours + _HOP_QUERY_SLACK, len(hit.rows))
    _, positions = hit.tree.query(index.features[row], k=query_k)
    near = _other_chain(index, hit.rows[np.atleast_1d(positions)], row)
    near = near[:index.neighbours]
    if len(near) == 0:
        return None
    return int(near[rng.integers(len(near))]) + 1


def _fallback_hop(index, slot, band, row, rng):
    """Successor of a uniform ANY row of another chain, else of any ANY row."""
    rows = index.candidates[(slot, band, _ANY)].rows
    pool = _other_chain(index, rows, row)
    if len(pool) == 0:
        pool = rows
    return _pick_uniform(pool, rng) + 1


def next_row(index, slot, band, row, rng):
    library = index.library
    if row is None:
        return _pick_uniform(index.starts[(slot, band)], rng)
    if library.has_next[row] and int(library.band[row + 1]) == band:
        return row + 1
    # A chain end hops to another chain: a hop back into the chain being left
    # is what closes the cycles. A share eps of hops is a uniform jump
    # instead, because the analog hop alone still traps the walk in cycles.
    jump = rng.random() < index.eps
    if jump:
        any_rows = index.candidates[(slot, band, _ANY)].rows
        pool = _other_chain(index, any_rows, row)
        if len(pool):
            return _pick_uniform(pool, rng) + 1
    if library.missed[row]:
        missed = index.candidates[(slot, band, _MISSED)].rows
        others = _other_chain(index, missed, row)
        if len(others):
            return _pick_uniform(others, rng) + 1
    else:
        hop = _analog_hop(index, slot, band, row, rng)
        if hop is not None:
            return hop
    return _fallback_hop(index, slot, band, row, rng)


def _donor_box(box: np.ndarray, library: DonorLibrary, row: int) -> np.ndarray:
    return apply_errors(box, library.err[row], bool(library.flip[row]))


def emulate_donor_frame(objects, state, rng, index):
    library = index.library
    present = {int(value) for value in objects.ids}
    for object_id in list(state):
        if object_id not in present:
            del state[object_id]
    classes_out: list[int] = []
    boxes_out: list[np.ndarray] = []
    for idx, object_id in enumerate(objects.ids):
        cls = int(objects.classes[idx])
        if not _class_in_library(library, cls):
            continue
        slot = CLASS_SLOTS[cls]
        box = objects.boxes[idx]
        band = box_band(box)
        oid = int(object_id)
        previous = state.get(oid)
        row = next_row(index, slot, band, previous, rng)
        state[oid] = row
        if library.missed[row]:
            continue
        classes_out.append(cls)
        boxes_out.append(_donor_box(box, library, row))
    if not boxes_out:
        return np.zeros(0, np.int64), np.zeros((0, 7))
    return np.array(classes_out, np.int64), np.stack(boxes_out)


def emulate_donor(frames, name, seed, index, model=None):
    empty = [
        (Objects(np.zeros(0, np.uint64), np.zeros(0, np.int64),
                 np.zeros((0, 7))), pose)
        for _, pose in frames]
    ghosts = emulate(empty, f"{name}:ghosts", seed, model)
    rng = make_rng(name, seed)
    state: dict[int, int] = {}
    output: list[tuple[np.ndarray, np.ndarray]] = []
    pairs = zip(frames, ghosts, strict=True)
    for (objects, _), (ghost_cls, ghost_boxes) in pairs:
        det_cls, det_boxes = emulate_donor_frame(objects, state, rng, index)
        if len(det_cls) == 0 and len(ghost_cls) == 0:
            output.append((np.zeros(0, np.int64), np.zeros((0, 7))))
            continue
        if len(ghost_cls):
            all_cls = np.concatenate([det_cls, ghost_cls])
            all_boxes = np.vstack([det_boxes, ghost_boxes])
        else:
            all_cls, all_boxes = det_cls, det_boxes
        output.append((all_cls, all_boxes))
    return output
