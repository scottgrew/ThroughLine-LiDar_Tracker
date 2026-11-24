"""CasA donor error library built from matched per-object frame histories."""

import argparse
import functools
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from detector_error import (
    COMPONENTS,
    FrameInput,
    band_of,
    match_frame,
    segment_frames,
)

_LIB_DIR = Path(__file__).resolve().parent
DEFAULT_LIBRARY_PATH = _LIB_DIR / "casa_donor_library.npz"
CLASS_SLOTS = {1: 0, 2: 1, 4: 2}
_N_ERR = len(COMPONENTS)


@dataclass(frozen=True)
class DonorLibrary:
    segments: tuple[str, ...]
    segment: np.ndarray
    cls: np.ndarray
    band: np.ndarray
    missed: np.ndarray
    flip: np.ndarray
    err: np.ndarray
    has_next: np.ndarray
    scale: np.ndarray


def _yaw_for_storage(yaw_err: float) -> tuple[bool, float]:
    flip = abs(yaw_err) > math.pi / 2
    if flip:
        return True, float(math.remainder(yaw_err - math.pi, 2.0 * math.pi))
    return False, float(yaw_err)


def build_library(frames: Sequence[FrameInput], gate_scale=1.0) -> DonorLibrary:
    """Build a donor library from labelled frames and CasA detections."""
    seg_names = tuple(sorted({frame.segment for frame in frames}))
    seg_index = {name: index for index, name in enumerate(seg_names)}
    rows: list[dict] = []
    for frame in frames:
        matched, missed, _ = match_frame(frame, gate_scale)
        for row in matched:
            cls = int(row[0])
            if cls not in CLASS_SLOTS:
                continue
            flip, yaw = _yaw_for_storage(float(row[8]))
            err = np.array(
                [row[5], row[6], row[7], yaw, row[9], row[10], row[11]],
                dtype=np.float32,
            )
            rows.append({
                "segment": seg_index[frame.segment],
                "cls": cls,
                "id": row[2],
                "frame": frame.frame,
                "band": int(band_of(np.array([row[1]]))[0]),
                "missed": False,
                "flip": flip,
                "err": err,
            })
        for row in missed:
            cls = int(row[0])
            if cls not in CLASS_SLOTS:
                continue
            rows.append({
                "segment": seg_index[frame.segment],
                "cls": cls,
                "id": row[2],
                "frame": frame.frame,
                "band": int(band_of(np.array([row[1]]))[0]),
                "missed": True,
                "flip": False,
                "err": np.zeros(_N_ERR, dtype=np.float32),
            })
    rows.sort(key=lambda item: (
        item["segment"], item["cls"], str(item["id"]), item["frame"]))
    n_rows = len(rows)
    segment = np.array([r["segment"] for r in rows], dtype=np.int16)
    cls = np.array([r["cls"] for r in rows], dtype=np.int8)
    band = np.array([r["band"] for r in rows], dtype=np.int8)
    missed = np.array([r["missed"] for r in rows], dtype=bool)
    flip = np.array([r["flip"] for r in rows], dtype=bool)
    err = np.stack([r["err"] for r in rows]) if rows else np.zeros(
        (0, _N_ERR), dtype=np.float32)
    has_next = np.zeros(n_rows, dtype=bool)
    for index in range(n_rows - 1):
        left, right = rows[index], rows[index + 1]
        has_next[index] = (
            left["segment"] == right["segment"]
            and left["cls"] == right["cls"]
            and left["id"] == right["id"]
            and right["frame"] == left["frame"] + 1
        )
    scale = np.ones((3, 3, _N_ERR), dtype=np.float64)
    for slot in range(3):
        for band_index in range(3):
            hits = [
                r["err"] for r in rows
                if not r["missed"] and CLASS_SLOTS[r["cls"]] == slot
                and r["band"] == band_index
            ]
            if not hits:
                continue
            matrix = np.stack(hits)
            for comp in range(_N_ERR):
                spread = (
                    np.percentile(matrix[:, comp], 84)
                    - np.percentile(matrix[:, comp], 16)
                ) / 2.0
                scale[slot, band_index, comp] = (
                    1.0 if spread < 1e-9 else float(spread))
    return DonorLibrary(
        seg_names, segment, cls, band, missed, flip, err, has_next, scale)


def exclude_segment(library: DonorLibrary, segment: str) -> DonorLibrary:
    """Return a copy of the library with one segment's rows removed."""
    if segment not in library.segments:
        return library
    drop = library.segments.index(segment)
    keep = library.segment != drop
    new_names = tuple(name for name in library.segments if name != segment)
    remap = {
        old: new for new, old in enumerate(
            index for index, name in enumerate(library.segments)
            if name != segment)
    }
    new_segment = np.array(
        [remap[int(value)] for value in library.segment[keep]], dtype=np.int16)
    return DonorLibrary(
        new_names,
        new_segment,
        library.cls[keep],
        library.band[keep],
        library.missed[keep],
        library.flip[keep],
        library.err[keep],
        library.has_next[keep],
        library.scale,
    )


def validate_library(library: DonorLibrary) -> None:
    """Ensure every class band has a continuing hit row for donor hops."""
    if len(library.cls) == 0:
        raise ValueError("donor library has no rows")
    classes_present = {int(value) for value in library.cls}
    next_band = np.append(library.band[1:], -1)
    for cls in sorted(classes_present):
        if cls not in CLASS_SLOTS:
            continue
        mask = library.cls == cls
        for band_index in range(3):
            hit_rows = np.where(mask & (library.band == band_index)
                                & ~library.missed)[0]
            ok = bool(np.any(library.has_next[hit_rows]
                             & (next_band[hit_rows] == band_index)))
            if not ok:
                raise ValueError(
                    f"class {cls} band {band_index} has no donor rows")


def write_library(library: DonorLibrary, path: Path) -> None:
    np.savez_compressed(
        path,
        segments=np.asarray(library.segments),
        segment=library.segment,
        cls=library.cls,
        band=library.band,
        missed=library.missed,
        flip=library.flip,
        err=library.err,
        has_next=library.has_next,
        scale=library.scale,
    )


def read_library(path: Path) -> DonorLibrary:
    with np.load(path, allow_pickle=True) as archive:
        segments = tuple(str(value) for value in archive["segments"].tolist())
        library = DonorLibrary(
            segments,
            archive["segment"],
            archive["cls"],
            archive["band"],
            archive["missed"],
            archive["flip"],
            archive["err"],
            archive["has_next"],
            archive["scale"],
        )
    validate_library(library)
    return library


@functools.cache
def default_library() -> DonorLibrary:
    return read_library(DEFAULT_LIBRARY_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("segments", nargs="+")
    args = parser.parse_args()
    frames = []
    for name in args.segments:
        frames.extend(segment_frames(
            args.parquet_root, args.detections, name))
    library = build_library(frames, args.gate_scale)
    write_library(library, args.out)
    print(f"wrote {len(library.cls)} donor rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
