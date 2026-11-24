"""Scores exported 3 s forecasts against Waymo labels with per-class NEES."""

import argparse
import csv
import os

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from waymo_boxes import TRACKED_OBJECT_CLASSES, read_ground_truth_boxes

# The forecast horizon and how closely a lidar frame must match t + 3 s.
FORECAST_HORIZON_MICROS = 3_000_000
TIME_TOLERANCE_MICROS = 50_000
MATCH_DISTANCE_METRES = 1.0
CHI_SQUARED_95_TWO_DOF = 5.991
SPEED_BAND_EDGES = np.array([0.5, 5.0, 15.0])
SPEED_BAND_LABELS = ("< 0.5", "0.5 - 5", "5 - 15", ">= 15")


def assign_speed_bands(speeds):
    """Returns band index 0..3 for speeds in m/s at time t."""
    return np.digitize(speeds, SPEED_BAND_EDGES, right=False)


def read_vehicle_to_world_poses(segment_path):
    """Returns {capture time in microseconds: 4x4 vehicle-to-world matrix}."""
    poses = {}
    with open(segment_path, "rb") as stream:
        reader = make_reader(stream, decoder_factories = [DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages():
            if channel.topic != "/frames":
                continue
            pose = np.array(message.vehicle_to_world, dtype = np.float64)
            poses[message.capture_time_micros] = pose.reshape(4, 4)
    return poses


def label_to_world_xy(box, vehicle_to_world):
    """Returns the label box centre (x, y) in the world frame."""
    local = np.array([box.center_x, box.center_y, 0.0, 1.0], dtype = np.float64)
    world = vehicle_to_world @ local
    return float(world[0]), float(world[1])


def build_label_indices(ground_truth, poses):
    """Builds per-frame label arrays and a (time, id) position lookup."""
    by_time = {}
    by_id = {}
    for label in ground_truth:
        if label.object_class not in TRACKED_OBJECT_CLASSES:
            continue
        if label.num_lidar_points_in_box <= 0:
            continue
        capture_time = label.capture_time_micros
        if capture_time not in poses:
            continue
        world_x, world_y = label_to_world_xy(label.box, poses[capture_time])
        by_time.setdefault(capture_time, []).append((
            label.object_class,
            label.laser_object_id,
            world_x,
            world_y,
        ))
        by_id[(capture_time, label.laser_object_id)] = (world_x, world_y)
    per_frame = {}
    for capture_time, rows in by_time.items():
        table = np.array(rows, dtype = object)
        per_frame[capture_time] = (
            table[:, 0].astype(np.int64),
            table[:, 1],
            table[:, 2].astype(np.float64),
            table[:, 3].astype(np.float64),
        )
    return per_frame, by_id


def nearest_capture_time(sorted_times, target_micros, tolerance_micros):
    """Returns the staged capture time within tolerance of target, or None."""
    index = np.searchsorted(sorted_times, target_micros)
    candidates = []
    if index < len(sorted_times):
        candidates.append(sorted_times[index])
    if index > 0:
        candidates.append(sorted_times[index - 1])
    for candidate in candidates:
        if abs(candidate - target_micros) <= tolerance_micros:
            return int(candidate)
    return None


def mutual_nearest_matches(track_xy, track_classes, label_classes, label_xy):
    """Returns matched (track_index, label_index) pairs within 1m, per class."""
    track_count = track_xy.shape[0]
    label_count = label_xy.shape[0]
    if track_count == 0 or label_count == 0:
        return []
    offsets = track_xy[:, None, :] - label_xy[None, :, :]
    distances = np.linalg.norm(offsets, axis = 2)
    same_class = track_classes[:, None] == label_classes[None, :]
    distances[~same_class] = np.inf
    nearest_label = np.argmin(distances, axis = 1)
    nearest_track = np.argmin(distances, axis = 0)
    track_indices = np.arange(track_count)
    mutual = nearest_track[nearest_label] == track_indices
    mutual &= np.isfinite(distances[track_indices, nearest_label])
    mutual &= distances[track_indices, nearest_label] <= MATCH_DISTANCE_METRES
    pairs = [(int(track_indices[i]), int(nearest_label[i]))
             for i in range(track_count) if mutual[i]]
    return pairs


def load_forecast_rows(forecast_path):
    """Returns forecast CSV rows grouped by capture time."""
    rows = []
    with open(forecast_path, newline = "") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            parsed = (
                int(row["frame_timestamp_micros"]),
                int(row["track_id"]),
                int(row["object_class"]),
                float(row["x"]),
                float(row["y"]),
                float(row["speed"]),
                float(row["forecast_x"]),
                float(row["forecast_y"]),
                float(row["cov_xx"]),
                float(row["cov_xy"]),
                float(row["cov_yy"]),
            )
            rows.append(parsed)
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row[0], []).append(row)
    return by_frame


def nees_values(errors, cov_xx, cov_xy, cov_yy):
    """Returns e^T S^-1 e for each row, vectorised over N (2,) errors."""
    ex = errors[:, 0]
    ey = errors[:, 1]
    determinant = cov_xx * cov_yy - cov_xy * cov_xy
    numerator = (cov_yy * ex * ex - 2.0 * cov_xy * ex * ey +
                 cov_xx * ey * ey)
    return numerator / determinant


def gather_frame_matches(rows_by_frame, labels_by_time):
    """Returns all rows mutual-nearest-matched to a label at capture time t."""
    frame_times = []
    object_classes = []
    label_ids = []
    speeds = []
    forecast_x = []
    forecast_y = []
    cov_xx = []
    cov_xy = []
    cov_yy = []
    for frame_time, frame_rows in rows_by_frame.items():
        if frame_time not in labels_by_time:
            continue
        classes, ids, label_x, label_y = labels_by_time[frame_time]
        label_xy = np.stack([label_x, label_y], axis = 1)
        track_xy = np.array([[row[3], row[4]] for row in frame_rows],
                            dtype = np.float64)
        track_classes = np.array([row[2] for row in frame_rows],
                                 dtype = np.int64)
        pairs = mutual_nearest_matches(track_xy, track_classes, classes,
                                       label_xy)
        for track_index, label_index in pairs:
            row = frame_rows[track_index]
            frame_times.append(frame_time)
            object_classes.append(row[2])
            label_ids.append(ids[label_index])
            speeds.append(row[5])
            forecast_x.append(row[6])
            forecast_y.append(row[7])
            cov_xx.append(row[8])
            cov_xy.append(row[9])
            cov_yy.append(row[10])
    if not frame_times:
        empty = np.array([], dtype = np.int64)
        empty_f = np.array([], dtype = np.float64)
        return (empty, empty, [], empty_f, empty_f, empty_f, empty_f,
                empty_f, empty_f)
    return (
        np.array(frame_times, dtype = np.int64),
        np.array(object_classes, dtype = np.int64),
        label_ids,
        np.array(speeds, dtype = np.float64),
        np.array(forecast_x, dtype = np.float64),
        np.array(forecast_y, dtype = np.float64),
        np.array(cov_xx, dtype = np.float64),
        np.array(cov_xy, dtype = np.float64),
        np.array(cov_yy, dtype = np.float64),
    )


def resolve_future_positions(frame_times, label_ids, positions, sorted_times):
    """Maps (t, id) to ground-truth (x, y) at the nearest frame to t + 3 s."""
    count = frame_times.shape[0]
    truth_x = np.full(count, np.nan, dtype = np.float64)
    truth_y = np.full(count, np.nan, dtype = np.float64)
    for index in range(count):
        target_time = int(frame_times[index]) + FORECAST_HORIZON_MICROS
        future_time = nearest_capture_time(sorted_times, target_time,
                                           TIME_TOLERANCE_MICROS)
        if future_time is None:
            continue
        position = positions.get((future_time, label_ids[index]))
        if position is None:
            continue
        truth_x[index], truth_y[index] = position
    valid = np.isfinite(truth_x)
    return truth_x, truth_y, valid


def score_forecasts(rows_by_frame, labels_by_time, positions, sorted_times):
    """Scores matched rows whose label id reappears within 50 ms of t + 3 s."""
    (frame_times, object_classes, label_ids, speeds, forecast_x, forecast_y,
     cov_xx, cov_xy, cov_yy) = gather_frame_matches(rows_by_frame,
                                                    labels_by_time)
    matched_at_t = {
        int(object_class): int(np.count_nonzero(object_classes == object_class))
        for object_class in TRACKED_OBJECT_CLASSES
    }
    if frame_times.size == 0:
        return (object_classes, np.array([], dtype = np.float64),
                np.array([], dtype = np.float64),
                np.array([], dtype = np.float64), matched_at_t,
                {object_class: 0 for object_class in TRACKED_OBJECT_CLASSES})

    truth_x, truth_y, valid = resolve_future_positions(
        frame_times, label_ids, positions, sorted_times)
    errors = np.column_stack([truth_x - forecast_x, truth_y - forecast_y])
    scored_nees = nees_values(errors[valid], cov_xx[valid], cov_xy[valid],
                              cov_yy[valid])
    scored_errors = np.linalg.norm(errors[valid], axis = 1)
    scored_classes = object_classes[valid]
    scored_speeds = speeds[valid]
    matched_at_future = {
        int(object_class):
        int(np.count_nonzero(scored_classes == object_class))
        for object_class in TRACKED_OBJECT_CLASSES
    }
    return (scored_classes, scored_nees, scored_errors, scored_speeds,
            matched_at_t, matched_at_future)


def print_speed_band_summary(classes, nees, errors, speeds):
    """Prints per-class NEES summary broken down by speed band at time t."""
    bands = assign_speed_bands(speeds)
    for object_class in TRACKED_OBJECT_CLASSES:
        class_mask = classes == object_class
        for band_index, band_label in enumerate(SPEED_BAND_LABELS):
            mask = class_mask & (bands == band_index)
            count = int(np.count_nonzero(mask))
            if count == 0:
                print(f"class {object_class} band {band_label}: "
                      f"rows scored 0")
                continue
            band_nees = nees[mask]
            band_errors = errors[mask]
            fraction = float(np.mean(band_nees <= CHI_SQUARED_95_TWO_DOF))
            median_error = float(np.median(band_errors))
            mean_nees = float(np.mean(band_nees))
            print(f"class {object_class} band {band_label}: "
                  f"rows scored {count} "
                  f"mean NEES {mean_nees:.4f} "
                  f"fraction <= {CHI_SQUARED_95_TWO_DOF} {fraction:.4f} "
                  f"median |e| {median_error:.4f}")


def print_class_summary(classes, nees, errors, at_t, at_future):
    """Prints per-class match counts, NEES summary, and median |e|."""
    for object_class in TRACKED_OBJECT_CLASSES:
        matched_t = at_t.get(object_class, 0)
        matched_future = at_future.get(object_class, 0)
        mask = classes == object_class
        count = int(np.count_nonzero(mask))
        if count == 0:
            print(f"class {object_class}: matched at t {matched_t} "
                  f"label at t+3s {matched_future} rows scored 0")
            continue
        class_nees = nees[mask]
        class_errors = errors[mask]
        fraction = float(np.mean(class_nees <= CHI_SQUARED_95_TWO_DOF))
        median_error = float(np.median(class_errors))
        mean_nees = float(np.mean(class_nees))
        print(f"class {object_class}: matched at t {matched_t} "
              f"label at t+3s {matched_future} rows scored {count} "
              f"mean NEES {mean_nees:.4f} "
              f"fraction <= {CHI_SQUARED_95_TWO_DOF} {fraction:.4f} "
              f"median |e| {median_error:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required = True)
    parser.add_argument("--parquet-root", required = True)
    parser.add_argument("--forecast", required = True)
    arguments = parser.parse_args()
    segment_path = arguments.segment
    segment_name = os.path.basename(segment_path)
    if segment_name.endswith(".mcap"):
        segment_name = segment_name[:-5]

    poses = read_vehicle_to_world_poses(segment_path)
    sorted_times = np.array(sorted(poses), dtype = np.int64)
    ground_truth = read_ground_truth_boxes(arguments.parquet_root, segment_name)
    labels_by_time, label_positions = build_label_indices(ground_truth, poses)
    rows_by_frame = load_forecast_rows(arguments.forecast)
    (scored_classes, scored_nees, scored_errors, scored_speeds, matched_at_t,
     matched_at_future) = score_forecasts(
         rows_by_frame, labels_by_time, label_positions, sorted_times)
    print_class_summary(scored_classes, scored_nees, scored_errors,
                        matched_at_t, matched_at_future)
    print_speed_band_summary(scored_classes, scored_nees, scored_errors,
                             scored_speeds)


if __name__ == "__main__":
    main()
