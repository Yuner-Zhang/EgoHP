#!/usr/bin/env python3
"""Generate adaptive temporal labels from geometry, person tracks, and IMU."""

import argparse
import bisect
import csv
import gzip
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def load_numeric_rows(path: Path, fields: Sequence[str]) -> List[Dict[str, float]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in fields):
            raise RuntimeError(f"Unexpected columns in {path.name}; required: {', '.join(fields)}")
        for source in reader:
            try:
                rows.append({field: float(source[field]) for field in fields})
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda row: row[fields[0]])
    return rows


def rows_between(rows, timestamps, start_ns: int, end_ns: int):
    return rows[
        bisect.bisect_left(timestamps, start_ns) : bisect.bisect_right(timestamps, end_ns)
    ]


def quaternion_rotation_angle(first: Dict[str, float], second: Dict[str, float]) -> float:
    """Return the axis-independent shortest relative orientation angle in radians."""
    first_q = [first[key] for key in ("qx", "qy", "qz", "qw")]
    second_q = [second[key] for key in ("qx", "qy", "qz", "qw")]
    first_norm = math.sqrt(sum(value * value for value in first_q))
    second_norm = math.sqrt(sum(value * value for value in second_q))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    dot = abs(
        sum(a * b for a, b in zip(first_q, second_q)) / (first_norm * second_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def camera_motion_features(
    trajectory, trajectory_ts, imu, imu_ts, start_ns, end_ns, vertical_reliable
):
    poses = rows_between(trajectory, trajectory_ts, start_ns, end_ns)
    imu_rows = rows_between(imu, imu_ts, start_ns, end_ns)
    sampled = []
    for pose in poses:
        if not sampled or pose["timestamp_ns"] - sampled[-1]["timestamp_ns"] >= 50_000_000:
            sampled.append(pose)
    if poses and sampled[-1] is not poses[-1]:
        sampled.append(poses[-1])

    path_length = 0.0
    speeds = []
    for previous, current in zip(sampled, sampled[1:]):
        distance = math.dist(
            (previous["tx"], previous["ty"], previous["tz"]),
            (current["tx"], current["ty"], current["tz"]),
        )
        path_length += distance
        duration = (current["timestamp_ns"] - previous["timestamp_ns"]) / 1e9
        if duration > 0:
            speeds.append(distance / duration)

    rotation_steps = []
    if sampled:
        anchor = sampled[0]
        for current in sampled[1:]:
            if current["timestamp_ns"] - anchor["timestamp_ns"] < 250_000_000:
                continue
            step = quaternion_rotation_angle(anchor, current)
            if abs(math.degrees(step)) >= 1.0:
                rotation_steps.append(step)
            anchor = current

    horizontal = vertical = orientation_change = 0.0
    if len(sampled) >= 2:
        horizontal = math.hypot(
            sampled[-1]["tx"] - sampled[0]["tx"],
            sampled[-1]["ty"] - sampled[0]["ty"],
        )
        vertical = sampled[-1]["tz"] - sampled[0]["tz"]
        orientation_change = math.degrees(
            quaternion_rotation_angle(sampled[0], sampled[-1])
        )

    gyro_sq = [row["w_x"] ** 2 + row["w_y"] ** 2 + row["w_z"] ** 2 for row in imu_rows]
    accel_sq = []
    for row in imu_rows:
        norm = math.sqrt(row["a_x"] ** 2 + row["a_y"] ** 2 + row["a_z"] ** 2)
        accel_sq.append((norm - 9.80665) ** 2)
    return {
        "trajectory_samples": len(poses),
        "imu_samples": len(imu_rows),
        "path_length_m": round(path_length, 4),
        "horizontal_displacement_m": round(horizontal, 4),
        "vertical_displacement_m": round(vertical, 4),
        "vertical_axis_reliable": bool(vertical_reliable),
        "median_speed_m_s": round(statistics.median(speeds), 4) if speeds else None,
        "p95_speed_m_s": round(percentile(speeds, 0.95), 4) if speeds else None,
        "orientation_change_deg": round(orientation_change, 2),
        "accumulated_rotation_deg": round(
            sum(rotation_steps) * 180 / math.pi, 2
        ),
        "gyro_rms_rad_s": round(math.sqrt(statistics.mean(gyro_sq)), 4) if gyro_sq else None,
        "dynamic_accel_rms_m_s2": round(math.sqrt(statistics.mean(accel_sq)), 4)
        if accel_sq
        else None,
    }


def classify_camera_motion(features: Dict[str, object]) -> Optional[str]:
    if int(features["trajectory_samples"]) < 2 or int(features["imu_samples"]) < 2:
        return None
    path = float(features["path_length_m"])
    horizontal = float(features["horizontal_displacement_m"])
    vertical = abs(float(features["vertical_displacement_m"]))
    speed = float(features["median_speed_m_s"] or 0)
    p95 = float(features["p95_speed_m_s"] or 0)
    accel_value = features["dynamic_accel_rms_m_s2"]
    accel = float(accel_value) if accel_value is not None else float("inf")
    vertical_reliable = bool(features["vertical_axis_reliable"])
    if vertical_reliable and vertical >= 1.0 and horizontal < 0.80 and accel < 0.80:
        return "elevator"
    if vertical_reliable and vertical >= 0.60 and accel >= 0.80:
        return "stairs"
    if speed > 2.20 or p95 > 3.0 or accel > 3.0:
        return "rapid_motion"
    if float(features["orientation_change_deg"]) >= 15 or float(
        features["accumulated_rotation_deg"]
    ) >= 30:
        return "turning"
    if path >= 0.25 or speed >= 0.10:
        return "walking"
    return "stationary"


def distance_level(distance_m: Optional[float]) -> Optional[str]:
    if distance_m is None:
        return None
    if distance_m < 1:
        return "very_close"
    if distance_m < 3:
        return "close"
    if distance_m < 10:
        return "medium"
    if distance_m <= 30:
        return "far"
    return "very_far"


def frame_track_evidence(frames, start_ns, end_ns, image_width, metric_depths):
    selected = [
        frame for frame in frames if start_ns <= int(frame["timestamp_ns"]) <= end_ns
    ]
    tracks: Dict[int, List[Dict[str, float]]] = {}
    for frame in selected:
        frame_id = int(frame["frame_id"])
        for person in frame.get("persons", []):
            person_id = int(person["person_id"])
            bbox = person["bbox"]
            width = max(1.0, float(bbox[2]) - float(bbox[0]))
            height = max(1.0, float(bbox[3]) - float(bbox[1]))
            geometry = metric_depths.get(frame_id, {}).get(person_id)
            distance = geometry.get("distance_m") if geometry is not None else None
            tracks.setdefault(person_id, []).append(
                {
                    "timestamp_ns": float(frame["timestamp_ns"]),
                    "area": width * height,
                    "center_x": (float(bbox[0]) + float(bbox[2])) / 2,
                    "distance_m": float(distance) if distance is not None else float("nan"),
                    "person_world": geometry.get("person_world") if geometry else None,
                    "camera_world": geometry.get("camera_world") if geometry else None,
                    "ray_rgb": geometry.get("ray_rgb") if geometry else None,
                }
            )

    summaries = []
    for person_id, observations in tracks.items():
        if len(observations) < 2:
            continue
        first, last = observations[0], observations[-1]
        linear_scales = [math.sqrt(row["area"]) for row in observations]
        center_fractions = [row["center_x"] / max(1, image_width) for row in observations]
        peak_scale_index = max(range(len(linear_scales)), key=linear_scales.__getitem__)
        edge_scale = max(1e-6, (linear_scales[0] + linear_scales[-1]) / 2)
        item = {
            "person_id": person_id,
            "observations": len(observations),
            "observation_fraction": round(len(observations) / max(1, len(selected)), 3),
            "median_bbox_area": round(statistics.median(row["area"] for row in observations), 1),
            "bbox_scale_end_over_start": round(math.sqrt(last["area"] / first["area"]), 3),
            "bbox_scale_peak_over_edges": round(max(linear_scales) / edge_scale, 3),
            "bbox_scale_peak_is_internal": 0 < peak_scale_index < len(linear_scales) - 1,
            "horizontal_center_change_image_fraction": round(
                (last["center_x"] - first["center_x"]) / max(1, image_width), 3
            ),
            "horizontal_center_span_image_fraction": round(
                max(center_fractions) - min(center_fractions), 3
            ),
        }
        valid = [row for row in observations if not math.isnan(row["distance_m"])]
        item["metric_samples"] = len(valid)
        if valid:
            item["nearest_distance_m"] = round(
                float(percentile([row["distance_m"] for row in valid], 0.10)), 3
            )
        if len(valid) >= 2:
            change = valid[-1]["distance_m"] - valid[0]["distance_m"]
            duration = (valid[-1]["timestamp_ns"] - valid[0]["timestamp_ns"]) / 1e9
            item["distance_change_m"] = round(change, 3)
            item["radial_speed_m_s"] = round(change / duration, 3) if duration > 0 else None
            closest_index = min(
                range(len(valid)), key=lambda index: valid[index]["distance_m"]
            )
            item["approach_then_recede"] = bool(
                0 < closest_index < len(valid) - 1
                and valid[0]["distance_m"] - valid[closest_index]["distance_m"] >= 0.50
                and valid[-1]["distance_m"] - valid[closest_index]["distance_m"] >= 0.50
            )

            geometry_rows = [
                row
                for row in valid
                if row["person_world"] is not None
                and row["camera_world"] is not None
                and row["ray_rgb"] is not None
            ]
            if len(geometry_rows) >= 2:
                geometry_first, geometry_last = geometry_rows[0], geometry_rows[-1]

                def subtract(left, right):
                    return [float(a) - float(b) for a, b in zip(left, right)]

                def vector_norm(vector):
                    return math.sqrt(sum(value * value for value in vector))

                person_displacement = subtract(
                    geometry_last["person_world"], geometry_first["person_world"]
                )
                camera_displacement = subtract(
                    geometry_last["camera_world"], geometry_first["camera_world"]
                )
                person_norm = vector_norm(person_displacement)
                camera_norm = vector_norm(camera_displacement)
                item["person_world_displacement_m"] = round(person_norm, 3)
                item["camera_world_displacement_m"] = round(camera_norm, 3)
                if person_norm > 1e-6 and camera_norm > 1e-6:
                    cosine = sum(
                        a * b for a, b in zip(person_displacement, camera_displacement)
                    ) / (person_norm * camera_norm)
                    item["person_camera_direction_cosine"] = round(
                        max(-1.0, min(1.0, cosine)), 3
                    )
                    relative_first = subtract(
                        geometry_first["person_world"], geometry_first["camera_world"]
                    )
                    relative_last = subtract(
                        geometry_last["person_world"], geometry_last["camera_world"]
                    )
                    relative_change = subtract(relative_last, relative_first)
                    camera_unit = [value / camera_norm for value in camera_displacement]
                    axial = sum(
                        value * direction
                        for value, direction in zip(relative_change, camera_unit)
                    )
                    lateral = subtract(
                        relative_change, [axial * value for value in camera_unit]
                    )
                    item["relative_lateral_displacement_m"] = round(
                        vector_norm(lateral), 3
                    )
                first_ray = geometry_first["ray_rgb"]
                last_ray = geometry_last["ray_rgb"]
                first_ray_norm, last_ray_norm = vector_norm(first_ray), vector_norm(last_ray)
                if first_ray_norm > 1e-6 and last_ray_norm > 1e-6:
                    ray_cosine = sum(a * b for a, b in zip(first_ray, last_ray)) / (
                        first_ray_norm * last_ray_norm
                    )
                    item["bearing_change_deg"] = round(
                        math.degrees(
                            math.acos(max(-1.0, min(1.0, ray_cosine)))
                        ),
                        2,
                    )
        summaries.append(item)
    return {
        "visual_track_changes": summaries,
    }


def classify_person_relative_motion(track, camera_motion) -> str:
    """Return one motion state; sub-threshold evidence is stationary."""
    states = []
    metric = int(track.get("metric_samples", 0)) >= 2
    camera_displacement = float(track.get("camera_world_displacement_m", 0.0))
    person_displacement = float(track.get("person_world_displacement_m", 0.0))
    direction_cosine = track.get("person_camera_direction_cosine")
    change = track.get("distance_change_m")
    speed = track.get("radial_speed_m_s")
    scale = float(track["bbox_scale_end_over_start"])
    horizontal_change = abs(float(track["horizontal_center_change_image_fraction"]))
    horizontal_span = float(track["horizontal_center_span_image_fraction"])

    if (
        direction_cosine is not None
        and camera_displacement >= 0.50
        and person_displacement >= 0.50
    ):
        if float(direction_cosine) >= 0.50 and (
            change is None or abs(float(change)) < 0.50
        ):
            states.append("same_direction")
        elif float(direction_cosine) <= -0.50:
            states.append("opposite_direction")
    elif (
        camera_displacement >= 0.50
        and (
            bool(track.get("approach_then_recede"))
            or (
                bool(track.get("bbox_scale_peak_is_internal"))
                and float(track.get("bbox_scale_peak_over_edges", 1.0)) >= 1.25
                and horizontal_span >= 0.10
            )
        )
    ):
        states.append("opposite_direction")

    bearing = float(track.get("bearing_change_deg", 0.0))
    lateral = float(track.get("relative_lateral_displacement_m", 0.0))
    visual_crossing = (
        camera_motion != "turning"
        and horizontal_span >= 0.20
        and 0.80 <= scale <= 1.25
    )
    if (
        (camera_motion != "turning" and bearing >= 20.0)
        or lateral >= 1.0
        or visual_crossing
    ):
        states.append("crossing")

    if metric and change is not None and speed is not None:
        if float(change) <= -0.50 and float(speed) <= -0.15:
            states.append("approaching")
        elif float(change) >= 0.50 and float(speed) >= 0.15:
            states.append("receding")
    else:
        if scale >= 1.25:
            states.append("approaching")
        elif scale <= 0.80:
            states.append("receding")

    if not states:
        stable_metric = (
            metric
            and change is not None
            and speed is not None
            and abs(float(change)) < 0.30
            and abs(float(speed)) < 0.20
        )
        stable_visual = (
            0.90 <= scale <= 1.10
            and horizontal_change < 0.05
        )
        if stable_metric or stable_visual:
            if camera_motion == "stationary" or camera_displacement < 0.25:
                states.append("stationary")
            elif camera_motion in {"walking", "stairs", "elevator", "rapid_motion"}:
                states.append("same_direction")

    priority = (
        "opposite_direction",
        "same_direction",
        "approaching",
        "receding",
        "crossing",
        "stationary",
    )
    return next((state for state in priority if state in states), "stationary")


def load_vrs(sequence: Path, cpu_decode: bool):
    if cpu_decode:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    try:
        from projectaria_tools.core import data_provider
    except ImportError as error:
        raise RuntimeError("Install environment.yml in Ubuntu 20.04 first") from error
    recording_vrs = sequence / f"{sequence.name}.vrs"
    provider = data_provider.create_vrs_data_provider(str(recording_vrs.resolve()))
    if provider is None:
        raise RuntimeError(f"Cannot open sequence recording: {recording_vrs.name}")
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    return provider, stream_id


def semidense_metric_depths(args, provider, frames):
    """Project confidence-filtered MPS world points into RGB person boxes when MPS exists."""
    slam = args.sequence / "mps" / "slam"
    required = [
        slam / "closed_loop_trajectory.csv",
        slam / "semidense_points.csv.gz",
        slam / "semidense_observations.csv.gz",
    ]
    if not all(path.is_file() for path in required):
        print("Metric person depth unavailable: mps/slam point files are missing")
        return {}
    try:
        import numpy as np
        from projectaria_tools.core import mps
    except ImportError as error:
        raise RuntimeError("Project Aria MPS Python APIs are unavailable") from error

    candidates = [
        frame
        for frame in frames[:: args.depth_frame_stride]
        if frame.get("persons")
    ]
    if not candidates:
        print("Metric person depth unavailable: no sampled frames contain people")
        return {}
    target_timestamps = [int(frame["timestamp_ns"]) for frame in candidates]
    observed_uids: Dict[int, List[int]] = {timestamp: [] for timestamp in target_timestamps}
    slam_serial = (
        provider.get_device_calibration()
        .get_camera_calib("slam-front-left")
        .get_serial_number()
    )
    # The Pilot Dataset observation file is several GB compressed.  The MPS
    # reader materializes it, which can exhaust an 8 GB workstation.  Stream
    # the timestamp-sorted CSV and retain only points near sampled RGB frames.
    tolerance_ns = int(args.depth_timestamp_tolerance_ms * 1_000_000)
    selected_uids = set()
    observation_rows = 0
    previous_timestamp = None
    target_for_timestamp = None
    with gzip.open(required[2], "rt", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        try:
            uid_index = header.index("uid")
            timestamp_index = header.index("frame_tracking_timestamp_us")
            serial_index = header.index("camera_serial")
        except ValueError as error:
            raise RuntimeError("Unsupported semidense observation columns") from error
        for line in handle:
            observation_rows += 1
            if observation_rows % 10_000_000 == 0:
                print(
                    f"  streamed MPS observations: {observation_rows:,}; "
                    f"selected UIDs: {len(selected_uids):,}",
                    flush=True,
                )
            values = line.rstrip("\r\n").split(",")
            timestamp = int(values[timestamp_index]) * 1000
            if target_timestamps and timestamp > target_timestamps[-1] + tolerance_ns:
                break
            if timestamp != previous_timestamp:
                previous_timestamp = timestamp
                index = bisect.bisect_left(target_timestamps, timestamp)
                nearest = [
                    item
                    for item in (index - 1, index)
                    if 0 <= item < len(target_timestamps)
                ]
                target_for_timestamp = (
                    target_timestamps[
                        min(nearest, key=lambda item: abs(target_timestamps[item] - timestamp))
                    ]
                    if nearest
                    else None
                )
                if (
                    target_for_timestamp is not None
                    and abs(target_for_timestamp - timestamp) > tolerance_ns
                ):
                    target_for_timestamp = None
            if target_for_timestamp is None or values[serial_index] != slam_serial:
                continue
            bucket = observed_uids[target_for_timestamp]
            if len(bucket) < args.max_points_per_frame:
                uid = int(values[uid_index])
                bucket.append(uid)
                selected_uids.add(uid)

    # Make a second, much smaller streaming pass over global points and retain
    # coordinates only for UIDs selected above and passing MPS uncertainty.
    point_positions = {}
    with gzip.open(required[1], "rt", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        required_columns = (
            "uid",
            "px_world",
            "py_world",
            "pz_world",
            "inv_dist_std",
            "dist_std",
        )
        try:
            indices = {name: header.index(name) for name in required_columns}
        except ValueError as error:
            raise RuntimeError("Unsupported semidense point columns") from error
        for line in handle:
            values = line.rstrip("\r\n").split(",")
            uid = int(values[indices["uid"]])
            if uid not in selected_uids:
                continue
            if (
                float(values[indices["inv_dist_std"]])
                > args.max_inverse_distance_std
                or float(values[indices["dist_std"]]) > args.max_distance_std
            ):
                continue
            point_positions[uid] = np.asarray(
                [
                    float(values[indices["px_world"]]),
                    float(values[indices["py_world"]]),
                    float(values[indices["pz_world"]]),
                ]
            )
    print(
        f"MPS depth preparation: {observation_rows:,} observations streamed, "
        f"{len(selected_uids):,} UIDs sampled, "
        f"{len(point_positions):,} confidence-filtered points",
        flush=True,
    )

    poses = mps.read_closed_loop_trajectory(str(required[0]))
    pose_timestamps = [int(pose.tracking_timestamp.total_seconds() * 1e9) for pose in poses]

    rgb_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    transform_device_rgb = rgb_calib.get_transform_device_camera()
    output = {}
    for frame, timestamp in zip(candidates, target_timestamps):
        if not observed_uids[timestamp] or not poses:
            continue
        pose_index = bisect.bisect_left(pose_timestamps, timestamp)
        options = [item for item in (pose_index - 1, pose_index) if 0 <= item < len(poses)]
        pose = poses[min(options, key=lambda item: abs(pose_timestamps[item] - timestamp))]
        distances: Dict[int, List[float]] = {
            int(person["person_id"]): [] for person in frame["persons"]
        }
        for uid in observed_uids[timestamp]:
            point_world = point_positions.get(uid)
            if point_world is None:
                continue
            point_device = pose.transform_world_device.inverse() @ point_world
            point_rgb = transform_device_rgb.inverse() @ point_device
            uv = rgb_calib.project(point_rgb)
            if uv is None:
                continue
            u, v = float(uv[0]), float(uv[1])
            for person in frame["persons"]:
                x1, y1, x2, y2 = map(float, person["bbox"])
                if x1 <= u <= x2 and y1 <= v <= y2:
                    distances[int(person["person_id"])].append(float(np.linalg.norm(point_rgb)))
        camera_world = np.asarray(
            pose.transform_world_device @ np.zeros(3), dtype=np.float64
        ).reshape(-1)
        per_person = {}
        persons_by_id = {
            int(person["person_id"]): person for person in frame["persons"]
        }
        for person_id, values in distances.items():
            if len(values) < args.min_depth_points:
                continue
            distance = percentile(values, 0.20)
            if distance is None:
                continue
            bbox = persons_by_id[person_id]["bbox"]
            center_uv = np.asarray(
                [
                    (float(bbox[0]) + float(bbox[2])) / 2,
                    (float(bbox[1]) + float(bbox[3])) / 2,
                ],
                dtype=np.float64,
            )
            ray_rgb = rgb_calib.unproject(center_uv)
            if ray_rgb is None:
                continue
            ray_rgb = np.asarray(ray_rgb, dtype=np.float64)
            ray_norm = float(np.linalg.norm(ray_rgb))
            if ray_norm <= 1e-9:
                continue
            ray_rgb /= ray_norm
            person_rgb = ray_rgb * float(distance)
            person_device = np.asarray(
                transform_device_rgb @ person_rgb, dtype=np.float64
            ).reshape(-1)
            person_world = np.asarray(
                pose.transform_world_device @ person_device, dtype=np.float64
            ).reshape(-1)
            per_person[person_id] = {
                "distance_m": float(distance),
                "person_world": [float(value) for value in person_world],
                "camera_world": [float(value) for value in camera_world],
                "ray_rgb": [float(value) for value in ray_rgb],
            }
        if per_person:
            output[int(frame["frame_id"])] = per_person
    print(f"Metric person depth frames from MPS: {len(output)}")
    return output


def analyze_interval(
    frames,
    start,
    end,
    image_width,
    metric_depths,
    trajectory,
    trajectory_ts,
    imu,
    imu_ts,
    vertical_reliable,
    min_person_fraction,
):
    """Calculate the clean public labels for one candidate time interval."""
    start_ns = int(frames[start]["timestamp_ns"])
    end_ns = int(frames[end]["timestamp_ns"])
    sensor = camera_motion_features(
        trajectory,
        trajectory_ts,
        imu,
        imu_ts,
        start_ns,
        end_ns,
        vertical_reliable,
    )
    camera_motion = classify_camera_motion(sensor)
    evidence = frame_track_evidence(
        frames, start_ns, end_ns, image_width, metric_depths
    )
    persons = []
    for track in evidence["visual_track_changes"]:
        if float(track["observation_fraction"]) < min_person_fraction:
            continue
        distance = track.get("nearest_distance_m")
        persons.append(
            {
                "person_id": int(track["person_id"]),
                "person_distance_m": (
                    round(float(distance), 3) if distance is not None else None
                ),
                "human_distance_level": distance_level(distance),
                "relative_motion": classify_person_relative_motion(
                    track, camera_motion
                ),
            }
        )
    persons.sort(key=lambda person: person["person_id"])
    return {
        "start_index": start,
        "end_index": end,
        "start_frame": int(frames[start]["frame_id"]),
        "end_frame": int(frames[end]["frame_id"]),
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "persons": persons,
        "camera_motion": camera_motion,
    }


def interval_signature(interval) -> Tuple[object, ...]:
    """Labels that define a dynamic-segmentation boundary."""
    return (
        interval["camera_motion"],
        tuple(
            (
                person["person_id"],
                person["human_distance_level"],
                person["relative_motion"],
            )
            for person in interval["persons"]
        ),
    )


def merge_equal_intervals(base_intervals):
    """Merge only adjacent windows with identical public motion labels."""
    merged = []
    for interval in base_intervals:
        signature = interval_signature(interval)
        if merged and merged[-1]["signature"] == signature:
            merged[-1]["end_index"] = interval["end_index"]
        else:
            merged.append(
                {
                    "start_index": interval["start_index"],
                    "end_index": interval["end_index"],
                    "signature": signature,
                }
            )
    return merged


def run(args) -> None:
    metadata = load_json(args.sequence / "metadata.json")
    labels = load_json(args.sequence / "frame_labels.json")
    frames = labels.get("frames", [])
    if not frames:
        raise RuntimeError("frame_labels.json contains no frames")
    provider, stream_id = load_vrs(args.sequence, args.cpu_decode)
    trajectory_fields = ("timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw")
    imu_fields = ("timestamp_ns", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z")
    trajectory = load_numeric_rows(args.sequence / "trajectory.txt", trajectory_fields)
    imu = load_numeric_rows(args.sequence / "imu_left.txt", imu_fields)
    trajectory_ts = [row["timestamp_ns"] for row in trajectory]
    imu_ts = [row["timestamp_ns"] for row in imu]
    metric_depths = semidense_metric_depths(args, provider, frames)
    vertical_reliable = (
        args.sequence / "mps" / "slam" / "closed_loop_trajectory.csv"
    ).is_file()

    frame_rate = float(metadata["frame_rate"])
    image_width = int(metadata.get("image_width", 2560))
    base_window_frames = max(2, round(args.motion_window_seconds * frame_rate))
    base_ranges = [
        [start, min(len(frames) - 1, start + base_window_frames - 1)]
        for start in range(0, len(frames), base_window_frames)
    ]
    if (
        len(base_ranges) >= 2
        and base_ranges[-1][1] - base_ranges[-1][0] + 1
        < max(2, base_window_frames // 2)
    ):
        base_ranges[-2][1] = base_ranges[-1][1]
        base_ranges.pop()
    base_intervals = []
    for start, end in base_ranges:
        base_intervals.append(
            analyze_interval(
                frames,
                start,
                end,
                image_width,
                metric_depths,
                trajectory,
                trajectory_ts,
                imu,
                imu_ts,
                vertical_reliable,
                args.min_person_window_fraction,
            )
        )

    merged_ranges = merge_equal_intervals(base_intervals)

    segments = []
    for segment_id, merged in enumerate(merged_ranges, start=1):
        interval = analyze_interval(
            frames,
            merged["start_index"],
            merged["end_index"],
            image_width,
            metric_depths,
            trajectory,
            trajectory_ts,
            imu,
            imu_ts,
            vertical_reliable,
            args.min_person_window_fraction,
        )
        interval.pop("start_index")
        interval.pop("end_index")
        segments.append({"segment_id": segment_id, **interval})
        print(f"  dynamic temporal segments: {segment_id}/{len(merged_ranges)}")
    write_json(
        args.sequence / "temporal_labels.json",
        {"sequence_id": metadata["sequence_id"], "segments": segments},
    )
    print(f"Temporal labels written: {args.sequence / 'temporal_labels.json'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument(
        "--motion-window-seconds",
        "--segment-seconds",
        dest="motion_window_seconds",
        type=float,
        default=2.0,
        help=(
            "base analysis window before adjacent equal states are merged; "
            "--segment-seconds is a backward-compatible alias"
        ),
    )
    parser.add_argument("--min-person-window-fraction", type=float, default=0.25)
    parser.add_argument("--cpu-decode", action="store_true")
    parser.add_argument("--depth-frame-stride", type=int, default=10)
    parser.add_argument("--depth-timestamp-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--max-inverse-distance-std", type=float, default=0.005)
    parser.add_argument("--max-distance-std", type=float, default=0.01)
    parser.add_argument("--min-depth-points", type=int, default=5)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    args = parser.parse_args(argv)
    if not args.sequence.is_dir():
        parser.error(f"Sequence does not exist: {args.sequence}")
    required = (
        f"{args.sequence.name}.vrs",
        "metadata.json",
        "frame_labels.json",
        "trajectory.txt",
        "imu_left.txt",
    )
    for name in required:
        if not (args.sequence / name).is_file():
            parser.error(f"Sequence is missing {name}")
    if args.motion_window_seconds <= 0 or args.depth_frame_stride < 1:
        parser.error("motion window seconds and depth stride must be positive")
    if not 0.0 <= args.min_person_window_fraction <= 1.0:
        parser.error("--min-person-window-fraction must be between 0 and 1")
    return args


def main(argv=None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
