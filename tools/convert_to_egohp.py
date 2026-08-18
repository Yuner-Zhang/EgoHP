#!/usr/bin/env python3
"""Convert an Aria Gen 2 VRS and MPS result to an EgoHP staging sequence."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


IMU_HEADER = ["timestamp_ns", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]
TRAJECTORY_HEADER = ["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
REQUIRED_SLAM_FILES = (
    "closed_loop_trajectory.csv",
    "open_loop_trajectory.csv",
    "online_calibration.jsonl",
    "semidense_observations.csv.gz",
    "semidense_points.csv.gz",
)


def normalize_mps_dir(path: Path) -> Path:
    """Return the MPS directory whose direct child is slam/."""
    path = path.resolve()
    if (path / "slam").is_dir():
        return path
    if (path / "mps" / "slam").is_dir():
        return path / "mps"
    raise RuntimeError(f"Cannot find an MPS folder containing slam/: {path}")


def load_aria_modules(cpu_decode: bool):
    if cpu_decode:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
    except ImportError as error:
        raise RuntimeError("Install environment.yml in Ubuntu 20.04 first") from error
    return data_provider, TimeDomain


def required_stream(provider, label: str):
    stream_id = provider.get_stream_id_from_label(label)
    if stream_id is None or not provider.check_stream_is_active(stream_id):
        raise RuntimeError(f"Required VRS stream is missing: {label}")
    return stream_id


def vrs_streams(provider, time_domain):
    rgb_stream = required_stream(provider, "camera-rgb")
    imu_streams = {
        "imu-left": required_stream(provider, "imu-left"),
        "imu-right": required_stream(provider, "imu-right"),
    }
    return rgb_stream, imu_streams


def export_imus(
    provider,
    imu_streams: Dict[str, object],
    output: Path,
    start_ns: int,
    end_ns: int,
) -> Dict[str, int]:
    paths = {
        "imu-left": output / "imu_left.txt",
        "imu-right": output / "imu_right.txt",
    }
    handles = {label: path.open("w", newline="", encoding="utf-8") for label, path in paths.items()}
    try:
        writers = {label: csv.writer(handle) for label, handle in handles.items()}
        for writer in writers.values():
            writer.writerow(IMU_HEADER)
        options = provider.get_default_deliver_queued_options()
        options.deactivate_stream_all()
        for stream_id in imu_streams.values():
            options.activate_stream(stream_id)
        counts = {label: 0 for label in imu_streams}
        for sensor_data in provider.deliver_queued_sensor_data(options):
            label = provider.get_label_from_stream_id(sensor_data.stream_id())
            if label not in writers:
                continue
            sample = sensor_data.imu_data()
            if not sample.gyro_valid or not sample.accel_valid:
                continue
            if not start_ns <= int(sample.capture_timestamp_ns) <= end_ns:
                continue
            writers[label].writerow(
                [
                    int(sample.capture_timestamp_ns),
                    *(f"{value:.9f}" for value in sample.gyro_radsec),
                    *(f"{value:.9f}" for value in sample.accel_msec2),
                ]
            )
            counts[label] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def export_video(
    provider,
    rgb_stream,
    output: Path,
    downsample: int,
    max_frames: Optional[int],
) -> Tuple[int, float, int, int]:
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Install imageio, imageio-ffmpeg, numpy, and Pillow") from error

    fps = float(provider.get_image_configuration(rgb_stream).nominal_rate_hz)
    calibration = provider.get_device_calibration()
    get_version = getattr(calibration, "get_device_version", None)
    rotate_gen1 = get_version is None or "gen1" in str(get_version()).lower()
    options = provider.get_default_deliver_queued_options()
    options.deactivate_stream_all()
    options.activate_stream(rgb_stream)

    count = 0
    first_timestamp_ns: Optional[int] = None
    last_timestamp_ns: Optional[int] = None
    with imageio.get_writer(
        str(output), fps=fps, codec="libx264", macro_block_size=None
    ) as writer:
        for sensor_data in provider.deliver_queued_sensor_data(options):
            image, record = sensor_data.image_data_and_record()
            timestamp_ns = int(record.capture_timestamp_ns)
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            last_timestamp_ns = timestamp_ns
            frame = image.to_numpy_array().copy()
            if downsample > 1:
                height, width = frame.shape[:2]
                frame = np.asarray(
                    Image.fromarray(frame).resize(
                        (max(1, width // downsample), max(1, height // downsample))
                    )
                )
            if rotate_gen1:
                frame = np.rot90(frame, -1)
            writer.append_data(frame)
            count += 1
            if count % 100 == 0:
                print(f"  exported RGB frames: {count}", file=sys.stderr)
            if max_frames is not None and count >= max_frames:
                break
    if first_timestamp_ns is None or last_timestamp_ns is None:
        raise RuntimeError("RGB stream contains no decodable frames")
    return count, fps, first_timestamp_ns, last_timestamp_ns


def trajectory_timestamp_us(row: Dict[str, str]) -> int:
    if "tracking_timestamp_us" in row:
        return int(row["tracking_timestamp_us"])
    if "tracking_timestamp_ns" in row:
        return int(row["tracking_timestamp_ns"]) // 1000
    raise RuntimeError("MPS trajectory lacks tracking_timestamp_us")


def export_trajectory(
    source: Path,
    destination: Path,
    start_ns: int,
    end_ns: int,
) -> Tuple[int, float]:
    closed_loop_fields = (
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "qx_world_device",
        "qy_world_device",
        "qz_world_device",
        "qw_world_device",
    )
    on_device_fields = (
        "tx_odometry_device",
        "ty_odometry_device",
        "tz_odometry_device",
        "qx_odometry_device",
        "qy_odometry_device",
        "qz_odometry_device",
        "qw_odometry_device",
    )
    count = 0
    length = 0.0
    previous: Optional[Tuple[float, float, float]] = None
    with source.open("r", newline="", encoding="utf-8-sig") as src, destination.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        available = set(reader.fieldnames or [])
        if all(field in available for field in closed_loop_fields):
            fields = closed_loop_fields
        elif all(field in available for field in on_device_fields):
            fields = on_device_fields
        else:
            expected = " or ".join(
                [", ".join(closed_loop_fields), ", ".join(on_device_fields)]
            )
            raise RuntimeError(f"Unsupported trajectory columns; expected {expected}")
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("MPS trajectory is missing: " + ", ".join(missing))
        writer = csv.writer(dst)
        writer.writerow(TRAJECTORY_HEADER)
        for row in reader:
            timestamp_ns = trajectory_timestamp_us(row) * 1000
            if not start_ns <= timestamp_ns <= end_ns:
                continue
            position = tuple(float(row[field]) for field in fields[:3])
            quaternion = tuple(float(row[field]) for field in fields[3:])
            if previous is not None:
                length += math.dist(previous, position)
            previous = position
            writer.writerow(
                [timestamp_ns, *(f"{v:.9f}" for v in position), *(f"{v:.9f}" for v in quaternion)]
            )
            count += 1
    if count == 0:
        raise RuntimeError("No trajectory samples overlap the VRS recording")
    return count, length


def place_vrs(source: Path, destination: Path) -> None:
    """Copy the source recording into the portable final sequence."""
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def resolve_collector_id(vrs_path: Path, explicit_id: Optional[int]) -> int:
    """Read collector_id from raw/<collector_id>/<recording>.vrs."""
    folder_name = vrs_path.resolve().parent.name
    inferred_id = int(folder_name) if folder_name.isdecimal() else None
    if explicit_id is not None:
        if inferred_id is not None and explicit_id != inferred_id:
            raise RuntimeError(
                f"collector_id mismatch: --collector-id={explicit_id}, "
                f"but the VRS is under raw/{inferred_id}/"
            )
        return explicit_id
    if inferred_id is None:
        raise RuntimeError(
            "Cannot infer collector_id. Put the VRS under "
            "raw/<collector_id>/ or pass --collector-id explicitly."
        )
    return inferred_id


def convert(args: argparse.Namespace) -> None:
    collector_id = resolve_collector_id(args.vrs, args.collector_id)
    data_provider, time_domain = load_aria_modules(args.cpu_decode)
    provider = data_provider.create_vrs_data_provider(str(args.vrs.resolve()))
    if provider is None:
        raise RuntimeError(f"Cannot open VRS: {args.vrs}")
    rgb_stream, imu_streams = vrs_streams(provider, time_domain.DEVICE_TIME)

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output is not empty: {args.output} (use --overwrite)")
    destination_vrs = args.output / f"{args.sequence_id}.vrs"
    place_vrs(args.vrs, destination_vrs)

    print("Copying complete MPS output...")
    source_mps = normalize_mps_dir(args.mps_dir)
    missing_mps = [
        name for name in REQUIRED_SLAM_FILES if not (source_mps / "slam" / name).is_file()
    ]
    if missing_mps:
        raise RuntimeError(
            "MPS SLAM result is incomplete; missing: " + ", ".join(missing_mps)
        )
    destination_mps = args.output / "mps"
    if destination_mps.exists() and not args.overwrite:
        raise RuntimeError(f"MPS output already exists: {destination_mps}")
    shutil.copytree(source_mps, destination_mps, dirs_exist_ok=args.overwrite)
    trajectory_csv = destination_mps / "slam" / "closed_loop_trajectory.csv"
    if not trajectory_csv.is_file():
        raise RuntimeError(f"Copied MPS output is missing: {trajectory_csv}")

    print(f"Exporting {args.sequence_id}.mp4...")
    frame_count, frame_rate, start_ns, end_ns = export_video(
        provider,
        rgb_stream,
        args.output / f"{args.sequence_id}.mp4",
        args.video_downsample,
        args.max_video_frames,
    )
    print("Exporting IMU TXT files...")
    imu_counts = export_imus(provider, imu_streams, args.output, start_ns, end_ns)
    print("Exporting simplified trajectory.txt...")
    trajectory_count, trajectory_length = export_trajectory(
        trajectory_csv, args.output / "trajectory.txt", start_ns, end_ns
    )

    duration_sec = max(0.0, (end_ns - start_ns) / 1e9)
    metadata = {
        "sequence_id": args.sequence_id,
        # These two values may be left null here. Step 5 infers them with
        # the configured vision API before routing the sequence into the final
        # Indoor/<scene>/seq_xxx or Outdoor/<scene>/seq_xxx directory.
        "scene_name": args.scene_name,
        "environment": args.environment,
        "time_of_day": None,
        "weather": None,
        "crowd_density": None,
        "occlusion_level": None,
        "collector_id": collector_id,
        "num_frames": frame_count,
        "frame_rate": round(frame_rate, 6),
        "duration_sec": round(duration_sec, 6),
        "trajectory_length_m": round(trajectory_length, 6),
    }
    write_json(args.output / "metadata.json", metadata)
    write_json(
        args.output / "temporal_labels.json",
        {"sequence_id": args.sequence_id, "segments": []},
    )
    write_json(
        args.output / "frame_labels.json",
        {"sequence_id": args.sequence_id, "frames": []},
    )
    print(
        f"Done: {frame_count} RGB frames, {imu_counts['imu-left']} left IMU, "
        f"{imu_counts['imu-right']} right IMU, {trajectory_count} trajectory poses"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument(
        "--mps-dir",
        type=Path,
        required=True,
        help="MPS directory containing slam/; copied into the final sequence",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--scene-name")
    parser.add_argument("--environment", choices=["indoor", "outdoor"])
    parser.add_argument("--collector-id", type=int)
    parser.add_argument("--video-downsample", type=int, default=1)
    parser.add_argument("--max-video-frames", type=int)
    parser.add_argument("--cpu-decode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.vrs.is_file():
        parser.error(f"VRS file does not exist: {args.vrs}")
    if not args.mps_dir.is_dir():
        parser.error(f"MPS directory does not exist: {args.mps_dir}")
    if args.video_downsample < 1:
        parser.error("--video-downsample must be at least 1")
    if args.max_video_frames is not None and args.max_video_frames < 1:
        parser.error("--max-video-frames must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        convert(parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
