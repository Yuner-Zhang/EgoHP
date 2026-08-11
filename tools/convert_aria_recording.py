#!/usr/bin/env python3
"""Convert a Project Aria VRS + MPS trajectory to the EgoHP sequence format."""

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


IMU_HEADER = ["timestamp_ns", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]
TRAJECTORY_HEADER = ["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]


def _load_aria_modules():
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
    except ImportError as error:
        raise RuntimeError(
            "projectaria-tools is required. Install requirements-conversion.txt "
            f"on Linux or macOS. Import error: {error}"
        ) from error
    return data_provider, TimeDomain


def _required_stream(provider, label: str):
    stream_id = provider.get_stream_id_from_label(label)
    if stream_id is None or not provider.check_stream_is_active(stream_id):
        raise RuntimeError(f"Required VRS stream is missing: {label}")
    return stream_id


def _vrs_timing(provider, time_domain) -> Tuple[int, int, object, Dict[str, object]]:
    rgb_stream = _required_stream(provider, "camera-rgb")
    imu_streams = {
        "imu-left": _required_stream(provider, "imu-left"),
        "imu-right": _required_stream(provider, "imu-right"),
    }
    streams = [rgb_stream] + list(imu_streams.values())
    start_ns = min(provider.get_first_time_ns(stream, time_domain) for stream in streams)
    end_ns = max(provider.get_last_time_ns(stream, time_domain) for stream in streams)
    return start_ns, end_ns, rgb_stream, imu_streams


def export_imus(
    provider,
    imu_streams: Dict[str, object],
    output_dir: Path,
) -> Dict[str, int]:
    output_paths = {
        "imu-left": output_dir / "imu_left.txt",
        "imu-right": output_dir / "imu_right.txt",
    }
    handles = {
        label: path.open("w", newline="", encoding="utf-8")
        for label, path in output_paths.items()
    }
    try:
        writers = {label: csv.writer(handles[label]) for label in handles}
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
            gyro = sample.gyro_radsec
            accel = sample.accel_msec2
            writers[label].writerow(
                [
                    int(sample.capture_timestamp_ns),
                    *(f"{value:.9f}" for value in gyro),
                    *(f"{value:.9f}" for value in accel),
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
    rotate_gen1: bool,
    output_path: Path,
    downsample: int,
) -> Tuple[int, float]:
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "imageio, imageio-ffmpeg, numpy, and Pillow are required for MP4 export"
        ) from error

    configuration = provider.get_image_configuration(rgb_stream)
    fps = float(configuration.nominal_rate_hz)
    options = provider.get_default_deliver_queued_options()
    options.deactivate_stream_all()
    options.activate_stream(rgb_stream)

    frame_count = 0
    with imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", macro_block_size=None
    ) as writer:
        for sensor_data in provider.deliver_queued_sensor_data(options):
            image, _ = sensor_data.image_data_and_record()
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
            frame_count += 1
    return frame_count, fps


def _timestamp_us(row: Dict[str, str]) -> int:
    if "tracking_timestamp_us" in row:
        return int(row["tracking_timestamp_us"])
    if "tracking_timestamp_ns" in row:
        return int(row["tracking_timestamp_ns"]) // 1000
    raise RuntimeError("MPS trajectory lacks tracking_timestamp_us")


def export_trajectory(
    source_csv: Path,
    output_txt: Path,
    recording_start_ns: int,
    recording_end_ns: Optional[int] = None,
) -> Tuple[int, float]:
    required = [
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "qx_world_device",
        "qy_world_device",
        "qz_world_device",
        "qw_world_device",
    ]
    rows_written = 0
    trajectory_length = 0.0
    previous_position: Optional[Tuple[float, float, float]] = None

    with source_csv.open("r", newline="", encoding="utf-8-sig") as source, output_txt.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty trajectory CSV: {source_csv}")
        missing = [field for field in required if field not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"MPS trajectory is missing fields: {', '.join(missing)}")

        writer = csv.writer(output)
        writer.writerow(TRAJECTORY_HEADER)
        for row in reader:
            timestamp_ns = _timestamp_us(row) * 1000
            if timestamp_ns < recording_start_ns:
                continue
            if recording_end_ns is not None and timestamp_ns > recording_end_ns:
                continue
            position = tuple(float(row[field]) for field in required[:3])
            quaternion = tuple(float(row[field]) for field in required[3:])
            if previous_position is not None:
                trajectory_length += math.dist(previous_position, position)
            previous_position = position
            writer.writerow(
                [
                    timestamp_ns,
                    *(f"{value:.9f}" for value in position),
                    *(f"{value:.9f}" for value in quaternion),
                ]
            )
            rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No MPS trajectory samples overlap the VRS recording")
    return rows_written, trajectory_length


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def convert(args: argparse.Namespace) -> None:
    data_provider, time_domain = _load_aria_modules()
    provider = data_provider.create_vrs_data_provider(str(args.vrs.resolve()))
    if provider is None:
        raise RuntimeError(f"Could not open VRS file: {args.vrs}")

    start_ns, end_ns, rgb_stream, imu_streams = _vrs_timing(
        provider, time_domain.DEVICE_TIME
    )
    args.output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in args.output.iterdir()]
    if existing and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output} (use --overwrite)")

    if not args.skip_vrs_copy:
        shutil.copy2(args.vrs, args.output / "video.vrs")

    print("Exporting RGB video...")
    device_calibration = provider.get_device_calibration()
    get_device_version = getattr(device_calibration, "get_device_version", None)
    # Project Aria Tools 1.5.x predates the DeviceVersion API and only supports
    # the Gen 1 sample used here. Newer releases expose an enum whose string
    # representation contains "Gen1" or "Gen2".
    rotate_gen1 = (
        True
        if get_device_version is None
        else "gen1" in str(get_device_version()).lower()
    )
    video_frame_count, frame_rate = export_video(
        provider,
        rgb_stream,
        rotate_gen1,
        args.output / "video.mp4",
        args.video_downsample,
    )

    print("Exporting IMU streams...")
    imu_counts = export_imus(provider, imu_streams, args.output)

    print("Converting MPS trajectory...")
    trajectory_count, trajectory_length = export_trajectory(
        args.trajectory_csv,
        args.output / "trajectory.txt",
        start_ns,
        end_ns,
    )

    if video_frame_count == 0:
        raise RuntimeError("The VRS recording contains no RGB frames")
    duration_sec = video_frame_count / frame_rate
    metadata_frame_rate = (
        int(round(frame_rate))
        if math.isclose(frame_rate, round(frame_rate))
        else round(frame_rate, 6)
    )
    metadata = {
        "sequence_id": args.sequence_id,
        "scene_name": args.scene_name,
        "time_of_day": args.time_of_day,
        "weather": args.weather,
        "crowd_density": args.crowd_density,
        "occlusion_level": args.occlusion_level,
        "collector_id": args.collector_id,
        "num_frames": video_frame_count,
        "frame_rate": metadata_frame_rate,
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
        "Done: "
        f"{video_frame_count} RGB frames, "
        f"{imu_counts['imu-left']} left IMU samples, "
        f"{imu_counts['imu-right']} right IMU samples, "
        f"{trajectory_count} trajectory poses."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--scene-name", "--scene", dest="scene_name", default=None)
    parser.add_argument("--collector-id", type=int, default=None)
    parser.add_argument(
        "--time-of-day", choices=["day", "dawn_dusk", "night"], default=None
    )
    parser.add_argument(
        "--weather", choices=["clear", "cloudy", "rain", "fog", "snow"], default=None
    )
    parser.add_argument(
        "--crowd-density",
        choices=["empty", "low", "medium", "high", "very_high"],
        default=None,
    )
    parser.add_argument(
        "--occlusion-level", choices=["none", "partial", "severe"], default=None
    )
    parser.add_argument("--video-downsample", type=int, default=1)
    parser.add_argument("--skip-vrs-copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.vrs.is_file():
        parser.error(f"VRS file does not exist: {args.vrs}")
    if not args.trajectory_csv.is_file():
        parser.error(f"trajectory CSV does not exist: {args.trajectory_csv}")
    if args.video_downsample < 1:
        parser.error("--video-downsample must be at least 1")
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
