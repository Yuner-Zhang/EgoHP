#!/usr/bin/env python3
"""Run the complete VRS-to-EgoHP conversion and labeling pipeline."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence


REQUIRED_SLAM_FILES = (
    "closed_loop_trajectory.csv",
    "open_loop_trajectory.csv",
    "online_calibration.jsonl",
    "semidense_observations.csv.gz",
    "semidense_points.csv.gz",
)
PILOT_DOWNLOAD_MANIFEST = (
    "https://explorer.projectaria.com/data/gen2pilot/download_links"
)
PILOT_MPS_ASSETS = (
    "mps_slam_trajectories",
    "mps_slam_calibration",
    "mps_slam_points",
)
PILOT_VRS_PATTERN = re.compile(
    r"^AriaGen2PilotDataset_v\d+(?:\.\d+)+_(.+)_main_recording\.vrs$"
)


def normalize_mps_dir(path: Path) -> Path:
    """Return the MPS directory whose direct child is slam/."""
    path = path.resolve()
    if (path / "slam").is_dir():
        return path
    if (path / "mps" / "slam").is_dir():
        return path / "mps"
    children = (
        [child for child in path.iterdir() if child.is_dir() and (child / "slam").is_dir()]
        if path.is_dir()
        else []
    )
    if len(children) == 1:
        return children[0]
    raise RuntimeError(f"Cannot find an MPS folder containing slam/: {path}")


def generated_mps_candidates(vrs_path: Path):
    # Match projectaria-mps AriaRecording.create(): only replace the .vrs
    # suffix, preserving dots that are part of the recording name (e.g. v1.0).
    generated_name = f"mps_{vrs_path.name.replace('.vrs', '_vrs')}"
    yield vrs_path.parent / generated_name
    yield vrs_path.parent / "mps"


def missing_mps_files(mps_dir: Path):
    slam = normalize_mps_dir(mps_dir) / "slam"
    return [name for name in REQUIRED_SLAM_FILES if not (slam / name).is_file()]


def validate_mps(mps_dir: Path) -> Path:
    mps_dir = normalize_mps_dir(mps_dir)
    missing = missing_mps_files(mps_dir)
    if missing:
        raise RuntimeError(
            "MPS SLAM result is incomplete; missing: " + ", ".join(missing)
        )
    return mps_dir


def discover_generated_mps(vrs_path: Path) -> Path:
    for candidate in generated_mps_candidates(vrs_path):
        if candidate.is_dir():
            try:
                return validate_mps(candidate)
            except RuntimeError:
                continue
    raise RuntimeError(f"MPS output was not found next to {vrs_path}")


def closed_loop_path(mps_dir: Path) -> Path:
    return validate_mps(mps_dir) / "slam" / "closed_loop_trajectory.csv"


def cached_mps_auth_available() -> bool:
    """Check for the legacy token or a usable Project Aria keyring entry."""
    if (Path.home() / ".projectaria" / "auth_token").is_file():
        return True
    try:
        import keyring

        return bool(keyring.get_password("projectaria_tools", "session_auth_token"))
    except Exception:
        return False


def official_pilot_sequence(vrs_path: Path) -> Optional[str]:
    match = PILOT_VRS_PATTERN.fullmatch(vrs_path.name)
    return match.group(1) if match else None


def sha1sum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.is_file() else 0
    if existing_size > expected_size:
        destination.unlink()
        existing_size = 0
    if existing_size == expected_size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "EgoHP/1.0"})
    if existing_size:
        request.add_header("Range", f"bytes={existing_size}-")
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Official Pilot Dataset download failed: {error}") from error
    status = getattr(response, "status", response.getcode())
    append = existing_size > 0 and status == 206
    if existing_size > 0 and not append:
        existing_size = 0
    mode = "ab" if append else "wb"
    downloaded = existing_size
    next_progress = downloaded + 256 * 1024 * 1024
    with response, destination.open(mode) as handle:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            downloaded += len(block)
            if downloaded >= next_progress:
                print(
                    f"  {destination.name}: {downloaded / (1024**3):.2f} / "
                    f"{expected_size / (1024**3):.2f} GiB",
                    flush=True,
                )
                next_progress += 256 * 1024 * 1024
    if destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete download {destination.name}: "
            f"{destination.stat().st_size} != {expected_size} bytes"
        )


def download_official_pilot_mps(
    sequence_id: str, cache: Path, download_root: Path
) -> Path:
    """Download Meta-published Pilot MPS archives without submitting the VRS."""
    try:
        with urllib.request.urlopen(PILOT_DOWNLOAD_MANIFEST, timeout=120) as response:
            manifest = json.load(response)
        assets = manifest["sequences"][sequence_id]
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot load official MPS manifest for Pilot sequence {sequence_id}"
        ) from error

    archives = []
    for asset_name in PILOT_MPS_ASSETS:
        try:
            asset = assets[asset_name]
            filename = str(asset["filename"])
            expected_size = int(asset["file_size_bytes"])
            expected_sha1 = str(asset["sha1sum"]).lower()
            url = str(asset["download_url"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid official MPS asset: {asset_name}") from error
        archive = download_root / sequence_id / filename
        print(f"Downloading official Pilot asset: {filename}", flush=True)
        download_with_resume(url, archive, expected_size)
        actual_sha1 = sha1sum(archive)
        if actual_sha1 != expected_sha1:
            raise RuntimeError(
                f"SHA-1 mismatch for {filename}: {actual_sha1} != {expected_sha1}"
            )
        archives.append(archive)

    slam = cache / "slam"
    slam.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                name = Path(member.filename).name
                if name not in REQUIRED_SLAM_FILES:
                    continue
                destination = slam / name
                print(f"Extracting official MPS: {name}", flush=True)
                with package.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    return validate_mps(cache)


def request_mps(vrs_path: Path, interactive_login: bool) -> Path:
    """Submit one VRS to Meta MPS and return the downloaded SLAM folder."""
    for candidate in generated_mps_candidates(vrs_path):
        if candidate.is_dir() and not missing_mps_files(candidate):
            print(f"Reusing complete MPS output next to VRS: {candidate}")
            return validate_mps(candidate)
    executable = shutil.which("aria_mps")
    if executable is None:
        raise RuntimeError("aria_mps is not installed in the active Ubuntu environment")
    if not interactive_login and not cached_mps_auth_available():
        raise RuntimeError(
            "Project Aria MPS is not authenticated. Run once with "
            "--mps-interactive-login (or run aria_mps without --no-ui), complete "
            "the Meta login, and keep that process open until results download."
        )
    print("Generating closed-loop SLAM with Meta MPS...", flush=True)
    command = [executable, "single", "-i", str(vrs_path)]
    if not interactive_login:
        command.append("--no-ui")
    command.extend(["--features", "SLAM"])
    subprocess.run(command, check=True)
    try:
        return discover_generated_mps(vrs_path)
    except RuntimeError as error:
        raise RuntimeError(
            "Meta MPS did not produce closed-loop SLAM. Run aria_mps once "
            "interactively to log in, then rerun this pipeline."
        ) from error


def cache_generated_mps(source: Path, cache: Path) -> Path:
    """Move downloaded MPS output away from raw/ and retain it as a cache."""
    source = normalize_mps_dir(source)
    if source.resolve() == cache.resolve():
        return validate_mps(source)
    if cache.exists():
        shutil.copytree(source, cache, dirs_exist_ok=True)
        return validate_mps(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    moved = Path(shutil.move(str(source), str(cache)))
    return validate_mps(moved)


def run_command(command) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def resolve_collector_id(vrs_path: Path, explicit_id: Optional[int]) -> int:
    """Read collector_id from raw/<collector_id>/<recording>.vrs."""
    folder_name = vrs_path.resolve().parent.name
    inferred_id: Optional[int] = None
    if folder_name.isdecimal():
        inferred_id = int(folder_name)
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


def process(args: argparse.Namespace) -> None:
    tools_dir = Path(__file__).resolve().parent
    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    collector_id = resolve_collector_id(args.vrs, args.collector_id)
    print(f"Collector ID: {collector_id} (from raw/{args.vrs.resolve().parent.name}/)")
    cache = dataset_root / ".egohp_cache" / args.sequence_id / "mps"

    if args.mps_dir is not None:
        mps_dir = validate_mps(args.mps_dir)
    elif cache.is_dir() and not missing_mps_files(cache):
        print(f"Reusing cached MPS SLAM: {cache}")
        mps_dir = validate_mps(cache)
    else:
        pilot_sequence = official_pilot_sequence(args.vrs)
        use_official = args.mps_source == "official" or (
            args.mps_source == "auto" and pilot_sequence is not None
        )
        if use_official:
            if pilot_sequence is None:
                raise RuntimeError(
                    "--mps-source official requires an official "
                    "AriaGen2PilotDataset VRS filename"
                )
            print(
                f"Official Pilot sequence detected: {pilot_sequence}; "
                "downloading published MPS instead of uploading the VRS",
                flush=True,
            )
            download_root = dataset_root / ".egohp_downloads" / "gen2pilot"
            mps_dir = download_official_pilot_mps(
                pilot_sequence, cache, download_root
            )
        else:
            generated = request_mps(args.vrs.resolve(), args.mps_interactive_login)
            mps_dir = cache_generated_mps(generated, cache)
    trajectory = closed_loop_path(mps_dir)
    print(f"Closed-loop trajectory: {trajectory}")

    staging = dataset_root / ".egohp_staging" / args.sequence_id
    if staging.exists() and any(staging.iterdir()) and not args.overwrite_staging:
        raise RuntimeError(
            f"Staging sequence is not empty: {staging} (use --overwrite-staging)"
        )

    convert_command = [
        sys.executable,
        tools_dir / "convert_to_egohp.py",
        "--vrs",
        args.vrs.resolve(),
        "--mps-dir",
        mps_dir,
        "--output",
        staging,
        "--sequence-id",
        args.sequence_id,
        "--video-downsample",
        args.video_downsample,
    ]
    convert_command.extend(["--collector-id", collector_id])
    if args.max_video_frames is not None:
        convert_command.extend(["--max-video-frames", args.max_video_frames])
    if args.cpu_decode:
        convert_command.append("--cpu-decode")
    if args.overwrite_staging:
        convert_command.append("--overwrite")
    run_command(convert_command)

    frame_command = [
        sys.executable,
        tools_dir / "generate_frame_labels.py",
        "--sequence",
        staging,
        "--person-model",
        args.person_model,
        "--screen-model",
        args.screen_model,
        "--tracker",
        args.tracker,
        "--min-track-hits",
        args.min_track_hits,
        "--person-confidence",
        args.person_confidence,
        "--detector-image-size",
        args.detector_image_size,
        "--occlusion-keypoint-confidence",
        args.occlusion_keypoint_confidence,
        "--occlusion-none-visible-ratio",
        args.occlusion_none_visible_ratio,
        "--occlusion-severe-visible-ratio",
        args.occlusion_severe_visible_ratio,
        "--occlusion-person-overlap",
        args.occlusion_person_overlap,
        "--occlusion-boundary-fraction",
        args.occlusion_boundary_fraction,
        "--vis-downsample",
        args.vis_downsample,
    ]
    if args.detector_device is not None:
        frame_command.extend(["--detector-device", args.detector_device])
    if args.expected_person_count is not None:
        frame_command.extend(["--expected-person-count", args.expected_person_count])
    if not args.visualize:
        frame_command.append("--no-visualize")
    if args.cpu_decode:
        frame_command.append("--cpu-decode")
    run_command(frame_command)

    temporal_command = [
        sys.executable,
        tools_dir / "generate_temporal_labels.py",
        "--sequence",
        staging,
    ]
    if args.cpu_decode:
        temporal_command.append("--cpu-decode")
    run_command(temporal_command)

    metadata_command = [
        sys.executable,
        tools_dir / "generate_metadata.py",
        "--sequence",
        staging,
        "--source-vrs",
        args.vrs.resolve(),
        "--dataset-root",
        dataset_root,
    ]
    if not args.keep_mps:
        metadata_command.append("--no-keep-mps")
    if args.cpu_decode:
        metadata_command.append("--cpu-decode")
    run_command(metadata_command)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True, help="for example seq_001")
    parser.add_argument(
        "--collector-id",
        type=int,
        help="normally inferred from raw/<collector_id>/; optional explicit check",
    )
    parser.add_argument(
        "--mps-dir",
        type=Path,
        help="reuse an existing MPS output and bypass official/cloud acquisition",
    )
    parser.add_argument(
        "--mps-source",
        choices=("auto", "official", "cloud"),
        default="auto",
        help=(
            "auto downloads published MPS for an official Pilot VRS and uses "
            "Meta MPS cloud for other recordings"
        ),
    )
    parser.add_argument(
        "--mps-interactive-login",
        action="store_true",
        help="first MPS run: show the official UI/login flow instead of using --no-ui",
    )
    parser.add_argument("--video-downsample", type=int, default=1)
    parser.add_argument("--max-video-frames", type=int)
    parser.add_argument("--person-model", default="yolo11s-pose.pt")
    parser.add_argument("--screen-model", default="yolo11s.pt")
    parser.add_argument(
        "--tracker",
        default=str(Path(__file__).resolve().with_name("botsort_reid.yaml")),
    )
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument(
        "--expected-person-count",
        type=int,
        help="merge fragmented tracks into this many appearance identities",
    )
    parser.add_argument("--person-confidence", type=float, default=0.35)
    parser.add_argument("--detector-image-size", type=int, default=1280)
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--occlusion-keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--occlusion-none-visible-ratio", type=float, default=0.80)
    parser.add_argument("--occlusion-severe-visible-ratio", type=float, default=0.30)
    parser.add_argument("--occlusion-person-overlap", type=float, default=0.15)
    parser.add_argument("--occlusion-boundary-fraction", type=float, default=0.02)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write frame_labels_vis.mp4; use --no-visualize to skip it",
    )
    parser.add_argument(
        "--keep-mps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain seq_xxx/mps after labeling; --no-keep-mps deletes it in Step 5",
    )
    parser.add_argument("--vis-downsample", type=int, default=2)
    parser.add_argument("--cpu-decode", action="store_true")
    parser.add_argument("--overwrite-staging", action="store_true")
    args = parser.parse_args(argv)
    if not args.vrs.is_file():
        parser.error(f"VRS file does not exist: {args.vrs}")
    if args.mps_dir is not None and not args.mps_dir.is_dir():
        parser.error(f"MPS directory does not exist: {args.mps_dir}")
    if args.video_downsample < 1 or args.vis_downsample < 1:
        parser.error("downsample values must be positive")
    if args.max_video_frames is not None and args.max_video_frames < 1:
        parser.error("--max-video-frames must be positive")
    if args.min_track_hits < 1:
        parser.error("--min-track-hits must be positive")
    if args.expected_person_count is not None and args.expected_person_count < 1:
        parser.error("--expected-person-count must be positive")
    occlusion_parameters = (
        args.occlusion_keypoint_confidence,
        args.occlusion_none_visible_ratio,
        args.occlusion_severe_visible_ratio,
        args.occlusion_person_overlap,
        args.occlusion_boundary_fraction,
    )
    if any(not 0.0 <= value <= 1.0 for value in occlusion_parameters):
        parser.error("occlusion thresholds must be between 0 and 1")
    if args.occlusion_severe_visible_ratio >= args.occlusion_none_visible_ratio:
        parser.error(
            "--occlusion-severe-visible-ratio must be smaller than "
            "--occlusion-none-visible-ratio"
        )
    if not args.sequence_id.startswith("seq_"):
        parser.error("--sequence-id must use the seq_xxx naming convention")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        process(parse_args(argv))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
