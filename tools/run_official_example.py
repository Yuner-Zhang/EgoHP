#!/usr/bin/env python3
"""Download one official Gen2 Pilot sequence and run the EgoHP pipeline."""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from prepare_data import (
    PILOT_DOWNLOAD_MANIFEST,
    PILOT_MPS_ASSETS,
    download_official_pilot_mps,
    download_with_resume,
    sha1sum,
)


YOLO_MODELS = ("yolo11s-pose.pt", "yolo11s.pt")


def load_manifest(url: str) -> dict:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "EgoHP/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            manifest = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load official Pilot manifest: {url}") from error
    if not isinstance(manifest.get("sequences"), dict):
        raise RuntimeError("Official Pilot manifest has no sequences object")
    return manifest


def sequence_assets(manifest: dict, sequence: str) -> dict:
    sequences = manifest["sequences"]
    if sequence not in sequences:
        available = ", ".join(sorted(sequences))
        raise RuntimeError(
            f"Official Pilot sequence does not exist: {sequence}. "
            f"Available sequences: {available}"
        )
    assets = sequences[sequence]
    required = ("main_vrs", *PILOT_MPS_ASSETS)
    missing = [name for name in required if name not in assets]
    if missing:
        raise RuntimeError(
            f"Official manifest entry {sequence} is missing: {', '.join(missing)}"
        )
    return assets


def asset_fields(asset: dict, asset_name: str):
    try:
        return (
            str(asset["filename"]),
            int(asset["file_size_bytes"]),
            str(asset["sha1sum"]).lower(),
            str(asset["download_url"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid official manifest asset: {asset_name}") from error


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def download_vrs(asset: dict, raw_dir: Path) -> Path:
    filename, expected_size, expected_sha1, url = asset_fields(asset, "main_vrs")
    destination = raw_dir / filename
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"Checking existing VRS: {destination}", flush=True)
        if sha1sum(destination) == expected_sha1:
            print("VRS is already complete and verified.", flush=True)
            return destination
        print("Existing VRS checksum is invalid; downloading it again.", flush=True)
        destination.unlink()

    print(f"Downloading official VRS: {filename} ({human_size(expected_size)})")
    download_with_resume(url, destination, expected_size)
    actual_sha1 = sha1sum(destination)
    if actual_sha1 != expected_sha1:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-1 mismatch for {filename}: {actual_sha1} != {expected_sha1}"
        )
    print(f"Verified official VRS: {destination}", flush=True)
    return destination


def ensure_yolo_models(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in YOLO_MODELS if not (model_dir / name).is_file()]
    if missing:
        print("Downloading Ultralytics YOLO11 weights: " + ", ".join(missing))
        code = (
            "from ultralytics import YOLO; "
            + "; ".join(f"YOLO({name!r})" for name in missing)
        )
        subprocess.run([sys.executable, "-c", code], cwd=model_dir, check=True)
    paths = tuple((model_dir / name).resolve() for name in YOLO_MODELS)
    missing_after = [str(path) for path in paths if not path.is_file()]
    if missing_after:
        raise RuntimeError("YOLO model download failed: " + ", ".join(missing_after))
    return paths  # type: ignore[return-value]


def ensure_dataset_layout(dataset_root: Path, collector_id: int) -> None:
    """Create dataset directories and a manually editable collector template."""
    converted_root = dataset_root / "converted"
    for directory in (
        dataset_root / "raw" / str(collector_id),
        dataset_root / "models",
        converted_root / "Indoor",
        converted_root / "Outdoor",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    collectors_path = converted_root / "collectors.json"
    if collectors_path.is_file():
        try:
            document = json.loads(collectors_path.read_text(encoding="utf-8"))
            collectors = document["collectors"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"Invalid collectors file: {collectors_path}") from error
        if not isinstance(collectors, list):
            raise RuntimeError(f"collectors must be a list: {collectors_path}")
    else:
        document = {"collectors": []}
        collectors = document["collectors"]

    collector_exists = any(
        item.get("collector_id") == collector_id
        for item in collectors
        if isinstance(item, dict)
    )
    if collector_exists:
        return
    collectors.append(
        {"collector_id": collector_id, "gender": None, "height_cm": None}
    )
    temporary = collectors_path.with_name(collectors_path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, collectors_path)
    print(f"Created collector template: {collectors_path}")


def print_download_plan(
    sequence: str,
    assets: dict,
    raw_dir: Path,
    mps_dir: Path,
    model_dir: Path,
) -> None:
    print(f"Official sequence: {sequence}")
    print(f"Raw VRS directory: {raw_dir}")
    print(f"Extracted MPS directory: {mps_dir}")
    print(f"YOLO model directory: {model_dir}")
    print("Downloads:")
    for name in ("main_vrs", *PILOT_MPS_ASSETS):
        filename, size, _, _ = asset_fields(assets[name], name)
        print(f"  {name}: {filename} ({human_size(size)})")


def run_pipeline(
    args: argparse.Namespace,
    vrs_path: Path,
    mps_dir: Path,
    person_model: Path,
    screen_model: Path,
) -> None:
    tools_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        tools_dir / "prepare_data.py",
        "--vrs",
        vrs_path,
        "--dataset-root",
        args.dataset_root.resolve() / "converted",
        "--sequence-id",
        args.sequence_id,
        "--mps-dir",
        mps_dir,
        "--person-model",
        person_model,
        "--screen-model",
        screen_model,
        "--detector-device",
        args.detector_device,
        "--min-track-hits",
        args.min_track_hits,
    ]
    if args.sample:
        command.extend(["--max-video-frames", args.max_video_frames])
    if args.cpu_decode:
        command.append("--cpu-decode")
    if not args.visualize:
        command.append("--no-visualize")
    if not args.keep_mps:
        command.append("--no-keep-mps")
    if args.overwrite_staging:
        command.append("--overwrite-staging")

    print("Running EgoHP pipeline:", flush=True)
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        default="play_0",
        help="official Gen2 Pilot sequence name, for example play_0",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data"),
        help="root containing raw/, models/, and converted/",
    )
    parser.add_argument("--collector-id", type=int, default=0)
    parser.add_argument(
        "--sequence-id",
        help="EgoHP output ID; default is seq_<name> or seq_<name>_sample",
    )
    parser.add_argument(
        "--manifest-url",
        default=PILOT_DOWNLOAD_MANIFEST,
        help="Project Aria official Gen2 Pilot download manifest",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download and verify VRS/MPS, but do not run EgoHP",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show official assets and local paths without downloading",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="process only the first sample frames instead of the full recording",
    )
    parser.add_argument("--max-video-frames", type=int, default=200)
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--min-track-hits", type=int, default=20)
    parser.add_argument(
        "--cpu-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-mps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite-staging", action="store_true")
    args = parser.parse_args(argv)
    if args.collector_id < 0:
        parser.error("--collector-id must be non-negative")
    if args.max_video_frames < 1:
        parser.error("--max-video-frames must be positive")
    if args.min_track_hits < 1:
        parser.error("--min-track-hits must be positive")
    if args.sequence_id is None:
        suffix = "_sample" if args.sample else ""
        args.sequence_id = f"seq_{args.sequence}{suffix}"
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest_url)
    assets = sequence_assets(manifest, args.sequence)
    dataset_root = args.dataset_root.resolve()
    raw_dir = dataset_root / "raw" / str(args.collector_id)
    archive_root = dataset_root / "converted" / ".egohp_downloads" / "gen2pilot"
    mps_dir = (
        dataset_root
        / "converted"
        / ".egohp_cache"
        / "gen2pilot"
        / args.sequence
        / "mps"
    )
    model_dir = dataset_root / "models"
    print_download_plan(args.sequence, assets, raw_dir, mps_dir, model_dir)
    mode = f"first {args.max_video_frames} frames" if args.sample else "full recording"
    print(f"EgoHP output sequence: {args.sequence_id} ({mode})")
    print(f"Retain MPS in final sequence: {args.keep_mps}")
    if args.dry_run:
        return

    ensure_dataset_layout(dataset_root, args.collector_id)
    vrs_path = download_vrs(assets["main_vrs"], raw_dir)
    mps_dir = download_official_pilot_mps(args.sequence, mps_dir, archive_root)
    print(f"Official MPS is ready: {mps_dir}", flush=True)
    if args.download_only:
        print("Download-only mode completed.")
        return

    if not os.environ.get("EGOHP_API_KEY"):
        raise RuntimeError(
            "EGOHP_API_KEY is not configured. Save it in the egohp_gen2 conda "
            "environment, reactivate that environment, and rerun. All downloaded "
            "VRS/MPS files will be reused."
        )
    person_model, screen_model = ensure_yolo_models(model_dir)
    run_pipeline(args, vrs_path, mps_dir, person_model, screen_model)


if __name__ == "__main__":
    main()
