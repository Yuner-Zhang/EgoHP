#!/usr/bin/env python3
"""Generate metadata with a vision API and route the completed sequence."""

import argparse
import base64
import gc
import io
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_API_MODEL = "gpt-4.1-mini"
METADATA_FIELDS = {
    "scene_name": "short lower_snake_case location or scene name",
    "environment": "indoor or outdoor",
    "time_of_day": "day, dawn_dusk, or night",
    "weather": "clear, cloudy, rain, fog, snow, or null when not reliably visible",
    "crowd_density": "empty, low, medium, high, or very_high",
}
METADATA_ALLOWED = {
    "environment": {"indoor", "outdoor"},
    "time_of_day": {"day", "dawn_dusk", "night"},
    "weather": {"clear", "cloudy", "rain", "fog", "snow"},
    "crowd_density": {"empty", "low", "medium", "high", "very_high"},
}
PERSON_OCCLUSION_LABELS = {
    "none",
    "inter_person_partial",
    "inter_person_severe",
    "object_partial",
    "object_severe",
    "boundary_partial",
    "boundary_severe",
    "mixed_partial",
    "mixed_severe",
}
METADATA_RULES = """
Labeling rules:
- sequence_id and collector_id are supplied by the pipeline; never infer them.
- scene_name: concise lower_snake_case semantic location, such as living_room.
- environment: indoor if enclosed by a building; otherwise outdoor.
- time_of_day: day (daylight), dawn_dusk (twilight/low sun), or night.
- weather: use clear, cloudy, rain, fog, or snow only when visual evidence is reliable.
  Return null when weather cannot be judged from the sampled frames. This is common indoors,
  but do not force null when outdoor conditions are clearly visible through a window.
- crowd_density uses the representative visible-person count across sampled frames:
  empty=0-2, low=3-5, medium=6-20, high=21-50, very_high=>50.
""".strip()


def load_modules(cpu_decode: bool):
    if cpu_decode:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    try:
        from PIL import Image
        from projectaria_tools.core import data_provider
    except ImportError as error:
        raise RuntimeError("Install environment.yml in Ubuntu 20.04 first") from error
    return data_provider, Image


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_json_atomic(path: Path, value: object) -> None:
    """Replace a shared JSON index only after the new document is complete."""
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def scene_slug(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_json_text(text: str) -> Dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise RuntimeError("API response must be a JSON object")
    return value


def image_data_url(image_array, image_module, quality: int) -> str:
    buffer = io.BytesIO()
    image_module.fromarray(image_array).save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def evenly_spaced_indices(count: int, wanted: int) -> List[int]:
    if count <= 0:
        return []
    if wanted <= 1:
        return [count // 2]
    return sorted({round(index * (count - 1) / (wanted - 1)) for index in range(wanted)})


def summarize_scene_occlusion(frame_labels: object) -> Dict[str, object]:
    """Aggregate Step 3 person-frame labels into one sequence-level quality label."""
    if not isinstance(frame_labels, dict) or not isinstance(
        frame_labels.get("frames"), list
    ):
        raise RuntimeError("frame_labels.json must contain a frames list")
    total = occluded = severe = 0
    for frame in frame_labels["frames"]:
        if not isinstance(frame, dict):
            continue
        for person in frame.get("persons", []):
            if not isinstance(person, dict):
                continue
            total += 1
            label = person.get("occlusion")
            if label not in PERSON_OCCLUSION_LABELS:
                raise RuntimeError(
                    "Every person bbox must have a valid occlusion label before Step 5; "
                    f"got {label!r}"
                )
            if label != "none":
                occluded += 1
            if str(label).endswith("_severe"):
                severe += 1
    if total == 0:
        return {
            "level": None,
            "person_frame_count": 0,
            "occluded_ratio": None,
            "severe_ratio": None,
        }
    occluded_ratio = occluded / total
    severe_ratio = severe / total
    if occluded_ratio < 0.20 and severe_ratio < 0.05:
        level = "low"
    elif occluded_ratio < 0.50 and severe_ratio < 0.20:
        level = "medium"
    else:
        level = "high"
    return {
        "level": level,
        "person_frame_count": total,
        "occluded_ratio": round(occluded_ratio, 6),
        "severe_ratio": round(severe_ratio, 6),
    }


def call_json_api(args, prompt: str, image_urls: List[str]) -> Dict[str, object]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")
    content: List[Dict[str, object]] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    if not args.no_response_format:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Metadata API request failed: {error}") from error
    try:
        message = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Unexpected API response structure") from error
    return parse_json_text(message)


def annotate_metadata(args, provider, stream_id, image_module, frame_count: int) -> dict:
    path = args.sequence / "metadata.json"
    metadata = load_json(path)
    # Drop the legacy none/partial/severe field. The replacement below is a
    # low/medium/high aggregate derived from Step 3 person-frame labels.
    metadata.pop("occlusion_level", None)
    occlusion_summary = summarize_scene_occlusion(
        load_json(args.sequence / "frame_labels.json")
    )
    metadata["occlusion_level"] = occlusion_summary["level"]
    images = []
    for index in evenly_spaced_indices(frame_count, args.metadata_frames):
        image, _ = provider.get_image_data_by_index(stream_id, index)
        images.append(image_data_url(image.to_numpy_array(), image_module, args.jpeg_quality))
    prompt = (
        "Infer sequence-level labels from these chronological egocentric frames. "
        "Return only the fields in the schema.\n\n"
        + METADATA_RULES
        + "\n\nJSON schema:\n"
        + json.dumps(METADATA_FIELDS, indent=2)
    )
    result = call_json_api(args, prompt, images)
    for field in METADATA_FIELDS:
        if field not in result:
            continue
        value = result[field]
        if field == "scene_name":
            normalized = scene_slug(value)
            if normalized:
                metadata[field] = normalized
        elif field == "weather" and value is None:
            metadata[field] = None
        elif value in METADATA_ALLOWED[field]:
            metadata[field] = value
    write_json(path, metadata)
    print(
        "  scene occlusion: "
        f"{occlusion_summary['level']} from "
        f"{occlusion_summary['person_frame_count']} person-frames; "
        f"occluded={occlusion_summary['occluded_ratio']}, "
        f"severe={occlusion_summary['severe_ratio']}"
    )
    return metadata


def write_recording_index(
    source_vrs: Path, dataset_root: Path, sequence: Path, metadata: dict
) -> None:
    source_folder = source_vrs.resolve().parent.name
    if source_folder.isdecimal():
        source_collector_id = int(source_folder)
        if metadata.get("collector_id") != source_collector_id:
            raise RuntimeError(
                "collector_id does not match the numeric raw folder: "
                f"metadata={metadata.get('collector_id')}, raw/{source_collector_id}/"
            )
    source = {
        "source_vrs": Path(os.path.relpath(source_vrs, dataset_root)).as_posix(),
        "source_vrs_filename": source_vrs.name,
        "sequence_id": metadata["sequence_id"],
        "collector_id": metadata.get("collector_id"),
        "converted_sequence": Path(os.path.relpath(sequence, dataset_root)).as_posix(),
        "converted_vrs": Path(
            os.path.relpath(
                sequence / f"{metadata['sequence_id']}.vrs", dataset_root
            )
        ).as_posix(),
        "converted_mp4": Path(
            os.path.relpath(
                sequence / f"{metadata['sequence_id']}.mp4", dataset_root
            )
        ).as_posix(),
        "environment": metadata["environment"],
        "scene_name": metadata["scene_name"],
    }
    index_path = dataset_root / "recording_index.json"
    index = load_json(index_path) if index_path.is_file() else {"schema_version": 1, "recordings": []}
    recordings = [
        item
        for item in index.get("recordings", [])
        if item.get("source_vrs") != source["source_vrs"]
        and item.get("sequence_id") != source["sequence_id"]
    ]
    recordings.append(source)
    recordings.sort(key=lambda item: item["sequence_id"])
    write_json_atomic(index_path, {"schema_version": 1, "recordings": recordings})


def organize_sequence(sequence: Path, dataset_root: Path, source_vrs: Optional[Path]) -> Path:
    metadata = load_json(sequence / "metadata.json")
    environment = metadata.get("environment")
    scene_name = scene_slug(metadata.get("scene_name"))
    sequence_id = scene_slug(metadata.get("sequence_id"))
    if environment not in {"indoor", "outdoor"}:
        raise RuntimeError("metadata.environment must be indoor or outdoor")
    if not scene_name or not sequence_id:
        raise RuntimeError("metadata scene_name and sequence_id must be non-empty")
    metadata["scene_name"] = scene_name
    write_json(sequence / "metadata.json", metadata)
    destination = (
        dataset_root.resolve()
        / ("Indoor" if environment == "indoor" else "Outdoor")
        / scene_name
        / sequence_id
    )
    if sequence.resolve() == destination:
        moved = destination
    else:
        if destination.exists():
            raise RuntimeError(f"Final sequence already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        moved = Path(shutil.move(str(sequence), str(destination)))
    if source_vrs is not None:
        write_recording_index(source_vrs, dataset_root, moved, metadata)
    return moved


def remove_sequence_mps(sequence: Path) -> bool:
    """Remove only the exact mps/ child of one resolved sequence directory."""
    sequence_root = sequence.resolve()
    target = (sequence_root / "mps").resolve()
    if target.parent != sequence_root or target.name != "mps":
        raise RuntimeError(f"Refusing unsafe MPS removal target: {target}")
    if not target.exists():
        return False
    if not target.is_dir():
        raise RuntimeError(f"Expected an MPS directory, found: {target}")
    shutil.rmtree(target)
    return True


def run(args) -> None:
    recording_vrs = args.sequence / f"{args.sequence.name}.vrs"
    required = [
        recording_vrs.name,
        f"{args.sequence.name}.mp4",
        "metadata.json",
        "frame_labels.json",
        "temporal_labels.json",
    ]
    missing = [name for name in required if not (args.sequence / name).is_file()]
    if missing:
        raise RuntimeError("Sequence is missing: " + ", ".join(missing))
    data_provider, image_module = load_modules(args.cpu_decode)
    provider = data_provider.create_vrs_data_provider(str(recording_vrs.resolve()))
    if provider is None:
        raise RuntimeError(f"Cannot open sequence recording: {recording_vrs.name}")
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    metadata = load_json(args.sequence / "metadata.json")
    frame_count = min(provider.get_num_data(stream_id), int(metadata.get("num_frames", 0)))
    if frame_count <= 0:
        raise RuntimeError("No RGB frames available for metadata labeling")
    if args.dry_run:
        print(f"Metadata dry run: {frame_count} frames; no API request or move")
        print(f"  keep MPS in final sequence: {args.keep_mps}")
        print(METADATA_RULES)
        return
    annotate_metadata(args, provider, stream_id, image_module, frame_count)
    provider = None
    gc.collect()
    final_sequence = args.sequence
    if args.dataset_root is not None:
        final_sequence = organize_sequence(args.sequence, args.dataset_root, args.source_vrs)
    if not args.keep_mps:
        removed = remove_sequence_mps(final_sequence)
        if removed:
            print(f"Removed generated MPS directory: {final_sequence / 'mps'}")
    print(f"Metadata completed: {final_sequence}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--source-vrs", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("EGOHP_API_ENDPOINT", DEFAULT_API_ENDPOINT))
    parser.add_argument("--model", default=os.environ.get("EGOHP_API_MODEL", DEFAULT_API_MODEL))
    parser.add_argument("--api-key-env", default="EGOHP_API_KEY")
    parser.add_argument("--metadata-frames", type=int, default=8)
    parser.add_argument(
        "--keep-mps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain seq_xxx/mps; --no-keep-mps removes it after final routing",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--cpu-decode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.sequence.is_dir():
        parser.error(f"Sequence does not exist: {args.sequence}")
    if args.source_vrs is not None and not args.source_vrs.is_file():
        parser.error(f"Source VRS does not exist: {args.source_vrs}")
    if args.metadata_frames < 1:
        parser.error("--metadata-frames must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
