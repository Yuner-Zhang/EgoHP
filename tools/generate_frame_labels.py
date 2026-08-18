#!/usr/bin/env python3
"""Generate person boxes, track IDs, occlusion labels, and a review MP4."""

import argparse
import colorsys
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


FRAME_PERSON_FIELDS = {
    "person_id": "persistent non-negative track ID",
    "bbox": "[x_min, y_min, x_max, y_max] in native RGB pixels",
    "occlusion": (
        "none or <inter_person|object|boundary|mixed>_<partial|severe>"
    ),
}

OCCLUSION_LABELS = {
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


def load_person_detector(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Person bbox generation requires ultralytics; install environment.yml"
        ) from error
    return YOLO(model_path)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")




def get_raw_frame(provider, stream_id, index: int):
    image, record = provider.get_image_data_by_index(stream_id, index)
    array = image.to_numpy_array()
    return array, int(record.capture_timestamp_ns)


def box_inside_screen(
    person_box: Sequence[float],
    screen_boxes: Sequence[Sequence[float]],
    frame_width: int,
    frame_height: int,
    padding_fraction: float,
    overlap_threshold: float,
) -> bool:
    """Return true when a person detection is visually contained in a screen."""
    px1, py1, px2, py2 = map(float, person_box)
    person_area = max(1.0, px2 - px1) * max(1.0, py2 - py1)
    center_x = (px1 + px2) / 2.0
    center_y = (py1 + py2) / 2.0
    for raw_screen in screen_boxes:
        sx1, sy1, sx2, sy2 = map(float, raw_screen)
        pad_x = (sx2 - sx1) * padding_fraction
        pad_y = (sy2 - sy1) * padding_fraction
        sx1 = max(0.0, sx1 - pad_x)
        sy1 = max(0.0, sy1 - pad_y)
        sx2 = min(float(frame_width), sx2 + pad_x)
        sy2 = min(float(frame_height), sy2 + pad_y)
        center_inside = sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2
        intersection_width = max(0.0, min(px2, sx2) - max(px1, sx1))
        intersection_height = max(0.0, min(py2, sy2) - max(py1, sy1))
        overlap = intersection_width * intersection_height / person_area
        if center_inside or overlap >= overlap_threshold:
            return True
    return False


def box_overlap_fraction(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection area divided by the first (target-person) box area."""
    x1, y1, x2, y2 = map(float, first)
    a1, b1, a2, b2 = map(float, second)
    intersection_width = max(0.0, min(x2, a2) - max(x1, a1))
    intersection_height = max(0.0, min(y2, b2) - max(y1, b1))
    area = max(1.0, x2 - x1) * max(1.0, y2 - y1)
    return intersection_width * intersection_height / area


def classify_occlusion(
    bbox: Sequence[float],
    other_boxes: Sequence[Sequence[float]],
    keypoint_confidences: Sequence[float],
    frame_width: int,
    frame_height: int,
    args,
) -> str:
    """Classify one detected person using pose visibility and spatial context."""
    if not keypoint_confidences:
        raise RuntimeError(
            "Occlusion labeling requires a YOLO pose model with keypoint confidence "
            "values (for example yolo11s-pose.pt)"
        )
    visible_ratio = sum(
        float(value) >= args.occlusion_keypoint_confidence
        for value in keypoint_confidences
    ) / len(keypoint_confidences)
    if visible_ratio >= args.occlusion_none_visible_ratio:
        return "none"

    severity = (
        "severe"
        if visible_ratio < args.occlusion_severe_visible_ratio
        else "partial"
    )
    x1, y1, x2, y2 = map(float, bbox)
    margin_x = frame_width * args.occlusion_boundary_fraction
    margin_y = frame_height * args.occlusion_boundary_fraction
    boundary = (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= frame_width - margin_x
        or y2 >= frame_height - margin_y
    )
    inter_person = any(
        box_overlap_fraction(bbox, other) >= args.occlusion_person_overlap
        for other in other_boxes
    )
    if boundary and inter_person:
        cause = "mixed"
    elif inter_person:
        cause = "inter_person"
    elif boundary:
        cause = "boundary"
    else:
        # Low pose visibility without person overlap or boundary truncation is
        # attributed to a scene object. This is a deterministic inference and
        # should be checked in frame_labels_vis.mp4.
        cause = "object"
    label = f"{cause}_{severity}"
    if label not in OCCLUSION_LABELS:
        raise RuntimeError(f"Internal unsupported occlusion label: {label}")
    return label


def smooth_occlusion_labels(frames: Sequence[Dict[str, object]]) -> int:
    """Remove isolated one-frame label flips independently for every track."""
    observations: Dict[int, List[Dict[str, object]]] = {}
    for frame in frames:
        for person in frame.get("persons", []):
            observations.setdefault(int(person["person_id"]), []).append(person)
    changed = 0
    for track in observations.values():
        original = [str(person["occlusion"]) for person in track]
        replacement = list(original)
        for index in range(1, len(original) - 1):
            if original[index - 1] == original[index + 1] != original[index]:
                replacement[index] = original[index - 1]
        for person, old, new in zip(track, original, replacement):
            if new != old:
                person["occlusion"] = new
                changed += 1
    return changed




def annotate_frames(args, provider, stream_id, frame_count: int) -> None:
    """Generate person boxes and sequence-local IDs with YOLO + BoT-SORT."""
    metadata = load_json(args.sequence / "metadata.json")
    detector = load_person_detector(args.person_model)
    screen_detector = (
        load_person_detector(args.screen_model) if args.exclude_screen_reflections else None
    )
    frames = []
    track_id_map: Dict[int, int] = {}
    next_person_id = 0
    detector_device = args.detector_device if args.detector_device else None
    screen_hits_by_person_id: Dict[int, int] = {}
    for frame_index in range(frame_count):
        rgb, timestamp_ns = get_raw_frame(provider, stream_id, frame_index)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        # Ultralytics treats an HWC uint8 ndarray as BGR; Project Aria returns RGB.
        bgr = rgb[:, :, ::-1].copy()
        result = detector.track(
            source=bgr,
            persist=True,
            tracker=args.tracker,
            classes=[args.person_class_id],
            conf=args.person_confidence,
            iou=args.person_iou,
            imgsz=args.detector_image_size,
            device=detector_device,
            verbose=False,
        )[0]

        screen_boxes: List[List[float]] = []
        if screen_detector is not None:
            screen_result = screen_detector.predict(
                source=bgr,
                classes=args.screen_class_ids,
                conf=args.screen_confidence,
                iou=args.screen_iou,
                imgsz=args.screen_image_size,
                device=detector_device,
                verbose=False,
            )[0]
            if screen_result.boxes is not None and len(screen_result.boxes) > 0:
                screen_boxes = screen_result.boxes.xyxy.detach().cpu().tolist()

        persons: List[Dict[str, object]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.detach().cpu().tolist()
            keypoints = getattr(result, "keypoints", None)
            keypoint_confidences = (
                keypoints.conf.detach().cpu().tolist()
                if keypoints is not None and keypoints.conf is not None
                else None
            )
            if keypoint_confidences is None or len(keypoint_confidences) != len(coordinates):
                raise RuntimeError(
                    "Per-person occlusion requires a YOLO pose model; "
                    f"the configured model produced {len(coordinates)} boxes without "
                    "matching keypoints"
                )
            raw_ids: List[Optional[int]]
            if boxes.id is None:
                raw_ids = [None] * len(coordinates)
            else:
                raw_ids = [int(value) for value in boxes.id.detach().cpu().tolist()]
            candidates: List[Dict[str, object]] = []
            for bbox, raw_id, pose_confidence in zip(
                coordinates, raw_ids, keypoint_confidences
            ):
                box_width = max(1.0, float(bbox[2]) - float(bbox[0]))
                box_height = max(1.0, float(bbox[3]) - float(bbox[1]))
                looks_like_egowearer = (
                    args.exclude_egowearer
                    and float(bbox[3]) >= height * args.egowearer_bottom_fraction
                    and box_width / box_height >= args.egowearer_min_aspect
                )
                if looks_like_egowearer:
                    continue
                if raw_id is None:
                    # Do not turn an unconfirmed single-frame detection into a person ID.
                    continue
                if raw_id not in track_id_map:
                    track_id_map[raw_id] = next_person_id
                    next_person_id += 1
                person_id = track_id_map[raw_id]
                if args.exclude_screen_reflections and box_inside_screen(
                    bbox,
                    screen_boxes,
                    width,
                    height,
                    args.screen_box_padding,
                    args.screen_overlap_threshold,
                ):
                    screen_hits_by_person_id[person_id] = (
                        screen_hits_by_person_id.get(person_id, 0) + 1
                    )
                candidates.append(
                    {
                        "person_id": person_id,
                        "raw_bbox": bbox,
                        "keypoint_confidences": pose_confidence,
                    }
                )

            for candidate_index, candidate in enumerate(candidates):
                bbox = candidate["raw_bbox"]
                x_min = max(0, min(width - 1, int(round(bbox[0]))))
                y_min = max(0, min(height - 1, int(round(bbox[1]))))
                x_max = max(x_min + 1, min(width, int(round(bbox[2]))))
                y_max = max(y_min + 1, min(height, int(round(bbox[3]))))
                other_boxes = [
                    row["raw_bbox"]
                    for index, row in enumerate(candidates)
                    if index != candidate_index
                ]
                persons.append(
                    {
                        "person_id": candidate["person_id"],
                        "bbox": [x_min, y_min, x_max, y_max],
                        "occlusion": classify_occlusion(
                            bbox,
                            other_boxes,
                            candidate["keypoint_confidences"],
                            width,
                            height,
                            args,
                        ),
                    }
                )

        # Track every frame for stable IDs, but optionally save a lower annotation rate.
        if frame_index % args.frame_stride == 0:
            frames.append(
                {
                    "frame_id": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "persons": persons,
                }
            )
        if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
            print(
                f"  tracked RGB frames: {frame_index + 1}/{frame_count}",
                file=sys.stderr,
            )
    reflection_ids = {
        person_id
        for person_id, hits in screen_hits_by_person_id.items()
        if hits >= args.screen_track_min_hits
    }
    track_hits: Dict[int, int] = {}
    for frame in frames:
        for person in frame["persons"]:
            person_id = int(person["person_id"])
            track_hits[person_id] = track_hits.get(person_id, 0) + 1
    short_track_ids = {
        person_id
        for person_id, hits in track_hits.items()
        if hits < args.min_track_hits
    }
    filtered_reflections = 0
    filtered_short_tracks = 0
    compact_ids: Dict[int, int] = {}
    for frame in frames:
        retained = []
        for person in frame["persons"]:
            old_id = int(person["person_id"])
            if old_id in reflection_ids:
                filtered_reflections += 1
                continue
            if old_id in short_track_ids:
                filtered_short_tracks += 1
                continue
            if old_id not in compact_ids:
                compact_ids[old_id] = len(compact_ids)
            person["person_id"] = compact_ids[old_id]
            retained.append(person)
        frame["persons"] = retained
    smoothed_occlusions = smooth_occlusion_labels(frames)
    write_json(
        args.sequence / "frame_labels.json",
        {"sequence_id": metadata["sequence_id"], "frames": frames},
    )
    if args.exclude_screen_reflections:
        print(
            "  filtered screen-reflection tracks/detections: "
            f"{len(reflection_ids)}/{filtered_reflections}"
        )
    print(
        "  filtered short tracks/detections: "
        f"{len(short_track_ids)}/{filtered_short_tracks}; "
        f"retained person tracks: {len(compact_ids)}; "
        f"smoothed occlusion labels: {smoothed_occlusions}"
    )


def person_appearance_feature(image, bbox: Sequence[float]):
    """Describe the central upper-body region with normalized color histograms."""
    import cv2
    import numpy as np

    x_min, y_min, x_max, y_max = map(int, bbox)
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    crop_x_min = max(0, x_min + int(0.20 * width))
    crop_x_max = min(image.shape[1], x_max - int(0.20 * width))
    crop_y_min = max(0, y_min + int(0.18 * height))
    crop_y_max = min(image.shape[0], crop_y_min + int(0.48 * height))
    crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_saturation = cv2.calcHist(
        [hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]
    ).reshape(-1)
    value = cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
    channels = [
        cv2.calcHist([crop], [channel], None, [8], [0, 256]).reshape(-1)
        for channel in range(3)
    ]
    parts = [
        hue_saturation / max(1.0, float(hue_saturation.sum())),
        value / max(1.0, float(value.sum())),
    ]
    parts.extend(channel / max(1.0, float(channel.sum())) for channel in channels)
    feature = np.concatenate(parts).astype(np.float32)
    return feature / max(1e-6, float(np.linalg.norm(feature)))


def cluster_appearance_features(
    features: Dict[int, List[object]], track_hits: Dict[int, int], cluster_count: int
) -> Dict[int, int]:
    """Cluster tracklets with deterministic weighted k-means initialization."""
    import numpy as np

    person_ids = sorted(features)
    if len(person_ids) < cluster_count:
        raise RuntimeError(
            f"Cannot form {cluster_count} identities from {len(person_ids)} tracks"
        )
    matrix = np.stack(
        [np.mean(features[person_id], axis=0) for person_id in person_ids]
    )
    matrix /= np.maximum(1e-6, np.linalg.norm(matrix, axis=1, keepdims=True))
    weights = np.asarray(
        [math.log1p(track_hits[person_id]) for person_id in person_ids],
        dtype=np.float32,
    )

    center_indices = [int(np.argmax(weights))]
    while len(center_indices) < cluster_count:
        distances = np.min(
            np.stack(
                [
                    np.sum((matrix - matrix[index]) ** 2, axis=1)
                    for index in center_indices
                ]
            ),
            axis=0,
        )
        distances[center_indices] = -1
        center_indices.append(int(np.argmax(distances * weights)))
    centers = matrix[center_indices].copy()

    for _ in range(30):
        distances = np.stack(
            [np.sum((matrix - center) ** 2, axis=1) for center in centers], axis=1
        )
        labels = np.argmin(distances, axis=1)
        updated = []
        for cluster in range(cluster_count):
            members = labels == cluster
            if not np.any(members):
                updated.append(centers[cluster])
                continue
            center = np.average(matrix[members], axis=0, weights=weights[members])
            center /= max(1e-6, float(np.linalg.norm(center)))
            updated.append(center)
        updated = np.stack(updated)
        if np.allclose(updated, centers, atol=1e-5):
            break
        centers = updated

    groups: Dict[int, List[int]] = {}
    for index, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(person_ids[index])
    ordered_groups = sorted(groups.values(), key=min)
    return {
        person_id: identity_id
        for identity_id, group in enumerate(ordered_groups)
        for person_id in group
    }


def reidentify_frame_labels(args, provider, stream_id) -> None:
    """Merge long fragmented tracklets into a known number of appearance identities."""
    if args.expected_person_count is None:
        raise RuntimeError("--expected-person-count is required for reidentify")
    path = args.sequence / "frame_labels.json"
    document = load_json(path)
    frames = document.get("frames", [])
    track_hits: Dict[int, int] = {}
    for frame in frames:
        for person in frame.get("persons", []):
            person_id = int(person["person_id"])
            track_hits[person_id] = track_hits.get(person_id, 0) + 1
    retained_ids = {
        person_id
        for person_id, hits in track_hits.items()
        if hits >= args.min_track_hits
    }
    sample_period = {
        person_id: max(1, track_hits[person_id] // args.appearance_samples_per_track)
        for person_id in retained_ids
    }
    observations: Dict[int, int] = {}
    sample_frames: Dict[int, List[Dict[str, object]]] = {}
    for frame in frames:
        selected = []
        for person in frame.get("persons", []):
            person_id = int(person["person_id"])
            if person_id not in retained_ids:
                continue
            observations[person_id] = observations.get(person_id, 0) + 1
            if observations[person_id] % sample_period[person_id] == 0:
                selected.append(person)
        if selected:
            sample_frames[int(frame["frame_id"])] = selected

    features: Dict[int, List[object]] = {}
    for frame_id, persons in sample_frames.items():
        rgb, _ = get_raw_frame(provider, stream_id, frame_id)
        bgr = rgb[:, :, ::-1].copy()
        for person in persons:
            person_id = int(person["person_id"])
            feature = person_appearance_feature(bgr, person["bbox"])
            if feature is not None:
                features.setdefault(person_id, []).append(feature)
    missing_features = sorted(retained_ids - features.keys())
    if missing_features:
        raise RuntimeError(
            "No appearance samples for retained tracks: "
            + ", ".join(map(str, missing_features))
        )
    identity_map = cluster_appearance_features(
        features, track_hits, args.expected_person_count
    )

    filtered_detections = 0
    merged_duplicates = 0
    for frame in frames:
        by_identity: Dict[int, Dict[str, object]] = {}
        for person in frame.get("persons", []):
            old_id = int(person["person_id"])
            if old_id not in identity_map:
                filtered_detections += 1
                continue
            identity_id = identity_map[old_id]
            person["person_id"] = identity_id
            previous = by_identity.get(identity_id)
            if previous is None:
                by_identity[identity_id] = person
                continue
            merged_duplicates += 1
            old_box = previous["bbox"]
            new_box = person["bbox"]
            old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
            new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            if new_area > old_area:
                by_identity[identity_id] = person
        frame["persons"] = [by_identity[key] for key in sorted(by_identity)]
    write_json(path, document)
    group_summary: Dict[int, List[int]] = {}
    for track_id, identity_id in identity_map.items():
        group_summary.setdefault(identity_id, []).append(track_id)
    print(
        f"  merged {len(identity_map)} tracklets into "
        f"{len(group_summary)} appearance identities: {group_summary}"
    )
    print(
        f"  reidentify filtered detections: {filtered_detections}; "
        f"merged duplicate boxes: {merged_duplicates}"
    )


def track_color(person_id: int) -> Tuple[int, int, int]:
    hue = (person_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def visualize_frame_labels(
    args,
    provider,
    stream_id,
    frame_count: int,
    frame_rate: float,
    image_module,
) -> None:
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Visualization requires imageio, imageio-ffmpeg, and Pillow") from error

    raw_labels = load_json(args.sequence / "frame_labels.json")
    if not isinstance(raw_labels, dict) or not isinstance(raw_labels.get("frames"), list):
        raise RuntimeError("frame_labels.json must contain a frames list")
    labeled_frames = [
        frame
        for frame in raw_labels["frames"]
        if isinstance(frame, dict)
        and isinstance(frame.get("frame_id"), int)
        and 0 <= frame["frame_id"] < frame_count
    ]
    labeled_frames.sort(key=lambda frame: frame["frame_id"])
    if args.vis_max_frames is not None:
        labeled_frames = labeled_frames[: args.vis_max_frames]
    if not labeled_frames:
        raise RuntimeError("frame_labels.json contains no valid labeled frames")

    frame_steps = [
        current["frame_id"] - previous["frame_id"]
        for previous, current in zip(labeled_frames, labeled_frames[1:])
        if current["frame_id"] > previous["frame_id"]
    ]
    label_stride = round(statistics.median(frame_steps)) if frame_steps else 1
    output_fps = frame_rate / max(1, label_stride)
    output = args.vis_output or (args.sequence / "frame_labels_vis.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    font_size = max(14, round(28 / args.vis_downsample))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    with imageio.get_writer(
        str(output), fps=output_fps, codec="libx264", macro_block_size=None
    ) as writer:
        for output_index, annotation in enumerate(labeled_frames, start=1):
            frame_id = int(annotation["frame_id"])
            rgb, _ = get_raw_frame(provider, stream_id, frame_id)
            image = image_module.fromarray(rgb)
            if args.vis_downsample > 1:
                image = image.resize(
                    (
                        max(1, image.width // args.vis_downsample),
                        max(1, image.height // args.vis_downsample),
                    )
                )
            draw = ImageDraw.Draw(image)
            scale = 1.0 / args.vis_downsample
            line_width = max(2, round(6 * scale))
            persons = annotation.get("persons", [])
            for person in persons:
                if not isinstance(person, dict):
                    continue
                bbox = person.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                person_id = int(person.get("person_id", -1))
                color = track_color(person_id)
                x_min, y_min, x_max, y_max = [round(float(value) * scale) for value in bbox]
                draw.rectangle(
                    [x_min, y_min, x_max, y_max], outline=color, width=line_width
                )
                occlusion = str(person.get("occlusion", "none"))
                label = f"person {person_id} | {occlusion}"
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                label_width = right - left + 8
                label_height = bottom - top + 6
                label_y = max(0, y_min - label_height)
                draw.rectangle(
                    [x_min, label_y, x_min + label_width, label_y + label_height],
                    fill=color,
                )
                draw.text((x_min + 4, label_y + 2), label, fill=(0, 0, 0), font=font)

            status = f"frame {frame_id} | persons {len(persons)}"
            status_box = draw.textbbox((0, 0), status, font=font)
            status_width = status_box[2] - status_box[0] + 12
            status_height = status_box[3] - status_box[1] + 10
            draw.rectangle([6, 6, 6 + status_width, 6 + status_height], fill=(0, 0, 0))
            draw.text((12, 10), status, fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(image))
            if output_index % 100 == 0 or output_index == len(labeled_frames):
                print(
                    f"  visualized labeled frames: {output_index}/{len(labeled_frames)}",
                    file=sys.stderr,
                )
    print(f"Visualization written: {output}")




def run(args: argparse.Namespace) -> None:
    recording_vrs = args.sequence / f"{args.sequence.name}.vrs"
    required = [recording_vrs.name, "metadata.json", "frame_labels.json"]
    missing = [name for name in required if not (args.sequence / name).is_file()]
    if missing:
        raise RuntimeError("Sequence is missing: " + ", ".join(missing))

    visibility_variables = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    previous_visibility = {name: os.environ.get(name) for name in visibility_variables}
    data_provider, image_module = load_modules(args.cpu_decode)
    provider = data_provider.create_vrs_data_provider(
        str(recording_vrs.resolve())
    )
    if provider is None:
        raise RuntimeError(f"Cannot open sequence recording: {recording_vrs.name}")
    if args.cpu_decode:
        for name, value in previous_visibility.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    stream_id = provider.get_stream_id_from_label("camera-rgb")
    vrs_frame_count = provider.get_num_data(stream_id)
    metadata = load_json(args.sequence / "metadata.json")
    frame_count = min(
        vrs_frame_count, int(metadata.get("num_frames", vrs_frame_count))
    )
    frame_rate = float(provider.get_image_configuration(stream_id).nominal_rate_hz)

    if args.dry_run:
        print("Frame-label dry run: no files changed")
        print(f"  RGB frames: {frame_count} at {frame_rate:g} Hz")
        print("  person schema:", json.dumps(FRAME_PERSON_FIELDS))
        print(
            f"  bbox backend: {args.person_model} + {args.tracker}; "
            f"process every frame, save every {args.frame_stride} frame(s)"
        )
        print(f"  visualization enabled: {args.visualize}")
        if args.visualize:
            print(
                "  visualization output:",
                args.vis_output or (args.sequence / "frame_labels_vis.mp4"),
            )
        return

    annotate_frames(args, provider, stream_id, frame_count)
    if args.expected_person_count is not None:
        reidentify_frame_labels(args, provider, stream_id)
    if args.visualize:
        visualize_frame_labels(
            args, provider, stream_id, frame_count, frame_rate, image_module
        )
    print("Frame labels completed; review detector outputs before publication")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="save one frame label every N frames; tracking still processes every frame",
    )
    parser.add_argument("--person-model", default="yolo11s-pose.pt")
    parser.add_argument(
        "--screen-model",
        default="yolo11s.pt",
        help="general object detector used to locate TV/screen regions",
    )
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument(
        "--tracker",
        default=str(Path(__file__).resolve().with_name("botsort_reid.yaml")),
    )
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument("--expected-person-count", type=int)
    parser.add_argument("--appearance-samples-per-track", type=int, default=30)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--person-iou", type=float, default=0.50)
    parser.add_argument("--detector-image-size", type=int, default=1280)
    parser.add_argument("--screen-image-size", type=int, default=1280)
    parser.add_argument("--detector-device")
    parser.add_argument("--occlusion-keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--occlusion-none-visible-ratio", type=float, default=0.80)
    parser.add_argument("--occlusion-severe-visible-ratio", type=float, default=0.30)
    parser.add_argument("--occlusion-person-overlap", type=float, default=0.15)
    parser.add_argument("--occlusion-boundary-fraction", type=float, default=0.02)
    parser.add_argument(
        "--exclude-egowearer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--egowearer-min-aspect", type=float, default=0.75)
    parser.add_argument("--egowearer-bottom-fraction", type=float, default=0.95)
    parser.add_argument(
        "--exclude-screen-reflections",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--screen-class-ids", type=int, nargs="+", default=[62])
    parser.add_argument("--screen-confidence", type=float, default=0.15)
    parser.add_argument("--screen-iou", type=float, default=0.80)
    parser.add_argument("--screen-box-padding", type=float, default=0.03)
    parser.add_argument("--screen-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--screen-track-min-hits", type=int, default=1)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write frame_labels_vis.mp4; use --no-visualize to skip it",
    )
    parser.add_argument("--vis-output", type=Path)
    parser.add_argument("--vis-downsample", type=int, default=2)
    parser.add_argument("--vis-max-frames", type=int)
    parser.add_argument("--cpu-decode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.sequence.is_dir():
        parser.error(f"Sequence directory does not exist: {args.sequence}")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    if args.detector_image_size < 32 or args.screen_image_size < 32:
        parser.error("detector image sizes must be at least 32")
    if args.vis_downsample < 1:
        parser.error("--vis-downsample must be positive")
    if args.vis_max_frames is not None and args.vis_max_frames < 1:
        parser.error("--vis-max-frames must be positive")
    if not 0.0 <= args.person_confidence <= 1.0 or not 0.0 <= args.person_iou <= 1.0:
        parser.error("detector confidence and IoU must be between 0 and 1")
    if not 0.0 <= args.screen_confidence <= 1.0 or not 0.0 <= args.screen_iou <= 1.0:
        parser.error("screen confidence and IoU must be between 0 and 1")
    if args.screen_box_padding < 0:
        parser.error("--screen-box-padding must be non-negative")
    if not 0.0 <= args.screen_overlap_threshold <= 1.0:
        parser.error("--screen-overlap-threshold must be between 0 and 1")
    if args.screen_track_min_hits < 1 or args.min_track_hits < 1:
        parser.error("track hit thresholds must be positive")
    if args.expected_person_count is not None and args.expected_person_count < 1:
        parser.error("--expected-person-count must be positive")
    if args.appearance_samples_per_track < 1:
        parser.error("--appearance-samples-per-track must be positive")
    if args.egowearer_min_aspect <= 0:
        parser.error("--egowearer-min-aspect must be positive")
    if not 0.0 <= args.egowearer_bottom_fraction <= 1.0:
        parser.error("--egowearer-bottom-fraction must be between 0 and 1")
    unit_interval_arguments = (
        ("--occlusion-keypoint-confidence", args.occlusion_keypoint_confidence),
        ("--occlusion-none-visible-ratio", args.occlusion_none_visible_ratio),
        ("--occlusion-severe-visible-ratio", args.occlusion_severe_visible_ratio),
        ("--occlusion-person-overlap", args.occlusion_person_overlap),
        ("--occlusion-boundary-fraction", args.occlusion_boundary_fraction),
    )
    for name, value in unit_interval_arguments:
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    if args.occlusion_severe_visible_ratio >= args.occlusion_none_visible_ratio:
        parser.error(
            "--occlusion-severe-visible-ratio must be smaller than "
            "--occlusion-none-visible-ratio"
        )
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
