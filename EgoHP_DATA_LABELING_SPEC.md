# EgoHP Data and Annotation Specification

This document defines the EgoHP dataset layout, file schemas, label vocabulary,
and annotation rules used to prepare and review published sequences.

## 1. Dataset Layout

```text
EgoHP_data/
|-- raw/
|   |-- 0/                              # collector_id = 0
|   |   `-- <recording>.vrs
|   `-- 1/                              # collector_id = 1
|       `-- <recording>.vrs
`-- converted/
    |-- collectors.json                 # manually maintained
    |-- recording_index.json            # raw-to-converted lookup
    |-- Indoor/
    |   `-- <scene_name>/
    |       `-- seq_xxx/
    |           |-- seq_xxx.vrs
    |           |-- seq_xxx.mp4
    |           |-- mps/
    |           |   `-- slam/
    |           |       |-- closed_loop_trajectory.csv
    |           |       |-- open_loop_trajectory.csv
    |           |       |-- online_calibration.jsonl
    |           |       |-- semidense_observations.csv.gz
    |           |       `-- semidense_points.csv.gz
    |           |-- imu_left.txt
    |           |-- imu_right.txt
    |           |-- trajectory.txt
    |           |-- metadata.json
    |           |-- frame_labels.json
    |           |-- temporal_labels.json
    |           `-- frame_labels_vis.mp4 # optional review video
    `-- Outdoor/
        `-- <scene_name>/
            `-- seq_xxx/
                `-- ...
```

The numeric directory immediately above a raw VRS is the anonymous collector
ID. For example, `raw/0/walk.vrs` is assigned `collector_id: 0`.

## 2. Processing Workflow

| Step | Script | Output |
|---|---|---|
| 1 | `prepare_data.py` | Acquires or reuses MPS data and runs the complete workflow. |
| 2 | `convert_to_egohp.py` | Copies the VRS and exports MP4, IMU, trajectory, and MPS files. |
| 3 | `generate_frame_labels.py` | Generates person boxes, track IDs, and per-person occlusion labels. |
| 4 | `generate_temporal_labels.py` | Generates adaptive person-distance and motion segments. |
| 5 | `generate_metadata.py` | Generates sequence metadata, routes the sequence, and updates the recording index. |

For an official Aria Gen 2 Pilot recording, published MPS archives are
downloaded and verified using their SHA-1 checksums. A newly collected VRS can
be processed by Meta MPS, or an existing complete MPS directory can be supplied
with `--mps-dir`.

MPS files are retained in each sequence by default. With `--no-keep-mps`, they
remain available during conversion and temporal labeling and are removed only
after all annotations are complete.

## 3. Sensor and Derived Files

### 3.1 `<sequence_id>.vrs`

The original Project Aria sensor recording. The file is copied without
modifying its sensor streams and is renamed to match the sequence ID.

### 3.2 `<sequence_id>.mp4`

A playable export of the complete RGB stream. Its frame order matches
`frame_id` in `frame_labels.json`. Frame limits and downsampling are intended
for debugging only and should not be used for a published sequence.

### 3.3 `imu_left.txt` and `imu_right.txt`

Both files are comma-separated text with the following fixed column order:

```text
timestamp_ns,w_x,w_y,w_z,a_x,a_y,a_z
```

| Field | Unit | Description |
|---|---:|---|
| `timestamp_ns` | ns | Project Aria device timestamp. |
| `w_x`, `w_y`, `w_z` | rad/s | Angular velocity in the corresponding IMU frame. |
| `a_x`, `a_y`, `a_z` | m/s² | Linear acceleration in the corresponding IMU frame. |

`imu_left.txt` is used for camera-motion classification. The synchronized right
IMU is retained as an additional sensor stream.

### 3.4 `trajectory.txt`

This file is simplified from `mps/slam/closed_loop_trajectory.csv` and uses the
following column order:

```text
timestamp_ns,tx,ty,tz,qx,qy,qz,qw
```

| Field | Unit | Description |
|---|---:|---|
| `timestamp_ns` | ns | Trajectory timestamp converted to integer nanoseconds. |
| `tx`, `ty`, `tz` | m | Device position in the MPS world frame. |
| `qx`, `qy`, `qz`, `qw` | — | Device orientation quaternion in the MPS world frame. |

Only trajectory samples that overlap the RGB recording are retained. EgoHP
uses the MPS closed-loop trajectory rather than on-device odometry.

## 4. Dataset-Level Files

### 4.1 `collectors.json`

Collector attributes are entered manually. The API and processing scripts do
not infer gender or height.

```json
{
  "collectors": [
    {
      "collector_id": 0,
      "gender": null,
      "height_cm": null
    }
  ]
}
```

| Field | Description |
|---|---|
| `collector_id` | Anonymous non-negative integer matching the raw directory name. |
| `gender` | Manually entered collector gender; `null` when not provided. |
| `height_cm` | Manually entered height in centimetres; `null` when not provided. |

### 4.2 `recording_index.json`

This file is the global mapping from each raw recording to its converted
sequence. Paths are stored relative to the `converted/` root so the dataset can
be moved without breaking the mapping.

```json
{
  "schema_version": 1,
  "recordings": [
    {
      "source_vrs": "../raw/0/example.vrs",
      "source_vrs_filename": "example.vrs",
      "sequence_id": "seq_001",
      "collector_id": 0,
      "converted_sequence": "Indoor/living_room/seq_001",
      "converted_vrs": "Indoor/living_room/seq_001/seq_001.vrs",
      "converted_mp4": "Indoor/living_room/seq_001/seq_001.mp4",
      "environment": "indoor",
      "scene_name": "living_room"
    }
  ]
}
```

An existing entry with the same `source_vrs` or `sequence_id` is replaced
rather than duplicated. The index is updated only after the completed sequence
has been moved into its final scene directory.

## 5. `metadata.json`

`metadata.json` contains one set of sequence-level labels and recording
statistics for each continuous recording.

```json
{
  "sequence_id": "seq_001",
  "scene_name": "station_hall",
  "environment": "indoor",
  "time_of_day": "day",
  "weather": null,
  "crowd_density": "high",
  "occlusion_level": "medium",
  "collector_id": 0,
  "num_frames": 3751,
  "frame_rate": 30.0,
  "duration_sec": 125.0,
  "trajectory_length_m": 133.36
}
```

Eight chronologically and uniformly sampled RGB frames are used for the visual
scene labels. Sequence identifiers and sensor statistics are never inferred by
the vision API.

### 5.1 Identification and Scene Fields

| Field | Values | Source or rule |
|---|---|---|
| `sequence_id` | `seq_xxx` | Supplied when the pipeline is started. |
| `collector_id` | Non-negative integer | Read from `raw/<collector_id>/`. |
| `scene_name` | Lowercase `snake_case` | Concise semantic location, such as `living_room` or `station_hall`. |
| `environment` | `indoor`, `outdoor` | Enclosed building interiors are indoor; other environments are outdoor. |

`environment` and `scene_name` determine the final sequence directory.

### 5.2 `time_of_day`

| Label | Definition |
|---|---|
| `day` | Daylight or a clearly daytime environment. |
| `dawn_dusk` | Twilight, low-angle sunlight, or transitional dawn/dusk lighting. |
| `night` | Nighttime or a scene primarily illuminated by artificial light. |
| `null` | No reliable visual evidence is available. |

The label describes the overall scene and is not determined from the
brightness of a single frame.

### 5.3 `weather`

| Label | Definition |
|---|---|
| `clear` | Reliable visual evidence of clear weather. |
| `cloudy` | Visible cloud cover or an overcast sky. |
| `rain` | Visible rainfall, raindrops, or other reliable rain evidence. |
| `fog` | Fog or haze that clearly reduces visibility. |
| `snow` | Visible snowfall or an unambiguous snowy-weather scene. |
| `null` | Weather cannot be judged reliably from the sampled frames. |

Indoor recordings are not forced to `null`: weather may be labeled when
outdoor conditions are clearly visible through a window.

### 5.4 `crowd_density`

Crowd density uses a representative simultaneous visible-person count across
the sampled frames. It is not the number of unique track IDs accumulated over
the full recording.

| Label | Representative visible people |
|---|---:|
| `empty` | 0–2 |
| `low` | 3–5 |
| `medium` | 6–20 |
| `high` | 21–50 |
| `very_high` | More than 50 |

Here, `empty` means no crowd or very few visible people rather than exactly
zero people.

### 5.5 `occlusion_level`

This sequence-level field summarizes how frequently people are occluded across
all person-frame annotations. Detailed causes and severities remain in
`frame_labels.json`.

- `occluded_ratio`: fraction of person-frame labels whose `occlusion` is not
  `none`.
- `severe_ratio`: fraction of person-frame labels ending in `_severe`.

| Label | Rule |
|---|---|
| `low` | `occluded_ratio < 20%` and `severe_ratio < 5%`. |
| `medium` | Not low, and `occluded_ratio < 50%` and `severe_ratio < 20%`. |
| `high` | `occluded_ratio >= 50%` or `severe_ratio >= 20%`. |
| `null` | The sequence contains no person boxes to summarize. |

This label measures person visibility. It does not measure missed detections or
identity switches.

### 5.6 Automatically Calculated Fields

| Field | Calculation |
|---|---|
| `num_frames` | Number of exported RGB frames. |
| `frame_rate` | Nominal RGB frame rate reported by the VRS. |
| `duration_sec` | `(last RGB timestamp - first RGB timestamp) / 1e9`. |
| `trajectory_length_m` | Sum of 3D Euclidean distances between consecutive closed-loop poses. |

## 6. `frame_labels.json`

This file stores per-frame person boxes, sequence-local track IDs, and
per-person occlusion labels.

```json
{
  "sequence_id": "seq_001",
  "frames": [
    {
      "frame_id": 1250,
      "timestamp_ns": 123456789000,
      "persons": [
        {
          "person_id": 3,
          "bbox": [542, 186, 693, 612],
          "occlusion": "inter_person_partial"
        }
      ]
    }
  ]
}
```

| Field | Description |
|---|---|
| `frame_id` | Zero-based RGB/MP4 frame index. |
| `timestamp_ns` | Original VRS RGB device timestamp. |
| `person_id` | Integer identity valid only within the current sequence. |
| `bbox` | RGB pixel coordinates `[x_min, y_min, x_max, y_max]`. |
| `occlusion` | Occlusion cause and severity; always present when a bbox exists. |

### 6.1 Person Detection and Tracking

The full pipeline uses the following annotation settings:

- YOLO pose detection restricted to the person class.
- Person confidence threshold: `0.35`.
- Detector input size: `1280` pixels.
- BoT-SORT association across consecutive RGB frames.
- Minimum retained track length: 3 detections.
- Bounding boxes clipped to the RGB image boundary.
- Unconfirmed detections without a stable tracker ID are not assigned a
  permanent `person_id`.

If the true number of people in a recording is known, fragmented tracks may be
merged with `--expected-person-count N`. Appearance matching uses up to 30
upper-body crops per track and normalized colour histograms. This option should
not be used when the number of identities is uncertain.

### 6.2 `occlusion`

Allowed labels are:

```text
none
inter_person_partial
inter_person_severe
object_partial
object_severe
boundary_partial
boundary_severe
mixed_partial
mixed_severe
```

The prefix describes the main cause:

- `inter_person`: another person overlaps the target person.
- `object`: a scene object obscures the person.
- `boundary`: the body is truncated by an image boundary.
- `mixed`: more than one cause is present.

The suffix describes severity. YOLO Pose provides 17 body-keypoint confidence
values. A keypoint is visible when its confidence is at least `0.35`.

| Severity | Visible-keypoint ratio |
|---|---:|
| `none` | At least 80% |
| `partial` | At least 30% but less than 80% |
| `severe` | Less than 30% |

Cause rules:

- `inter_person` when overlap with another person covers at least 15% of the
  target bbox.
- `boundary` when the bbox lies within 2% of any image edge.
- `mixed` when both inter-person overlap and boundary truncation are present.
- `object` when keypoint visibility indicates occlusion without person overlap
  or boundary contact.

An isolated one-frame occlusion change is replaced by the matching labels on
its adjacent frames. Bbox overlap is only an approximation of physical
occlusion, so `frame_labels_vis.mp4` should be used for quality review.

### 6.3 Screen and Wearer-Body Filtering

A bottom-of-frame person detection is removed as likely wearer body when both
conditions hold:

- `y_max >= 0.95 × image_height`.
- `bbox_width / bbox_height >= 0.75`.

TV/screen filtering uses COCO class 62 with confidence `0.15`. The detected
screen box is expanded by 3%. A person track is removed when its center lies in
the expanded screen box or at least 35% of its bbox overlaps the screen. This
filter should be checked carefully when a real person stands in front of a
screen.

## 7. `temporal_labels.json`

Temporal labels use adaptive segmentation rather than fixed ten-second clips.
The recording is first analyzed in approximately two-second windows. Adjacent
windows are merged when the visible identity set, every person's distance
level and relative motion, and camera motion are unchanged.

`start_frame` and `end_frame` are inclusive, so a final segment may be shorter
or longer than two seconds.

```json
{
  "sequence_id": "seq_001",
  "segments": [
    {
      "segment_id": 1,
      "start_frame": 0,
      "end_frame": 299,
      "start_timestamp_ns": 123456789000,
      "end_timestamp_ns": 133356789000,
      "persons": [
        {
          "person_id": 3,
          "person_distance_m": 1.43,
          "human_distance_level": "close",
          "relative_motion": "approaching"
        }
      ],
      "camera_motion": "walking"
    }
  ]
}
```

| Field | Description |
|---|---|
| `segment_id` | One-based segment index. |
| `start_frame`, `end_frame` | Inclusive first and last RGB frame. |
| `start_timestamp_ns`, `end_timestamp_ns` | First and last VRS RGB timestamps. |
| `persons` | Stable visible identities in the segment, sorted by `person_id`. |
| `persons[].person_id` | Identity matching `frame_labels.json`. |
| `persons[].person_distance_m` | Metric distance for that person. |
| `persons[].human_distance_level` | Discrete level derived from metric distance. |
| `persons[].relative_motion` | One dominant person-relative motion state. |
| `camera_motion` | Camera motion derived from trajectory and IMU. |

### 7.1 `person_distance_m`

Metric person distance is derived only from MPS geometry. Bbox size is never
converted directly into metres.

1. Sample one frame every 10 frames among frames containing person boxes.
2. Match each RGB timestamp to `slam-front-left` semidense observations within
   50 ms.
3. Use at most 5,000 candidate points per sampled frame.
4. Retain points with `inverse_distance_std <= 0.005` and
   `distance_std <= 0.01`.
5. Transform MPS world points into the RGB camera frame using the closed-loop
   pose and online calibration.
6. Project the points into RGB and retain points inside each person's bbox.
7. Require at least five valid points for a person in one frame.
8. Use the 20th percentile of valid point distances for that person-frame.
9. Keep distances separated by `person_id`.
10. Use the 10th percentile of valid person-frame distances in the final
    segment as `person_distance_m`.

The value is `null` when required MPS files are absent or too few reliable
points project into the person's bbox.

### 7.2 `human_distance_level`

| Label | Metric interval |
|---|---:|
| `very_close` | `< 1 m` |
| `close` | `1 m <= distance < 3 m` |
| `medium` | `3 m <= distance < 10 m` |
| `far` | `10 m <= distance <= 30 m` |
| `very_far` | `> 30 m` |
| `null` | `person_distance_m` is unavailable. |

### 7.3 `relative_motion`

Relative motion describes how each person moves relative to the wearer/camera.

| Label | Metric rule | Visual fallback without depth |
|---|---|---|
| `stationary` | Distance change `< 0.30 m` and radial speed `< 0.20 m/s` in magnitude. | Bbox linear scale changes by at most 10% and horizontal center changes by less than 5% of image width. |
| `approaching` | Distance decreases by at least 0.50 m with radial speed `<= -0.15 m/s`. | Bbox linear scale increases by at least 25%. |
| `receding` | Distance increases by at least 0.50 m with radial speed `>= 0.15 m/s`. | Bbox linear scale decreases by at least 20%. |
| `crossing` | Bearing changes by at least 20°, or relative lateral displacement is at least 1 m. | Horizontal span is at least 20% of image width while bbox scale remains stable. |
| `same_direction` | Person and camera displacements are each at least 0.50 m, direction cosine is at least 0.50, and separation changes by less than 0.50 m. | Stable scale and image position while the camera translates. |
| `opposite_direction` | Person and camera displacements are each at least 0.50 m and direction cosine is at most -0.50. | A supported approach-then-recede passing pattern. |

One person receives exactly one relative-motion state per segment. When
multiple rules match, priority is:

```text
opposite_direction -> same_direction -> approaching -> receding -> crossing -> stationary
```

If no motion rule reaches its threshold, the person is labeled `stationary`.
Therefore, `relative_motion` is never `null` for a person retained in a
segment. Different people in the same segment may have different states.

### 7.4 `camera_motion`

Camera motion is derived from `trajectory.txt` and `imu_left.txt`. Trajectory
features are sampled at approximately 50 ms intervals, and accumulated
rotation uses approximately 250 ms anchors.

| Label | Rule |
|---|---|
| `elevator` | Reliable vertical axis, vertical displacement `>= 1.0 m`, horizontal displacement `< 0.80 m`, and dynamic acceleration RMS `< 0.80 m/s²`. |
| `stairs` | Reliable vertical axis, vertical displacement `>= 0.60 m`, and dynamic acceleration RMS `>= 0.80 m/s²`. |
| `rapid_motion` | Median speed `> 2.20 m/s`, speed p95 `> 3.0 m/s`, or dynamic acceleration RMS `> 3.0 m/s²`. |
| `turning` | Net orientation change `>= 15°`, or accumulated rotation `>= 30°`. |
| `walking` | Path length `>= 0.25 m`, or median speed `>= 0.10 m/s`. |
| `stationary` | Complete trajectory and IMU evidence that does not reach a higher-priority threshold. |
| `null` | Fewer than two trajectory samples or fewer than two IMU samples. |

Classification priority is:

```text
elevator -> stairs -> rapid_motion -> turning -> walking -> stationary
```

Dynamic acceleration RMS is calculated from acceleration-vector magnitude
after subtracting standard gravity (`9.80665 m/s²`). A high gyro RMS alone does
not prevent `stationary`, because small head movements are common in wearable
camera recordings.

## 8. Meaning of `null`

`null` means that required evidence is unavailable or unreliable. It does not
mean that the object or condition does not exist. The string `"unknown"` is not
part of the label vocabulary.

| Field | Meaning of `null` |
|---|---|
| `weather` | The sampled frames do not provide reliable weather evidence. |
| `person_distance_m` | Required MPS geometry is missing or too few reliable points project into the bbox. |
| `human_distance_level` | `person_distance_m` is unavailable. |
| `camera_motion` | Trajectory or IMU contains fewer than two samples for the segment. |
| `occlusion_level` | No person-frame annotations exist to summarize. |
| `gender`, `height_cm` | Collector information has not been entered manually. |

`relative_motion` is not nullable for a retained person. Evidence below all
motion thresholds is represented by `stationary`.

## 9. Quality Review

Detector, tracker, and API outputs should be reviewed before publication.
Recommended checks include:

- Inspect `frame_labels_vis.mp4` for missed people, false detections, reflection
  filtering errors, ID switches, and occlusion errors.
- Confirm `scene_name`, indoor/outdoor routing, time of day, weather, and crowd
  density in `metadata.json`.
- Check that metric distances are plausible and that `null` distances coincide
  with insufficient MPS projection evidence.
- Inspect transitions between adaptive temporal segments, especially turning,
  walking, and person relative-motion changes.
