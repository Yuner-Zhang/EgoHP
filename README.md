# EgoHP

EgoHP converts a Project Aria Gen 2 `.vrs` recording into a labeled
egocentric-human sequence. It exports sensor data, tracks people, creates
dynamic labels, and sorts each sequence by indoor/outdoor scene.

## Setup

Run these steps in Ubuntu 20.04 or WSL Ubuntu 20.04.

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate egohp_gen2
```

### 2. Add your own API key

The default pipeline uses OpenAI's vision API to generate the scene labels in
`metadata.json`. Replace `PASTE_YOUR_OWN_API_KEY_HERE` with your own OpenAI API
key:

```bash
conda env config vars set EGOHP_API_KEY="PASTE_YOUR_OWN_API_KEY_HERE" -n egohp_gen2
conda deactivate
conda activate egohp_gen2
```

> Do not use the placeholder text and do not commit your API key to Git.

Confirm that the key is available without printing the secret:

```bash
python -c "import os; print('API key configured:', bool(os.getenv('EGOHP_API_KEY')))"
```

### 3. Download the YOLO11 weights

```bash
mkdir -p data/models
cd data/models
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt'); YOLO('yolo11s.pt')"
cd ../..
```

`environment.yml` installs Ultralytics and the other software dependencies; it
does not contain pretrained `.pt` model files. The command above downloads the
two weights once to `data/models/`.

To download and process any sequence from the Project Aria Gen2 Pilot website,
follow [Quick Start: Official Project Aria Example](QUICK_START_EXAMPLE.md).
That workflow downloads VRS, MPS, and missing YOLO weights automatically.

## Data Structure

```text
data/
|-- raw/
|   |-- 0/                              # collector_id = 0
|   |   `-- <recording>.vrs
|   `-- 1/                              # collector_id = 1
|       `-- <recording>.vrs
`-- converted/
    |-- collectors.json                 # fill manually
    |-- recording_index.json            # raw-to-converted lookup
    |-- Indoor/
    |   `-- <scene_name>/
    |       `-- seq_xxx/
    `-- Outdoor/
        `-- <scene_name>/
            `-- seq_xxx/
```

Each `seq_xxx` contains:

```text
seq_xxx/
|-- seq_xxx.vrs                         # original recording copy
|-- seq_xxx.mp4                         # playable RGB video
|-- mps/slam/                           # retained by default
|   |-- closed_loop_trajectory.csv
|   |-- open_loop_trajectory.csv
|   |-- online_calibration.jsonl
|   |-- semidense_observations.csv.gz
|   `-- semidense_points.csv.gz
|-- imu_left.txt
|-- imu_right.txt
|-- trajectory.txt
|-- frame_labels.json
|-- temporal_labels.json
|-- metadata.json
`-- frame_labels_vis.mp4                 # optional review video
```

- `imu_left.txt`, `imu_right.txt`: `timestamp_ns,w_x,w_y,w_z,a_x,a_y,a_z`.
- `trajectory.txt`: `timestamp_ns,tx,ty,tz,qx,qy,qz,qw`, simplified from the
  closed-loop trajectory.
- `frame_labels.json`: per-frame person boxes, track IDs, and occlusion.
- `temporal_labels.json`: adaptive per-person distance and motion segments.
- `metadata.json`: scene-level labels and recording statistics.
- `collectors.json`: anonymous collector gender and height; edit manually.

## Process Your Own VRS

Run this command in Ubuntu 20.04/WSL:

```bash
python tools/prepare_data.py \
  --vrs data/raw/0/recording.vrs \
  --dataset-root data/converted \
  --sequence-id seq_001 \
  --person-model data/models/yolo11s-pose.pt \
  --screen-model data/models/yolo11s.pt \
  --detector-device 0 \
  --cpu-decode
```

The numeric raw folder supplies `collector_id`. For example,
`raw/0/recording.vrs` gives `collector_id: 0`.

Official Gen 2 Pilot recordings automatically download Meta's published MPS
files. A newly recorded VRS uses Meta MPS cloud; add `--mps-interactive-login`
on the first run. Use `--mps-dir /path/to/mps` when complete MPS results already
exist locally.

## Pipeline Steps

The steps run in order. `prepare_data.py` is the main entry point and calls the
remaining scripts automatically.

0. `run_official_example.py` — optionally downloads an official Pilot VRS,
   matching MPS, and YOLO11 weights before calling the pipeline.
1. `prepare_data.py` — obtains or reuses MPS data and runs the full pipeline.
2. `convert_to_egohp.py` — copies the VRS and exports MP4, IMU, trajectory, and
   MPS files.
3. `generate_frame_labels.py` — detects and tracks people, writes bbox and
   occlusion labels, and optionally creates the review video.
4. `generate_temporal_labels.py` — creates adaptive distance, relative-motion,
   and camera-motion segments from tracks, MPS geometry, trajectory, and IMU.
5. `generate_metadata.py` — calls the vision API for scene labels, moves the
   sequence to its final directory, and updates `recording_index.json`.

To rerun one labeling step:

```bash
python tools/generate_frame_labels.py --sequence /path/to/seq_xxx --cpu-decode
python tools/generate_temporal_labels.py --sequence /path/to/seq_xxx --cpu-decode
```

## Label Files

### `frame_labels.json`

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

`bbox` uses `[x_min, y_min, x_max, y_max]`. Occlusion is `none`, or a cause
(`inter_person`, `object`, `boundary`, `mixed`) plus `partial` or `severe`.

### `temporal_labels.json`

```json
{
  "sequence_id": "seq_001",
  "segments": [
    {
      "segment_id": 1,
      "start_frame": 0,
      "end_frame": 299,
      "start_timestamp_ns": 123456789000,
      "end_timestamp_ns": 133456789000,
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

- Human distance: `very_close` (<1 m), `close` (1–3 m), `medium` (3–10 m),
  `far` (10–30 m), or `very_far` (>30 m).
- Relative motion: `stationary`, `same_direction`, `opposite_direction`,
  `crossing`, `approaching`, or `receding`. Each person has one state per
  segment. Evidence below every motion threshold is labeled `stationary`, not
  `null`.
- Camera motion: `stationary`, `walking`, `turning`, `stairs`, `elevator`, or
  `rapid_motion`.
- Segments split when the visible people, distance level, relative motion, or
  camera motion changes; identical neighboring segments are merged.
- A value is JSON `null` when the required sensor evidence is unavailable.

### `metadata.json`

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
  "frame_rate": 30,
  "duration_sec": 125.0,
  "trajectory_length_m": 133.36
}
```

- `scene_name`, `environment`, `time_of_day`, `weather`, and `crowd_density`
  come from the vision API.
- Crowd density: `empty` (0–2 people), `low` (3–5), `medium` (6–20), `high`
  (21–50), or `very_high` (>50).
- `occlusion_level` summarizes all person-frame occlusion labels as `low`,
  `medium`, or `high`; it is `null` when no person is detected.
- Frame count, frame rate, duration, and trajectory length are calculated from
  the converted data.

Detector and API labels should be reviewed before publication.
