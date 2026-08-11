# EgoHP

EgoHP is an egocentric human perception dataset collected with Project Aria
glasses in crowded indoor and outdoor environments.

## Data Structure

```text
EgoHP/
|-- collectors.json
|-- Indoor/
|   `-- <scene_name>/
|       `-- seq_xxx/
`-- Outdoor/
    `-- <scene_name>/
        `-- seq_xxx/
```

Each sequence contains:

```text
seq_xxx/
|-- video.vrs
|-- video.mp4
|-- imu_left.txt
|-- imu_right.txt
|-- trajectory.txt
|-- metadata.json
|-- temporal_labels.json
`-- frame_labels.json
```

### Video

- `video.vrs`: raw recording containing the original Project Aria sensor streams.
- `video.mp4`: exported RGB video used for playback and annotation.

### IMU

`imu_left.txt` and `imu_right.txt` contain synchronized measurements from the
two IMUs. Both files are comma-separated text files with the same header:

```text
timestamp_ns,w_x,w_y,w_z,a_x,a_y,a_z
```

`timestamp_ns` is the Project Aria device timestamp. Angular velocity
`w_x, w_y, w_z` is in rad/s, and linear acceleration `a_x, a_y, a_z` is in
m/s². Both measurements use the corresponding IMU sensor frame.

### Trajectory

`trajectory.txt` contains the MPS-estimated 6DoF device pose. It is a
comma-separated text file with the following header:

```text
timestamp_ns,tx,ty,tz,qx,qy,qz,qw
```

`tx, ty, tz` is the device position in metres, and `qx, qy, qz, qw` is its
orientation quaternion. Both are expressed in the MPS world frame.

### Sequence Metadata

Each sequence has one `metadata.json`. The scene and collection conditions are
filled manually; video and trajectory statistics are calculated during data
processing.

```json
{
  "sequence_id": "seq_001",
  "scene_name": "station_hall",
  "time_of_day": "day",
  "weather": "clear",
  "crowd_density": "high",
  "occlusion_level": "partial",
  "collector_id": 1,
  "num_frames": 3751,
  "frame_rate": 30,
  "duration_sec": 125.0,
  "trajectory_length_m": 133.36
}
```

Manual fields:

- `sequence_id`: unique sequence name.
- `scene_name`: scene folder or location name.
- `time_of_day`: `day`, `dawn_dusk`, or `night`.
- `weather`: `clear`, `cloudy`, `rain`, `fog`, or `snow`.
- `crowd_density`: `empty` (0-2 visible people), `low` (3-5), `medium`
  (6-20), `high` (21-50), or `very_high` (>50).
- `occlusion_level`: overall scene occlusion: `none`, `partial`, or `severe`.
- `collector_id`: anonymous ID defined in `collectors.json`.

Automatically calculated fields are `num_frames`, `frame_rate`, `duration_sec`,
and `trajectory_length_m`.

### Temporal Labels

`temporal_labels.json` stores dynamic labels for approximately 10-second
segments. Frame ranges are zero-based and inclusive.

```json
{
  "sequence_id": "seq_001",
  "segments": [
    {
      "segment_id": 1,
      "start_frame": 0,
      "end_frame": 299,
      "start_timestamp_ns": 123456789000,
      "end_timestamp_ns": 133423455000,
      "nearest_person_distance_m": 1.2,
      "relative_motion": "approaching",
      "camera_motion": "walking"
    }
  ]
}
```

- `nearest_person_distance_m`: estimated distance to the nearest visible person.
- `relative_motion`: `stationary`, `same_direction`, `opposite_direction`,
  `crossing`, `approaching`, or `receding`.
- `camera_motion`: `stationary`, `walking`, `turning`, `stairs`, `elevator`, or
  `rapid_motion`.

Distance levels can be derived as `very_close` (<1 m), `close` (1-3 m),
`medium` (3-10 m), `far` (10-30 m), and `very_far` (>30 m).

### Frame Labels

`frame_labels.json` stores per-frame person annotations. Bounding boxes and
track IDs are produced later by a separate annotation or detection-and-tracking
program.

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
          "occlusion": "partial"
        }
      ]
    }
  ]
}
```

- `frame_id`: zero-based MP4 frame index.
- `timestamp_ns`: timestamp of the corresponding RGB frame.
- `person_id`: track ID kept consistent across frames in the same sequence.
- `bbox`: `[x_min, y_min, x_max, y_max]` in MP4 pixel coordinates.
- `occlusion`: `none`, `partial`, or `severe` for that person.

### Collector Information

The root-level `collectors.json` stores anonymous collector information shared
across sequences:

```json
{
  "collectors": [
    {
      "collector_id": 1,
      "gender": "female",
      "height_cm": 165
    }
  ]
}
```

`collector_id` is anonymous. `gender` and `height_cm` record the collector's
gender and height in centimetres.

All sensor and annotation timestamps use the original Project Aria device time
in integer nanoseconds (`timestamp_ns`).

## Conversion

```bash
pip install -r requirements-conversion.txt
python tools/convert_aria_recording.py \
  --vrs path/to/recording.vrs \
  --trajectory-csv path/to/closed_loop_trajectory.csv \
  --output EgoHP/Indoor/<scene_name>/seq_001 \
  --sequence-id seq_001 \
  --scene-name <scene_name>
```

See the [official Project Aria example](examples/official_projectaria/README.md).
