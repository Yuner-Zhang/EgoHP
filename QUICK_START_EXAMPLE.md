# Quick Start: Official Project Aria Example

This guide processes the first 20 seconds of the official `play_0` recording.
It covers the environment, official VRS/MPS data, YOLO11 weights, API setup,
and one complete EgoHP command.

## 1. Clone and create the environment

Run the following commands in Ubuntu 20.04 or WSL Ubuntu 20.04:

```bash
git clone https://github.com/Yuner-Zhang/EgoHP.git
cd EgoHP

conda env create -f environment.yml
conda activate egohp_gen2
```

The environment includes Project Aria, PyTorch, Ultralytics, video, and
tracking dependencies. An NVIDIA GPU is recommended for a complete recording,
but the short validation run below can use the CPU.

Create local data directories. The entire `data/` directory is ignored by Git.

```bash
mkdir -p data/raw/0 data/models data/converted
```

## 2. Download the official `play_0` VRS

Open the [Project Aria Gen2 Pilot `play_0` download page](https://explorer.projectaria.com/gen2pilot/play_0).
In **Downloads**, download the `main_vrs` asset and save it as:

```text
data/raw/0/AriaGen2PilotDataset_v1.0_play_0_main_recording.vrs
```

Keep the official filename. EgoHP uses it to identify `play_0` and match the
correct published MPS data. The parent directory `0` means
`collector_id: 0`.

Check the file before continuing:

```bash
ls -lh data/raw/0/AriaGen2PilotDataset_v1.0_play_0_main_recording.vrs
```

### How MPS is obtained

No MPS cloud login is required for this official recording. On the first run,
`tools/prepare_data.py` reads Project Aria's official download manifest and
automatically downloads these matching `play_0` packages:

- `mps_slam_trajectories`
- `mps_slam_calibration`
- `mps_slam_points`

Downloads resume when possible, are checked with the published SHA-1 values,
and are cached under `data/converted/.egohp_downloads/` and
`data/converted/.egohp_cache/`.

If automatic downloading is unavailable, download the same three MPS packages
from the `play_0` page, extract them into one directory containing `slam/`, and
run the pipeline with `--mps-dir /path/to/mps`. The `slam/` directory must
contain:

```text
closed_loop_trajectory.csv
open_loop_trajectory.csv
online_calibration.jsonl
semidense_observations.csv.gz
semidense_points.csv.gz
```

You do not need to download `video_main_rgb`; EgoHP exports its own MP4 from
the VRS.

## 3. Download the YOLO11 weights

EgoHP uses the pose model for person boxes/keypoints and the detection model to
reject people shown only on televisions or screens.

```bash
cd data/models
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt'); YOLO('yolo11s.pt')"
cd ../..

ls -lh data/models/yolo11s-pose.pt data/models/yolo11s.pt
```

Ultralytics downloads the pretrained weights automatically the first time the
model filenames are loaded.

## 4. Configure the vision API once

`generate_metadata.py` uses a vision API for scene-level labels. Store the key
in the conda environment so it is available after future activations:

```bash
conda env config vars set EGOHP_API_KEY="your-api-key" -n egohp_gen2
conda deactivate
conda activate egohp_gen2
```

The defaults are OpenAI's chat-completions endpoint and `gpt-4.1-mini`. For a
compatible service, save its endpoint and model as well:

```bash
conda env config vars set EGOHP_API_ENDPOINT="https://your-service/v1/chat/completions" -n egohp_gen2
conda env config vars set EGOHP_API_MODEL="your-vision-model" -n egohp_gen2
conda deactivate
conda activate egohp_gen2
```

Confirm that the key is present without printing it:

```bash
python -c "import os; print('API key configured:', bool(os.getenv('EGOHP_API_KEY')))"
```

## 5. Process the first 20 seconds

The official `play_0` RGB stream is 10 FPS, so 200 frames are approximately 20
seconds. This CPU command is slower but is the most portable first test:

```bash
python tools/prepare_data.py \
  --vrs data/raw/0/AriaGen2PilotDataset_v1.0_play_0_main_recording.vrs \
  --dataset-root data/converted \
  --sequence-id seq_play_0 \
  --person-model data/models/yolo11s-pose.pt \
  --screen-model data/models/yolo11s.pt \
  --detector-device cpu \
  --max-video-frames 200 \
  --expected-person-count 3 \
  --min-track-hits 20 \
  --cpu-decode \
  --no-keep-mps
```

`--no-keep-mps` removes the copied MPS directory from the final sequence after
labeling; the downloaded cache remains available for later runs. Omit this
option if the final sequence should retain `mps/slam/`.

For NVIDIA GPU person detection, change `--detector-device cpu` to
`--detector-device 0`. Keeping `--cpu-decode` is fine: it applies software
decoding to the VRS reader while YOLO can still use the selected GPU. Remove
`--cpu-decode` only when Project Aria hardware decoding works on the machine.

## 6. Inspect the result

The API places the sequence under the inferred environment and scene, normally:

```text
data/converted/Indoor/living_room/seq_play_0/
```

Important outputs are:

```text
seq_play_0.mp4
frame_labels_vis.mp4
imu_left.txt
imu_right.txt
trajectory.txt
frame_labels.json
temporal_labels.json
metadata.json
```

If the API chooses a different scene name, locate the result with:

```bash
find data/converted/Indoor data/converted/Outdoor -type d -name seq_play_0
```

The repository also contains a checked-in processed excerpt at
[`examples/official_projectaria/`](examples/official_projectaria/).

## Full recording

After the short result has been checked, copy the command and:

- change `--sequence-id seq_play_0` to `--sequence-id seq_play_0_full`;
- remove `--max-video-frames 200`;
- add `--mps-dir data/converted/.egohp_cache/seq_play_0/mps` to reuse the
  already downloaded official MPS data.

The new sequence ID prevents the full result from overwriting the short test.
Use `--overwrite-staging` if an interrupted run left files in its staging
directory.

Allow enough disk space: `play_0` includes a multi-gigabyte VRS, MPS point data,
download archives, cache files, and generated video. About 20 GB of free space
is a practical minimum for the complete example.

## Common failures

- **`EGOHP_API_KEY` is not set**: reactivate `egohp_gen2` after saving the conda
  environment variable.
- **Official Pilot sequence is not detected**: restore the official VRS
  filename shown in Step 2.
- **MPS download was interrupted**: rerun the same command; completed bytes are
  reused when the server supports resuming.
- **CUDA or VRS hardware decoding fails**: use `--detector-device cpu` together
  with `--cpu-decode`.
- **Staging sequence is not empty**: verify that it belongs to this sequence,
  then rerun with `--overwrite-staging`.
