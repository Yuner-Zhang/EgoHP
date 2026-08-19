# Quick Start: Official Project Aria Example

EgoHP supports every sequence in the current
[Project Aria Gen2 Pilot Dataset](https://explorer.projectaria.com/gen2pilot)
that provides a main VRS and the required MPS SLAM packages. This guide uses
`play_0` as one example. VRS, MPS, YOLO11 weights, local directories, and EgoHP
outputs are handled automatically.

The current official release contains 12 sequences:

```text
clean_0
cook_0
eat_0   eat_1   eat_2   eat_3
play_0  play_1  play_2  play_3
walk_0  walk_1
```

Choose any name above with `--sequence`. For example, replace `play_0` with
`walk_1` to download and process the complete `walk_1` sequence. The script
reads the live official manifest, so newly published compatible sequences do
not require hard-coded download URLs.

## 1. Create the environment

Run in Ubuntu 20.04 or WSL Ubuntu 20.04:

```bash
git clone https://github.com/Yuner-Zhang/EgoHP.git
cd EgoHP

conda env create -f environment.yml
conda activate egohp_gen2
```

`environment.yml` installs the required software packages. The command in
Step 3 automatically downloads missing YOLO11 weights to `data/models/`.

## 2. Configure the vision API

The default pipeline uses OpenAI's vision API to generate the scene-level
labels in `metadata.json`. Replace `PASTE_YOUR_OWN_API_KEY_HERE` with your own
OpenAI API key and save it once in the conda environment:

```bash
conda env config vars set EGOHP_API_KEY="PASTE_YOUR_OWN_API_KEY_HERE" -n egohp_gen2
conda deactivate
conda activate egohp_gen2
```

Only `EGOHP_API_KEY` is required for the default API configuration. Do not use
the placeholder text and do not commit the real key to Git. Confirm that the
key is available without printing it:

```bash
python -c "import os; print('API key configured:', bool(os.getenv('EGOHP_API_KEY')))"
```

## 3. Run the complete example

```bash
python tools/run_official_example.py --sequence play_0
```

Add either option to the same command when the corresponding final output is
not needed:

```text
--no-visualize   # do not generate frame_labels_vis.mp4
--no-keep-mps    # do not retain mps/slam/ in the final sequence
```

For example, omit both optional outputs with:

```bash
python tools/run_official_example.py --sequence play_0 --no-visualize --no-keep-mps
```

Without these options, both `frame_labels_vis.mp4` and `mps/slam/` are retained.

This single command:

1. Reads the current Project Aria Gen2 Pilot download manifest.
2. Downloads and verifies the complete official `play_0` VRS.
3. Downloads and extracts its trajectory, calibration, and semidense MPS data.
4. Downloads `yolo11s-pose.pt` and `yolo11s.pt` when missing.
5. Runs conversion, person labeling, temporal labeling, and metadata labeling.
6. Places the result under the API-inferred indoor/outdoor scene directory.

The script processes the full recording and uses GPU `0` for YOLO by default.

## Automatically created directories

No `data/` directories need to be created manually:

```text
data/
|-- raw/0/
|   `-- AriaGen2PilotDataset_v1.0_play_0_main_recording.vrs
|-- models/
|   |-- yolo11s-pose.pt
|   `-- yolo11s.pt
`-- converted/
    |-- .egohp_downloads/       # resumable official MPS archives
    |-- .egohp_cache/           # extracted reusable MPS
    |-- .egohp_staging/         # temporary processing directory
    |-- recording_index.json
    |-- collectors.json          # fill gender and height manually
    |-- Indoor/
    `-- Outdoor/
```

If `data` is a symbolic link, all directories are created at its target.
`raw/0/` gives this recording `collector_id: 0`; use `--collector-id N` to
select another numeric collector folder. The script creates a null-valued
collector template when needed, without overwriting existing collector data.

The final location is normally:

```text
data/converted/Indoor/living_room/seq_play_0/
```

Its contents include:

```text
seq_play_0.vrs
seq_play_0.mp4
mps/slam/
imu_left.txt
imu_right.txt
trajectory.txt
frame_labels.json
temporal_labels.json
metadata.json
frame_labels_vis.mp4
```

## Notes

- Keep about 20 GB free for the full VRS, MPS archives, extracted cache, and
  generated outputs.
- Interrupted official downloads are resumed when the server supports it.
- If `EGOHP_API_KEY` is reported missing, reactivate the conda environment.
- If CUDA inference fails, rerun with `--detector-device cpu`.
- If an interrupted run left staging files, inspect them and rerun with
  `--overwrite-staging`.

The official dataset and its formats are documented by
[Project Aria Gen2 Pilot](https://github.com/facebookresearch/projectaria_gen2_pilot_dataset)
and the [Dataset Explorer](https://explorer.projectaria.com/gen2pilot/play_0).
