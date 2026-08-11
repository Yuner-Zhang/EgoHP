# Official Project Aria Conversion Example

This example uses Meta's official Gen 1 `mps_sample`: an Aria VRS recording and
its matching MPS closed-loop trajectory. The source files are downloaded from a
pinned revision of the Apache-2.0-licensed
[`facebookresearch/projectaria_tools`](https://github.com/facebookresearch/projectaria_tools)
repository.

```bash
pip install -r requirements-conversion.txt
python tools/convert_aria_recording.py \
  --vrs examples/official_projectaria/raw/sample.vrs \
  --trajectory-csv examples/official_projectaria/raw/closed_loop_trajectory.csv \
  --output examples/official_projectaria/converted/seq_official_sample \
  --sequence-id seq_official_sample \
  --scene-name official_projectaria_sample
```

The example source is Meta's official
[`data/gen1/mps_sample`](https://github.com/facebookresearch/projectaria_tools/tree/main/data/gen1/mps_sample).
Place `sample.vrs` and `trajectory/closed_loop_trajectory.csv` in `raw/` before
running the command above.

`raw/` and the copied `video.vrs` are ignored by Git because the official VRS is
about 79 MB. The converter and compact converted files remain in this repository.

The MP4 contains the RGB stream without audio. Sensor, trajectory, and label
files retain synchronized Project Aria device timestamps in integer nanoseconds.
