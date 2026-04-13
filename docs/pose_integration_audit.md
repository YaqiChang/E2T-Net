# Pose Integration Audit

## Purpose
Stage 0 repository audit for connecting offline HRNet pose NPZ files into the current E2T-Net pipeline without changing baseline behavior yet.

## Files inspected
- `AGENTS.md`
- `PLAN.md`
- `train.py`
- `scripts/train_e2t_proto.py`
- `datasets/__init__.py`
- `datasets/jaad.py`
- `datasets/pie.py`
- `datasets/pose+utils.py`
- `model/network_image.py`
- `model/e2t_net.py`
- `model/pose_encoder.py`
- `preprocess/modules/pose_hrnet.py`
- `preprocess/jaad_data.py`
- `scripts/tools/inspect_jaad_npz.py`
- `scripts/tools/fix_jaad_pose_npz.py`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/train/video_0001.csv`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations.npz`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz`

## Current entry points

### Training entry
- Active baseline training entry: `train.py`
- Dataset construction happens in `train.py` via `eval('datasets.' + args.dataset)(...)`
- Baseline model construction happens inside `train.py` training path using `model.network_image`

### Dataset entry
- Dataset registration: `datasets/__init__.py`
- Active baseline dataset classes:
  - `datasets/jaad.py`
  - `datasets/pie.py`
  - `datasets/titan.py`
- For the requested JAAD path, the relevant dataset entry is `datasets/jaad.py`

### Model entry
- Active baseline model entry used by current training: `model/network_image.py`, class `PTINet`
- Prototype scaffold not used by current training:
  - `scripts/train_e2t_proto.py`
  - `model/e2t_net.py`

## HRNet-related files found
- `preprocess/modules/pose_hrnet.py`
- `scripts/tools/inspect_jaad_npz.py`
- `scripts/tools/fix_jaad_pose_npz.py`
- `datasets/pose+utils.py`
- `model/pose_encoder.py`
- `hrnet_crops/...`
- `hrnet_jaad_vis.jpg`
- `hrnet_pie_vis.jpg`

## Observed pipeline structure

### Baseline train path
`train.py`:
- parses args and `config.yml`
- builds train/val datasets
- passes dataset outputs through `DataLoader`
- consumes output dict keys such as `pos`, `speed`, `ped_attribute`, `ped_behavior`, `scene_attribute`, `image`, `optical`

### JAAD dataset output point
`datasets/jaad.py::__getitem__` returns a dict with:
- `image`
- `optical`
- `ped_attribute`
- `scene_attribute`
- `ped_behavior`
- `cross_label`
- `future_cross`
- `future_speed`
- `speed`
- `pos`
- `future_pos`
- `id`

This is the smallest safe integration point for adding:
- `pose`
- `pose_conf`

No train-loop refactor is required for Stage 0.

## Where offline pose should connect

### Recommended dataset-side load location
For JAAD, load offline pose in `datasets/jaad.py`:
- build or load a pose index once in `__init__`
- resolve each observation frame in `__getitem__`
- append `pose` and `pose_conf` to the returned sample dict

This is safer than modifying the model first because:
- the train path already consumes dataset dict outputs
- pose availability and missing-file fallback belong to the data adapter layer
- baseline behavior can remain unchanged when pose is disabled

### Likely lookup keys
The actual JAAD CSV and NPZ fields align on:
- `video_id`: derived from csv filename such as `video_0001.csv`
- `ped_id`: csv column `ID`, e.g. `pedestrian1`
- `frame`: csv column `frame`, 0-based

These are enough to build a stable lookup key:
- `(video_id, ped_id, frame)`

## Smallest safe integration path

### Dataset pose loading
1. Add a dedicated adapter utility under `datasets/` for JAAD pose NPZ loading.
2. Load `jaad_pose_annotations_fixed.npz` once when `use_pose` is enabled.
3. Convert global NPZ records into a fast index keyed by `(video_id, ped_id, frame)`.
4. In `__getitem__`, gather pose for the observed frames only.
5. If any frame is missing, return zero pose and zero confidence for that frame.

### Standardized tensor contract
- `pose`: `B x T x J x 2`
- `pose_conf`: `B x T x J`

Raw NPZ `(J x 3)` records should be normalized at adapter load time:
- first two channels -> pose xy
- third channel -> confidence

### Minimal pose encoder interface
Existing `model/pose_encoder.py` already matches the intended contract:
- input `pose`: `B x T x J x 2`
- input `pose_conf`: `B x T x J`
- output feature sequence: `B x T x D`

Stage 0 conclusion:
- keep this as the target interface
- do not wire it into `PTINet` yet

## Uncertain points
- `datasets/pose+utils.py` exists but is untracked and not integrated into the train path.
- `model/pose_encoder.py` exists but is also not integrated into the baseline model path.
- The active JAAD directory `PN_ego` does not contain the raw `images/` tree expected by `preprocess/jaad_data.py`; it contains split CSVs plus aggregated pose NPZ files.
- `imagefolderpath` values inside the CSV point to another machine path (`/media/meta/...`). This does not block pose integration but is a separate environment portability issue.

## Risks
- If future NPZ exports change joint count or key names, a hard-coded loader will break unless it supports candidate keys.
- If multiple pose files are introduced per split instead of one global NPZ, the indexing strategy must be adjusted.
- `ID` remapping inside `datasets/jaad.py` only affects in-memory training sequence IDs when building cached sequences; pose matching should use the original per-row CSV `ID` before such remapping assumptions leak into cache design.
