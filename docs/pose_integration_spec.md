# Pose Integration Spec

## Purpose
Define the smallest safe implementation path for offline HRNet pose integration after Stage 0, while keeping baseline behavior unchanged when pose is disabled.

## Files inspected
- `train.py`
- `datasets/__init__.py`
- `datasets/jaad.py`
- `datasets/pie.py`
- `datasets/pose+utils.py`
- `model/network_image.py`
- `model/pose_encoder.py`
- `scripts/tools/inspect_jaad_npz.py`
- `scripts/tools/fix_jaad_pose_npz.py`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/train/video_0001.csv`

## Current entry points

### Train entry
- `train.py`

### Dataset entry
- `datasets/jaad.py` for JAAD
- dataset registration through `datasets/__init__.py`

### Model entry
- baseline active model: `model/network_image.py`, class `PTINet`
- future optional pose encoder file already present: `model/pose_encoder.py`

## Standardized interface

### Dataset output contract
When pose is enabled, each sample should additionally expose:
- `pose`
- `pose_conf`

Existing output keys must remain unchanged.

### Shape contract
- `pose`: `B x T x J x 2`
- `pose_conf`: `B x T x J`
- `pose_feat_seq`: `B x T x D`

For a single sample before collation:
- `pose`: `T x J x 2`
- `pose_conf`: `T x J`

## Proposed minimal integration path

### Step 1: dataset-side adapter
Add or cleanly replace the current pose utility with a JAAD-focused adapter that:
- loads `jaad_pose_annotations_fixed.npz`
- expands `ped_ptr` to per-record `ped_id`
- builds a dictionary keyed by `(video_id, ped_id, frame)`
- returns `pose` and `pose_conf` in standardized shape

Recommended adapter responsibilities:
- key normalization
- confidence extraction from `keypoints[..., 2]`
- zero fallback for missing frames
- optional missing-count statistics

### Step 2: JAAD dataset integration
Update `datasets/jaad.py` only where needed:
- add optional config inputs such as `use_pose`, `pose_dir`, `pose_format`
- initialize pose adapter only when `use_pose=True`
- derive `video_id` from the source csv filename or cached sequence metadata
- gather pose for observed frames in `__getitem__`
- append:
  - `pose`
  - `pose_conf`

This keeps the baseline path unchanged when pose is disabled.

### Step 3: minimal encoder interface
Use the existing `model/pose_encoder.py` interface as the target:
- input `pose`
- input `pose_conf`
- output `pose_feat_seq`

No model behavior change should happen until later stages.

## Why this path is the smallest safe one
- No broad refactor of `train.py`
- No immediate change to `PTINet.forward`
- No change to baseline outputs when pose is disabled
- Pose schema normalization stays in the adapter layer, where dataset-specific quirks belong
- Missing pose handling is isolated to one place

## Recommended config flags for later stages

### Required
- `use_pose`
- `pose_dir`
- `pose_format`
- `use_pose_encoder`

### Optional
- `pose_missing_policy`
- `pose_num_joints`
- `pose_input_dim`

## Adapter assumptions
- Current JAAD pose source is one aggregated NPZ file, not per-sample files.
- The preferred file is `jaad_pose_annotations_fixed.npz`.
- Join key should be `(video_id, ped_id, frame)`.
- Raw `keypoints` is `J x 3` with `x, y, conf`.

## Baseline compatibility requirements
- If `use_pose=False`, dataset outputs and model behavior must remain unchanged.
- Existing configs must continue to run.
- Missing or unreadable pose rows must not crash the pipeline by default.
- Missing pose frame fallback:
  - `pose` -> zeros with shape `J x 2`
  - `pose_conf` -> zeros with shape `J`

## Validation plan for later implementation stages

### Level 1
- import adapter successfully
- no syntax errors
- no unresolved references

### Level 2
- one JAAD sample returns `pose` and `pose_conf`
- one batch collates successfully
- shapes match:
  - `pose`: `B x T x J x 2`
  - `pose_conf`: `B x T x J`

### Level 3
- baseline still runs with `use_pose=False`
- pose-enabled path reaches one forward pass

## Known risks
- Cached sequence files may omit enough source metadata if pose lookup is added after caching; if so, cached rows need a stable `video_id` field preserved.
- CSV `ID` remapping in `datasets/jaad.py` must not corrupt matching against original JAAD pose ids.
- `imagefolderpath` currently contains machine-specific absolute paths and should not be used as the primary pose join key.
