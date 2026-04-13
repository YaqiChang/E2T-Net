# Pose Dataset Stage 1

## Purpose
Implement JAAD dataset-side loading for offline HRNet pose data from the aggregated fixed NPZ file and expose standardized pose tensors in each sample.

## Files modified
- `datasets/pose_utils.py`
- `datasets/jaad.py`
- `docs/pose_dataset_stage1.md`

## Matching keys used
The pose join key is:
- `video_id`
- `ped_id`
- `frame`

Specifically:
- `video_id` comes from the JAAD split csv basename, e.g. `video_0001.csv -> video_0001`
- `ped_id` comes from the original per-row csv identity before `datasets/jaad.py` remaps IDs
- `frame` comes from the original JAAD csv `frame` column

To preserve these keys after sequence construction, each cached JAAD sample now retains:
- `source_video_id`
- `source_ped_id`
- `obs_frame_sequence`

## Standardized output shapes
When `use_pose=True`:
- `pose`: `T x 17 x 2`
- `pose_conf`: `T x 17`

After dataloader collation:
- `pose`: `B x T x 17 x 2`
- `pose_conf`: `B x T x 17`

## Pose source
Current JAAD pose source:
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz`

Observed schema used by the adapter:
- `ped_ids`
- `ped_ptr`
- `video_id`
- `frame`
- `keypoints`

The adapter interprets `keypoints[..., :2]` as pose xy and `keypoints[..., 2]` as confidence.

## Missing-pose fallback policy
If a frame is not found in the aggregated NPZ index:
- `pose[t]` falls back to zeros with shape `17 x 2`
- `pose_conf[t]` falls back to zeros with shape `17`

The adapter tracks missing counts lightly through:
- `missing_frames`
- `total_frames`
- `missing_sequences`

## Commands run
Environment checks:
```bash
/bin/bash -c 'source /home/cyq/anaconda3/bin/activate py38 && which python && python -V && python -c "import torch; print(torch.__version__)" && python -c "import torchvision; print(torchvision.__version__)" && python -c "import cv2; print(cv2.__version__)"'
```

Static checks:
```bash
/home/cyq/anaconda3/envs/py38/bin/python -m py_compile datasets/jaad.py datasets/pose_utils.py
```

Stage 1 smoke tests:
```bash
/home/cyq/anaconda3/envs/py38/bin/python - <<'PY'
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.jaad import JAAD

root = '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'
out_dir = '/tmp/e2t_stage1_jaad'
Path(out_dir).mkdir(parents=True, exist_ok=True)

common_kwargs = dict(
    data_dir=root,
    out_dir=out_dir,
    dtype='train',
    input=16,
    output=32,
    stride=16,
    skip=1,
    use_images=False,
    use_attribute=False,
    use_opticalflow=False,
)

pose_ds = JAAD(from_file=False, save=True, use_pose=True,
               pose_file=str(Path(root) / 'jaad_pose_annotations_fixed.npz'),
               **common_kwargs)
sample = pose_ds[0]
loader = DataLoader(pose_ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))

baseline_ds = JAAD(from_file=True, save=False, use_pose=False, **common_kwargs)
baseline_sample = baseline_ds[0]
PY
```

## Observed smoke-test results
- one-sample JAAD pose path succeeded
- one-batch dataloader pose path succeeded
- baseline no-pose path succeeded using cached sequence file
- observed sample shape:
  - `pose`: `(16, 17, 2)`
  - `pose_conf`: `(16, 17)`
- observed batch shape:
  - `pose`: `(2, 16, 17, 2)`
  - `pose_conf`: `(2, 16, 17)`
- no NaN observed in sample or batch pose tensors
- baseline fields remained present

## Notes
- This stage does not modify model behavior.
- This stage does not add pose handling to PIE.
- This stage keeps baseline output fields unchanged and only appends `pose` and `pose_conf` when `use_pose=True`.
