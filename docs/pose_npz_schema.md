# Pose NPZ Schema

## Purpose
Document the actual offline JAAD pose NPZ schema observed in local data and define the normalized adapter output expected by later stages.

## Files inspected
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations.npz`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz`
- `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/train/video_0001.csv`
- `scripts/tools/inspect_jaad_npz.py`
- `scripts/tools/fix_jaad_pose_npz.py`

## Actual NPZ files found
- `jaad_pose_annotations.npz`
- `jaad_pose_annotations_fixed.npz`

Both files were successfully opened.

## Actual keys observed
- `ped_ids`
- `ped_ptr`
- `video_id`
- `frame`
- `bbox`
- `keypoints`

No standalone confidence key was present.

## Actual shapes observed
- `ped_ids`: `(648,)`, `dtype=object`
- `ped_ptr`: `(649,)`, `dtype=int32`
- `video_id`: `(124354,)`, `dtype=object`
- `frame`: `(124354,)`, `dtype=int32`
- `bbox`: `(124354, 4)`, `dtype=float32`
- `keypoints`: `(124354, 17, 3)`, `dtype=float32`

## Meaning of fields

### `ped_ids`
One pedestrian track id per contiguous segment.

Examples:
- `pedestrian1`
- `pedestrian2`
- `pedestrian`

### `ped_ptr`
CSR-like segment pointer array.

For segment `i`:
- start index: `ped_ptr[i]`
- end index: `ped_ptr[i + 1]`

Observed invariants:
- `len(ped_ids) == 648`
- `len(ped_ptr) == 649`
- `ped_ptr[-1] == len(video_id) == 124354`
- every segment stayed within one `video_id`
- every segment had monotonic non-decreasing `frame`

### `video_id`
One video id per pose record.

Observed format:
- `video_0001`
- `video_0003`

### `frame`
One frame index per pose record.

Observed convention:
- 0-based frame indexing
- example head: `0, 1, 2, 3, 4`

This matches the CSV `frame` column, while csv `filename` is 1-based string naming such as `00001.png`.

### `bbox`
Per-record box array with shape `(4,)`.

Observed example:
- `[465., 730., 533., 848.]`

This looks like `xyxy`, not the `x,y,w,h` format used in the JAAD CSV training tables.
The bbox is useful as a consistency reference but should not be the primary join key.

### `keypoints`
Per-record pose array with shape `(17, 3)`.

Observed example from fixed file:
```text
[[753.375   526.5       0.90926236]
 [749.4167  528.0278    0.8623302 ]
 [751.3958  524.9722    0.9214775 ]]
```

Interpretation:
- channel 0: `x`
- channel 1: `y`
- channel 2: confidence score

The original file stored the first two coordinates swapped.
`scripts/tools/fix_jaad_pose_npz.py` rewrites only this orientation issue and preserves the rest of the schema.

## Missing-data observations
- `keypoints` contained no NaN in the inspected fixed file
- `nan_all_frames = 0`
- `nan_total = 0 / 6342054`

This should not be assumed universally for future exports; the adapter should still support missing rows and malformed records.

## CSV alignment observed
From `train/video_0001.csv`:
- `ID`: `pedestrian1`, `pedestrian2`
- `frame`: `0..568`
- `filename`: `00001.png`, `00002.png`, ...
- csv filename stem encodes the same sequence as `frame + 1`

This aligns with pose NPZ conventions:
- `video_id` <- split CSV basename without `.csv`
- `ped_id` <- csv `ID`
- `frame` <- csv `frame`

## Recommended normalized adapter output

### Standard output
- `pose`: `T x J x 2`
- `pose_conf`: `T x J`

### Conversion from raw NPZ
Given raw `keypoints[t]` with shape `J x 3`:
- `pose[t] = keypoints[t, :, :2]`
- `pose_conf[t] = keypoints[t, :, 2]`

### Batched training contract
- `pose`: `B x T x J x 2`
- `pose_conf`: `B x T x J`

## Robust adapter strategy
The loader should support these candidate cases even though the inspected JAAD file uses only one:
- `keypoints` in `N x J x 3`
- separate confidence array under keys such as `scores`, `confidence`, `conf`
- direct sample-local storage in `T x J x 3` or `T x J x 2`

For the current JAAD file, the safest path is:
1. load global aggregated NPZ
2. expand `ped_ptr` into per-record `ped_id`
3. build index on `(video_id, ped_id, frame)`
4. materialize `pose` and `pose_conf` for each dataset sequence window

## Remaining uncertainties
- Whether future JAAD exports will always use the fixed orientation file
- Whether other datasets will reuse the same aggregated NPZ format
- Whether any split-specific NPZ files will be introduced later
