
# PLAN.md

## Current Stage

Stage 3: safe model wiring

## Objective

Integrate the existing pose encoder into the current E2T-Net training and model path with minimal changes and strict backward compatibility.

The current goal is:

```text
train path -> pose, pose_conf -> model forward -> pose_feat_seq
````

This stage only establishes safe data and forward-path connectivity.
It does not redesign fusion or prediction logic.

## Current Status

Completed:

* Stage 0: repository audit and pose NPZ schema check
* Stage 1: JAAD dataset-side pose integration
* Stage 2: standalone pose encoder implementation and validation

Current repository state:

* JAAD can return `pose` and `pose_conf` when `use_pose=True`
* cached offline pose is available through the dataset path
* `model/pose_encoder.py` exists
* random input forward smoke test for `PoseSequenceEncoder` passed
* real JAAD batch forward smoke test for `PoseSequenceEncoder` passed
* output shape is `B x T x D`
* no NaN
* no Inf
* baseline model behavior is still preserved when pose is disabled at dataset stage

## Scope of the Current Stage

In scope:

* add pose-related runtime args to `train.py`
* pass pose-related args into JAAD dataset construction
* extend `PTINet` to accept optional `pose` and `pose_conf`
* instantiate `PoseSequenceEncoder` inside the model when `use_pose=True`
* run pose encoding inside model forward when enabled
* preserve baseline behavior when `use_pose=False`
* validate one-batch forward and one-step backward with pose enabled

Out of scope:

* fusion redesign
* evidence accumulation
* SSM-based temporal modeling
* trajectory decoder changes
* loss redesign
* graph pose reasoning
* PIE extension
* online pose mode
* large refactor

## Deliverable of the Current Stage

Required modified files:

* `train.py`
* `model/network_image.py`

Optional helper file if needed:

* `smoke_test_pose_wiring.py`

## Standard Data Contract

Dataset-side input to model:

* `pose: B x T x J x 2`
* `pose_conf: B x T x J`

Encoder output inside model:

* `pose_feat_seq: B x T x D`

Default for JAAD:

* `J = 17`

## Stage 3 Design Constraints

### Minimal Change

Only add the smallest set of changes needed to route pose tensors from dataset to model forward.

### Backward Compatibility

When `use_pose=False`, training and validation behavior must remain unchanged.

### Optional Interface

`PTINet.forward` must accept optional pose inputs without forcing pose usage in all paths.

### Local Responsibility

This stage only makes pose features available inside the model.
It does not define the final multimodal fusion rule.

### Stable Toggle

Pose support must be controlled by explicit runtime arguments rather than hard-coded behavior.

## Required Runtime Arguments

The training entry should support:

* `--use_pose`
* `--pose_file`
* `--pose_format`

These arguments must be propagated into dataset construction and model configuration.

## Minimal Model Requirement

The first wiring version should remain lightweight.

Required behavior:

* import `PoseSequenceEncoder`
* create `self.pose_encoder` only when pose is enabled
* allow `forward` to receive `pose=None` and `pose_conf=None`
* run `pose_feat_seq = self.pose_encoder(pose, pose_conf)` when enabled
* keep pose features available for later integration without redesigning current decoders in this stage

Recommended handling for this stage:

* compute `pose_feat_seq`
* keep it as an internal intermediate
* do not yet redesign the current multimodal fusion rule unless strictly needed for tensor compatibility verification

## Acceptance for Stage 3

Stage 3 is done only if all of the following are satisfied:

1. `train.py` accepts `--use_pose`, `--pose_file`, and `--pose_format`
2. pose-related args are passed into JAAD dataset construction
3. `PTINet.forward` accepts optional `pose` and `pose_conf`
4. `PoseSequenceEncoder` is instantiated safely inside the model when enabled
5. `use_pose=False` preserves original forward behavior
6. `use_pose=True` one real JAAD batch forward succeeds
7. `use_pose=True` one training-step backward succeeds
8. no NaN
9. no Inf
10. no unrelated baseline path is broken

## Validation Checklist

Required checks:

* static syntax check
* import check
* one-batch forward smoke test with `use_pose=False`
* one-batch forward smoke test with `use_pose=True`
* one-step backward smoke test with `use_pose=True`

Each report must include:

* files modified
* exact commands run
* observed tensor shapes
* whether NaN or Inf occurred
* whether `use_pose=False` remained unchanged
* remaining risks

## Next Stages

### Stage 4

Pose branch fusion interface

Goal:

* define how `pose_feat_seq` interacts with existing branches
* define clean fusion input and output interfaces

### Stage 5

Main pipeline integration

Goal:

* integrate pose as one branch of the full prediction pipeline

### Stage 6

Online pose mode

Goal:

* replace cached pose source with an online pose provider while keeping the same downstream interface

## Notes

Current experiments are still allowed to use cached offline pose.

The final system should support online pose generation through a separate provider module.

Only the provider should change between offline mode and online mode.

The downstream pose encoder interface should remain unchanged.

