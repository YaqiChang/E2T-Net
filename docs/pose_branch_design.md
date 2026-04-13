
# Pose Branch Design

## Purpose

This document defines the pose branch used in the current E2T-Net pipeline.

The pose branch is a main input branch in the prediction pipeline.
It is not a training-only auxiliary signal.

Its role is to convert pedestrian pose observations into temporal pose features that can be aligned with other branches such as location and appearance.

## Pipeline Position

The full pose branch is defined as:

```text
pedestrian crop sequence -> PoseProvider -> pose, pose_conf -> PoseSequenceEncoder -> pose_feat_seq
````

This definition separates upstream pose estimation from downstream pose sequence encoding.

## Core Modules

### PoseProvider

PoseProvider is the upstream pose source.

Its job is to produce standardized pose observations for each input sequence.

Standard output contract:

* pose: T x J x 2
* pose_conf: T x J

where:

* T is the observation length
* J is the number of joints
* pose contains 2D joint coordinates
* pose_conf contains per-joint confidence scores

Current mode:

* cached offline pose loaded from aggregated NPZ files

Future mode:

* online pose estimation from pedestrian crop sequences

PoseProvider does not perform temporal encoding.
It only provides pose observations in a stable format.

### PoseSequenceEncoder

PoseSequenceEncoder is the downstream pose feature encoder.

Its job is to convert pose observations into temporal pose features.

Input contract:

* pose: B x T x J x 2
* pose_conf: B x T x J

Output contract:

* pose_feat_seq: B x T x D

where:

* B is the batch size
* D is the pose feature dimension

PoseSequenceEncoder is not a pose estimator.
It operates on pose coordinates and confidence scores that have already been produced by PoseProvider.

### Pose Fusion Interface

The pose branch is expected to provide one output to the later fusion stage:

* pose_feat_seq

Later fusion modules may combine:

* loc_feat_seq
* app_feat_seq
* pose_feat_seq

This document only defines the pose branch boundary.
It does not define the final fusion rule.

## Current Repository Mode

At the current stage, the repository uses cached offline pose.

The effective data path is:

```text
aggregated JAAD pose NPZ -> pose adapter -> pose, pose_conf -> PoseSequenceEncoder
```

This mode is used to decouple upstream pose estimation from downstream intention modeling.

The goal of the current stage is to verify that pose can be integrated as a stable branch with a clear interface.

## Future Online Mode

The final pipeline should also support online pose generation.

The intended online path is:

```text
pedestrian crop sequence -> online pose estimator -> pose, pose_conf -> PoseSequenceEncoder
```

The online estimator may be implemented with HRNet, Lite-HRNet, MMPose, or another top-down pose model.

The online and offline modes must share the same output contract:

* pose: T x J x 2
* pose_conf: T x J

This keeps the downstream encoder and fusion interface unchanged.

## Design Constraints

The pose branch must satisfy the following constraints.

### Stable Interface

The branch must keep a fixed tensor contract between provider and encoder.

### Narrow Responsibility

The provider generates pose observations.
The encoder converts pose observations into sequence features.
These two roles must not be mixed into one module.

### Temporal Alignment

The pose branch output must remain aligned with the same observation window used by the location branch and other branches.

### Replaceable Provider

The repository must be able to switch from cached pose mode to online pose mode without changing the downstream pose encoder interface.

## Out of Scope

The pose branch design in this document does not include:

* evidence accumulation
* final intention prediction
* trajectory decoding
* graph reasoning over pose skeletons
* online detector implementation
* end-to-end training of the pose estimator

## Minimal Implementation Target

The current implementation target is limited to the following components:

* a pose adapter that reads standardized pose from the offline NPZ source
* a PoseSequenceEncoder that maps pose and pose_conf to pose_feat_seq
* a future-ready provider abstraction that allows cached and online modes to share one interface

## Summary

The pose branch should be understood as a two-level structure:

```text
PoseProvider -> PoseSequenceEncoder -> pose_feat_seq
```

Current experiments use cached offline pose to validate the downstream branch design.
The final system can replace the provider with an online pose estimator while keeping the same downstream interface.

````

### `docs/pose_encoder.md`

```markdown
# PoseSequenceEncoder

## Purpose

This document defines the local design of PoseSequenceEncoder.

PoseSequenceEncoder is a downstream sequence encoder.
It is not a pose estimator.

Its role is to convert pose coordinates and pose confidence into temporal pose features for later fusion and prediction.

## Input and Output

Input:

- pose: B x T x J x 2
- pose_conf: B x T x J

Output:

- pose_feat_seq: B x T x D

where:

- B is the batch size
- T is the observation length
- J is the number of joints
- D is the encoded pose feature dimension

Default joint count for JAAD:

- J = 17

## Design Goal

The encoder should provide stable and compact temporal pose features while keeping the module lightweight and clearly scoped.

The encoder should capture:

- normalized body geometry
- short-term pose motion
- confidence-aware reliability

The encoder should not absorb responsibilities that belong to later modules such as intention accumulation.

## Module Structure

A minimal first version should contain the following parts.

### Frame Normalization

Each frame should be normalized using a body-centered reference and a stable body scale.

Recommended rule:

- root from hip center
- scale from torso length or shoulder-to-hip distance

This reduces variation caused by translation and scale changes.

### Confidence-Aware Feature Construction

Confidence should directly affect pose feature construction.

Recommended rule:

- use pose_conf to suppress unreliable joints during coordinate and motion feature construction

This prevents low-quality joints from contributing equally to the encoded representation.

### Short-Term Motion Feature

The encoder should include explicit pose motion.

Recommended rule:

- first-order difference of normalized joints across time

This makes the encoder sensitive to pre-motion changes rather than only static posture.

### Frame Feature Mapping

For each frame, construct a feature vector from:

- normalized joints
- normalized joint velocity
- joint confidence

Then apply a small frame-level MLP.

### Temporal Encoding

Apply a lightweight temporal encoder after frame-level mapping.

Recommended first version:

- one small temporal convolution stack
- or one-layer LSTM

The temporal encoder at this stage is only for local sequence modeling.
It is not the decision-state accumulator.

## Recommended Minimal Feature Template

Per-frame feature input:

```text
normalized joints + normalized joint velocity + joint confidence
````

Per-sequence output:

```text
pose_feat_seq
```

## What This Module Should Not Do

PoseSequenceEncoder should not include:

* online pose estimation
* HRNet inference
* heatmap decoding
* graph convolution reasoning
* evidence accumulation
* final intention prediction
* trajectory prediction

These functions belong to other modules.

## Interface Constraint

PoseSequenceEncoder should remain a standalone module.

It must be possible to test it independently with:

* random pose tensors
* real JAAD pose batches

It should not require changes to baseline prediction heads at the current stage.

## Validation Requirement

A valid implementation should pass the following checks.

### Static Check

* import succeeds
* syntax is valid

### Random Forward Check

* random pose input produces output with shape B x T x D
* no NaN
* no Inf

### Real Batch Check

* one real JAAD batch pose tensor can run through forward
* output shape remains B x T x D
* no NaN
* no Inf

## Summary

PoseSequenceEncoder is a local sequence encoder with a narrow role:

```text
pose, pose_conf -> pose_feat_seq
```

It should encode normalized geometry, short-term motion, and confidence-aware reliability.
It should remain lightweight and independent from later decision modules.

```
