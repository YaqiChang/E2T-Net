# PLAN.md

## Current Stage
Pose data integration for offline HRNet pose results

## Objective
Integrate offline human pose NPZ files into the current E2T-Net codebase in a minimal, modular, and testable way.

The immediate target is to make this path work correctly:

dataset -> dataloader -> model forward

This stage is only for interface completion and feasibility verification.

---

## In Scope

1. Inspect the repository root for HRNet-related code and use it as a reference for pose output conventions.
2. Add robust NPZ pose loading.
3. Integrate pose loading into dataset classes.
4. Add pose and pose_conf to dataset outputs.
5. Add a minimal pose encoder template.
6. Wire pose inputs into the model interface behind config flags.
7. Add smoke tests for one sample, one batch, and one forward pass.

---

## Out of Scope

1. Evidence accumulation
2. SSM-based temporal modeling
3. Belief-conditioned trajectory decoding
4. New loss design
5. Pose graph network
6. Full pose-driven fusion redesign
7. Performance tuning
8. Large training infrastructure refactor

---

## Current Repository Understanding

The current repository already supports:
- bbox position
- speed
- pedestrian attributes
- scene attributes
- image features
- optical flow features

The current repository does not yet reliably support:
- offline pose NPZ loading
- pose confidence loading
- pose encoder
- pose-aware forward interface

---

## Deliverables

### Stage 0
Repository and pose schema audit

Files:
- docs/pose_integration_audit.md
- docs/pose_npz_schema.md
- docs/pose_integration_spec.md

Acceptance:
- train entry identified
- dataset entry identified
- model entry identified
- HRNet-related code location identified
- likely NPZ schema documented
- uncertain schema cases documented with a robust adapter strategy

### Stage 1
Dataset-side pose loading

Files:
- datasets/pose_utils.py
- updates to dataset classes used by training

Acceptance:
- dataset returns pose
- dataset returns pose_conf
- missing pose files handled safely
- one sample shape check passes

### Stage 2
Minimal pose encoder template

Files:
- models/pose_encoder.py

Acceptance:
- encoder accepts pose and pose_conf
- encoder returns pose_feat_seq
- one forward-pass sanity test passes

### Stage 3
Safe model wiring

Files:
- updates to model interface
- config flags for pose integration

Acceptance:
- model forward accepts pose and pose_conf
- baseline behavior unchanged when pose is disabled
- one batch forward pass succeeds with pose enabled

---

## Standard Data Contract

### Dataset output
When pose is enabled, each sample should expose:
- pose
- pose_conf

Existing fields must remain unchanged.

### Standardized tensor shapes
- pose: B x T x J x 2
- pose_conf: B x T x J
- pose_feat_seq: B x T x D

If raw pose files use another shape, normalize them in the adapter layer.

---

## Expected Config Flags

Required:
- use_pose
- pose_dir
- pose_format
- use_pose_encoder

Optional if needed:
- pose_missing_policy
- pose_num_joints
- pose_input_dim

---

## Recommended Implementation Order

### Step 1
Audit repository and infer pose file conventions

Tasks:
- inspect current train path
- inspect dataset path
- inspect model path
- inspect HRNet-related code in repo root
- inspect available NPZ examples if present

### Step 2
Implement pose adapter

Tasks:
- load NPZ files
- support common candidate keys
- normalize outputs into pose and pose_conf
- add safe fallback for missing data

### Step 3
Integrate dataset outputs

Tasks:
- connect pose adapter to dataset classes
- return pose and pose_conf in __getitem__
- keep existing outputs unchanged

### Step 4
Implement minimal pose encoder

Tasks:
- center-normalize joints
- flatten joints and confidence
- encode per-frame pose
- produce pose_feat_seq

### Step 5
Wire pose into model interface

Tasks:
- pass pose and pose_conf through train path
- extend model forward signature
- keep baseline path intact
- only compute pose features when enabled

### Step 6
Run smoke tests

Tasks:
- dataset one-sample test
- dataloader one-batch test
- model one-forward test
- optional one-iteration train start test

---

## Smoke Test Checklist

Before marking a step complete, verify:

1. one sample can be loaded
2. pose shape is correct
3. pose_conf shape is correct
4. one batch can be collated
5. one forward pass runs
6. baseline still runs when pose is disabled
7. pose-enabled path does not produce NaN
8. exact commands and results are documented

---

## Known Risks To Keep Visible

1. Dataset split usage must be checked before trusting ablations.
2. Config parsing types must be checked before trusting training behavior.
3. NPZ schema may vary across files and should not be hard-coded without inspection.
4. Missing pose files must be handled deterministically.

---

## Definition of Done

This stage is done only when:

1. pose NPZ files are loadable through the dataset pipeline
2. pose and pose_conf appear in batch data
3. the model forward path accepts pose inputs safely
4. the baseline still works when pose is disabled
5. smoke tests have been run and reported
6. assumptions and risks have been documented