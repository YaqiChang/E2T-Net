# PLAN.md

## Stage
Stage 4: pose evidence and decision accumulation

## Objective
Make this path work:

pose, pose_conf
-> PoseEvidenceEncoder
-> pose_evidence_seq
-> LeakyDecisionAccumulator
-> belief_seq
-> crossing head and intention head

## Scope
In scope:
- replace pose temporal encoder with frame-level evidence encoder
- add explicit leaky accumulator
- connect belief to crossing and intention
- keep motion and image encoders unchanged
- keep baseline compatible
- add smoke tests

Out of scope:
- replacing motion encoders
- replacing image encoders
- trajectory decoder redesign
- advanced SSM variants
- dataset contract changes

## Files
Modify:
- model/pose_encoder.py
- model/network_image.py
- train.py

Add:
- model/evidence_accumulator.py
- scripts/smoke_test_pose_evidence.py
- scripts/smoke_test_evidence_accumulator.py
- scripts/smoke_test_pose_belief_pipeline.py

## Interfaces
Dataset input:
- pose: B x T x J x 2
- pose_conf: B x T x J

Pose encoder output:
- pose_evidence_seq: B x T x D

Accumulator output:
- belief_seq: B x T x D
- belief_last: B x D

## Design
Task 1:
- rewrite pose_encoder.py
- keep normalization, confidence weighting, first-order difference, frame MLP
- remove temporal LSTM
- output pose_evidence_seq

Task 2:
- add LeakyDecisionAccumulator
- candidate_t = tanh(W_b * b_prev + W_e * e_t)
- leak_t = sigmoid(W_l * e_t)
- write_t = sigmoid(W_w * e_t)
- b_t = leak_t * b_prev + write_t * candidate_t

Task 3:
- in PTINet.forward:
  pose, pose_conf -> pose_evidence_seq -> belief_seq -> belief_last
- use belief_last for crossing and intention
- do not inject belief into speed decoder in this stage

Task 4:
- add args:
  --use_decision_accumulator
  --belief_dim
  --belief_readout

## Variants
- baseline: use_pose=False
- pose_direct_last: use_pose=True, no accumulator
- pose_accumulator: use_pose=True, accumulator enabled

## Acceptance
- baseline path still runs
- pose_evidence_seq shape correct
- belief_seq shape correct
- no NaN
- no Inf
- one forward smoke test passes
- one backward smoke test passes