"""
Reference only.
Goal:
- verify pose -> evidence -> belief contract on random tensors
"""

import torch

from codex_ref_pose_evidence_encoder import PoseEvidenceEncoder
from codex_ref_leaky_decision_accumulator import LeakyDecisionAccumulator


def main():
    pose = torch.randn(2, 5, 17, 2)
    pose_conf = torch.rand(2, 5, 17)

    encoder = PoseEvidenceEncoder(num_joints=17, out_dim=128)
    accumulator = LeakyDecisionAccumulator(feature_dim=128)

    pose_evidence_seq = encoder(pose, pose_conf)
    belief_seq = accumulator(pose_evidence_seq)
    belief_last = belief_seq[:, -1, :]

    print("pose:", tuple(pose.shape))
    print("pose_conf:", tuple(pose_conf.shape))
    print("pose_evidence_seq:", tuple(pose_evidence_seq.shape))
    print("belief_seq:", tuple(belief_seq.shape))
    print("belief_last:", tuple(belief_last.shape))
    print("belief_seq has NaN:", torch.isnan(belief_seq).any().item())
    print("belief_seq has Inf:", torch.isinf(belief_seq).any().item())


if __name__ == "__main__":
    main()
