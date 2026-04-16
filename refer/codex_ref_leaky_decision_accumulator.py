"""
Reference only.
Goal:
- show the target API for LeakyDecisionAccumulator
"""

import torch
import torch.nn as nn


class LeakyDecisionAccumulator(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.evidence_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.prev_state_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.leak_gate = nn.Linear(self.feature_dim, self.feature_dim)
        self.write_gate = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, evidence_seq):
        if evidence_seq.ndim != 3:
            raise ValueError(f"evidence_seq must have shape B x T x D, got {tuple(evidence_seq.shape)}.")
        if evidence_seq.shape[-1] != self.feature_dim:
            raise ValueError("feature dim mismatch.")

        evidence_seq = evidence_seq.float()
        batch_size, time_steps, _ = evidence_seq.shape
        belief_prev = torch.zeros(batch_size, self.feature_dim, device=evidence_seq.device, dtype=evidence_seq.dtype)
        belief_steps = []

        for t in range(time_steps):
            evidence_t = evidence_seq[:, t, :]
            candidate_t = torch.tanh(self.prev_state_proj(belief_prev) + self.evidence_proj(evidence_t))
            leak_t = torch.sigmoid(self.leak_gate(evidence_t))
            write_t = torch.sigmoid(self.write_gate(evidence_t))
            belief_t = leak_t * belief_prev + write_t * candidate_t
            belief_steps.append(belief_t.unsqueeze(1))
            belief_prev = belief_t

        belief_seq = torch.cat(belief_steps, dim=1)

        if torch.isnan(belief_seq).any():
            raise ValueError("belief_seq contains NaN values.")
        if torch.isinf(belief_seq).any():
            raise ValueError("belief_seq contains Inf values.")
        return belief_seq
