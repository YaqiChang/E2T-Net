import torch
import torch.nn as nn


class LeakyDecisionAccumulator(nn.Module):
    """Accumulate instantaneous evidence into latent belief states.

    Input:
    - evidence_seq: B x T x D

    Output:
    - belief_seq: B x T x D
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)

        self.evidence_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.prev_state_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.leak_gate = nn.Linear(self.feature_dim, self.feature_dim)
        self.write_gate = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, evidence_seq):
        self._check_inputs(evidence_seq)

        evidence_seq = evidence_seq.float()
        self._check_finite(evidence_seq, 'evidence_seq')

        batch_size, time_steps, _ = evidence_seq.shape
        belief_prev = torch.zeros(
            batch_size,
            self.feature_dim,
            device=evidence_seq.device,
            dtype=evidence_seq.dtype,
        )
        belief_steps = []

        for t in range(time_steps):
            evidence_t = evidence_seq[:, t, :]
            candidate_t = torch.tanh(
                self.prev_state_proj(belief_prev) + self.evidence_proj(evidence_t)
            )
            leak_t = torch.sigmoid(self.leak_gate(evidence_t))
            write_t = torch.sigmoid(self.write_gate(evidence_t))
            belief_t = leak_t * belief_prev + write_t * candidate_t

            self._check_finite(candidate_t, f'candidate_t[{t}]')
            self._check_finite(leak_t, f'leak_t[{t}]')
            self._check_finite(write_t, f'write_t[{t}]')
            self._check_finite(belief_t, f'belief_t[{t}]')

            belief_steps.append(belief_t.unsqueeze(1))
            belief_prev = belief_t

        belief_seq = torch.cat(belief_steps, dim=1)
        self._check_finite(belief_seq, 'belief_seq')
        return belief_seq

    def _check_inputs(self, evidence_seq):
        if not isinstance(evidence_seq, torch.Tensor):
            raise TypeError('evidence_seq must be a torch.Tensor.')
        if evidence_seq.ndim != 3:
            raise ValueError(
                f'evidence_seq must have shape B x T x D, got {tuple(evidence_seq.shape)}.'
            )
        if evidence_seq.shape[-1] != self.feature_dim:
            raise ValueError(
                f'feature dim mismatch: expected {self.feature_dim}, got {evidence_seq.shape[-1]}.'
            )

    def _check_finite(self, tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f'{name} contains NaN values.')
        if torch.isinf(tensor).any():
            raise ValueError(f'{name} contains Inf values.')


def _smoke_test():
    torch.manual_seed(0)

    model = LeakyDecisionAccumulator(feature_dim=128)
    evidence_seq = torch.randn(2, 5, 128)

    with torch.no_grad():
        belief_seq = model(evidence_seq)

    belief_last = belief_seq[:, -1, :]

    assert belief_seq.shape == (2, 5, 128), (
        f'unexpected belief_seq shape: {tuple(belief_seq.shape)}'
    )
    assert belief_last.shape == (2, 128), (
        f'unexpected belief_last shape: {tuple(belief_last.shape)}'
    )
    assert not torch.isnan(belief_seq).any(), 'belief_seq contains NaN.'
    assert not torch.isinf(belief_seq).any(), 'belief_seq contains Inf.'

    print(f'belief_seq shape: {tuple(belief_seq.shape)}')
    print(f'belief_last shape: {tuple(belief_last.shape)}')
    print(f'has_nan: {bool(torch.isnan(belief_seq).any())}')
    print(f'has_inf: {bool(torch.isinf(belief_seq).any())}')


if __name__ == '__main__':
    _smoke_test()
