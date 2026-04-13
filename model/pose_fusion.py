import torch
import torch.nn as nn


class BranchFeatureFusion(nn.Module):
    """Minimal branch-level fusion interface for aligned feature sequences.

    Input branch names:
    - loc_feat_seq
    - app_feat_seq
    - pose_feat_seq

    Expected input shape for each provided branch:
    - B x T x D_i

    Output:
    - fused_feat_seq: B x T x D_out
    """

    def __init__(self, branch_dims, out_dim=128, use_projection=True):
        super().__init__()
        if not branch_dims:
            raise ValueError('branch_dims must not be empty.')

        self.branch_dims = dict(branch_dims)
        self.branch_names = list(self.branch_dims.keys())
        self.concat_dim = int(sum(self.branch_dims.values()))
        self.out_dim = int(out_dim)
        self.use_projection = bool(use_projection)

        if self.use_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.concat_dim, self.out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            if self.concat_dim != self.out_dim:
                raise ValueError(
                    'concat_dim must equal out_dim when use_projection=False, '
                    f'got concat_dim={self.concat_dim}, out_dim={self.out_dim}.'
                )
            self.proj = nn.Identity()

    def forward(self, branch_features):
        if not isinstance(branch_features, dict):
            raise TypeError('branch_features must be a dict of branch_name -> tensor.')

        provided = []
        batch_size = None
        time_steps = None
        device = None
        dtype = None

        for name in self.branch_names:
            feat = branch_features.get(name)
            if feat is None:
                continue
            if not isinstance(feat, torch.Tensor):
                raise TypeError(f'{name} must be a torch.Tensor when provided.')
            if feat.ndim != 3:
                raise ValueError(f'{name} must have shape B x T x D, got {tuple(feat.shape)}.')
            if feat.shape[-1] != self.branch_dims[name]:
                raise ValueError(
                    f'{name} feature dim mismatch: expected {self.branch_dims[name]}, got {feat.shape[-1]}.'
                )

            if batch_size is None:
                batch_size, time_steps = feat.shape[:2]
                device = feat.device
                dtype = feat.dtype
            else:
                if feat.shape[0] != batch_size or feat.shape[1] != time_steps:
                    raise ValueError(
                        'All provided branches must align on batch and time dimensions, '
                        f'got {name}={tuple(feat.shape)} with expected B={batch_size}, T={time_steps}.'
                    )
            provided.append(feat)

        if not provided:
            raise ValueError('At least one branch feature tensor must be provided.')

        concat_parts = []
        for name in self.branch_names:
            feat = branch_features.get(name)
            if feat is None:
                zeros = torch.zeros(
                    batch_size,
                    time_steps,
                    self.branch_dims[name],
                    device=device,
                    dtype=dtype,
                )
                concat_parts.append(zeros)
            else:
                concat_parts.append(feat)

        fused = torch.cat(concat_parts, dim=-1)
        fused = self.proj(fused)

        if torch.isnan(fused).any():
            raise ValueError('fused_feat_seq contains NaN values.')
        if torch.isinf(fused).any():
            raise ValueError('fused_feat_seq contains Inf values.')

        return fused
