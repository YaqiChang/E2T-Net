import torch
import torch.nn as nn


class PoseEvidenceEncoder(nn.Module):
    """Encode per-frame pose evidence without temporal accumulation.

    Input:
    - pose: B x T x J x 2
    - pose_conf: B x T x J

    Output:
    - pose_evidence_seq: B x T x D
    """

    def __init__(
        self,
        num_joints=17,
        frame_hidden_dim=128,
        temporal_hidden_dim=128,
        out_dim=128,
        conf_power=1.0,
        eps=1e-6,
    ):
        super().__init__()

        self.num_joints = int(num_joints)
        self.frame_hidden_dim = int(frame_hidden_dim)
        # Kept only to avoid breaking existing config construction paths.
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.out_dim = int(out_dim)
        self.conf_power = float(conf_power)
        self.eps = float(eps)

        # Per-frame feature:
        # normalized coordinates -> J * 2
        # first-order motion     -> J * 2
        # confidence             -> J
        frame_input_dim = self.num_joints * 5

        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_input_dim, self.frame_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.frame_hidden_dim, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pose, pose_conf):
        self._check_inputs(pose, pose_conf)

        pose = pose.float()
        pose_conf = pose_conf.float()
        self._check_finite(pose, 'pose')
        self._check_finite(pose_conf, 'pose_conf')

        pose_norm = self._body_center_normalize(pose, pose_conf)
        self._check_finite(pose_norm, 'pose_norm')

        conf = torch.clamp(pose_conf, min=0.0, max=1.0).pow(self.conf_power)
        self._check_finite(conf, 'conf')

        # pose_norm / pose_weighted / pose_velocity all follow B x T x J x 2.
        conf_expanded = conf.unsqueeze(-1)
        pose_weighted = pose_norm * conf_expanded
        self._check_finite(pose_weighted, 'pose_weighted')

        pose_velocity = self._first_order_difference(pose_weighted)
        self._check_finite(pose_velocity, 'pose_velocity')

        frame_feat = torch.cat(
            [
                pose_weighted.reshape(pose.shape[0], pose.shape[1], self.num_joints * 2),
                pose_velocity.reshape(pose.shape[0], pose.shape[1], self.num_joints * 2),
                conf,
            ],
            dim=-1,
        )
        self._check_finite(frame_feat, 'frame_feat')

        pose_evidence_seq = self.frame_mlp(frame_feat)
        self._check_finite(pose_evidence_seq, 'pose_evidence_seq')

        return pose_evidence_seq

    def _check_inputs(self, pose, pose_conf):
        if not isinstance(pose, torch.Tensor):
            raise TypeError('pose must be a torch.Tensor.')
        if not isinstance(pose_conf, torch.Tensor):
            raise TypeError('pose_conf must be a torch.Tensor.')

        if pose.ndim != 4:
            raise ValueError(f'pose must have shape B x T x J x 2, got {tuple(pose.shape)}.')
        if pose_conf.ndim != 3:
            raise ValueError(f'pose_conf must have shape B x T x J, got {tuple(pose_conf.shape)}.')

        batch_size, time_steps, num_joints, channels = pose.shape
        conf_batch, conf_time, conf_joints = pose_conf.shape

        if channels != 2:
            raise ValueError(f'pose last dimension must be 2, got {channels}.')
        if num_joints != self.num_joints:
            raise ValueError(
                f'pose joint dimension mismatch: expected {self.num_joints}, got {num_joints}.'
            )
        if conf_joints != self.num_joints:
            raise ValueError(
                f'pose_conf joint dimension mismatch: expected {self.num_joints}, got {conf_joints}.'
            )
        if batch_size != conf_batch or time_steps != conf_time:
            raise ValueError(
                'pose and pose_conf batch/time mismatch: '
                f'pose={tuple(pose.shape)}, pose_conf={tuple(pose_conf.shape)}.'
            )

    def _check_finite(self, tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f'{name} contains NaN values.')
        if torch.isinf(tensor).any():
            raise ValueError(f'{name} contains Inf values.')

    def _body_center_normalize(self, pose, pose_conf):
        # COCO-17 indexing used by JAAD HRNet export.
        left_shoulder = 5
        right_shoulder = 6
        left_hip = 11
        right_hip = 12

        conf = torch.clamp(pose_conf, min=0.0, max=1.0).unsqueeze(-1)

        shoulder_weights = conf[:, :, [left_shoulder, right_shoulder], :]
        hip_weights = conf[:, :, [left_hip, right_hip], :]

        shoulder_center = (
            pose[:, :, [left_shoulder, right_shoulder], :] * shoulder_weights
        ).sum(dim=2) / torch.clamp(shoulder_weights.sum(dim=2), min=self.eps)

        hip_center = (
            pose[:, :, [left_hip, right_hip], :] * hip_weights
        ).sum(dim=2) / torch.clamp(hip_weights.sum(dim=2), min=self.eps)

        root = hip_center.unsqueeze(2)
        torso_scale = torch.norm(shoulder_center - hip_center, dim=-1, keepdim=True)
        torso_scale = torch.clamp(torso_scale, min=self.eps).unsqueeze(2)

        return (pose - root) / torso_scale

    def _first_order_difference(self, pose_feat):
        motion = torch.zeros_like(pose_feat)
        motion[:, 1:] = pose_feat[:, 1:] - pose_feat[:, :-1]
        return motion


PoseSequenceEncoder = PoseEvidenceEncoder


def _smoke_test():
    torch.manual_seed(0)

    model = PoseEvidenceEncoder(num_joints=17, frame_hidden_dim=128, out_dim=128)
    pose = torch.randn(2, 5, 17, 2)
    pose_conf = torch.rand(2, 5, 17)

    with torch.no_grad():
        pose_evidence_seq = model(pose, pose_conf)

    assert pose_evidence_seq.shape == (2, 5, 128), (
        f'unexpected pose_evidence_seq shape: {tuple(pose_evidence_seq.shape)}'
    )
    assert not torch.isnan(pose_evidence_seq).any(), 'pose_evidence_seq contains NaN.'
    assert not torch.isinf(pose_evidence_seq).any(), 'pose_evidence_seq contains Inf.'

    print(f'pose_evidence_seq shape: {tuple(pose_evidence_seq.shape)}')
    print(f'has_nan: {bool(torch.isnan(pose_evidence_seq).any())}')
    print(f'has_inf: {bool(torch.isinf(pose_evidence_seq).any())}')


if __name__ == '__main__':
    _smoke_test()
