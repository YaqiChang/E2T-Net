import torch
import torch.nn as nn


class PoseSequenceEncoder(nn.Module):
    """Standalone Stage-2 pose sequence encoder.

    Input:
    - pose: B x T x J x 2
    - pose_conf: B x T x J

    Output:
    - pose_feat_seq: B x T x D
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

        self.num_joints = num_joints
        self.frame_hidden_dim = frame_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.out_dim = out_dim
        self.conf_power = conf_power
        self.eps = eps

        # Per-frame feature:
        # normalized coordinates -> J * 2
        # first-order motion     -> J * 2
        # confidence             -> J
        frame_input_dim = num_joints * 5

        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_input_dim, frame_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(frame_hidden_dim, frame_hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.temporal_encoder = nn.LSTM(
            input_size=frame_hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_proj = nn.Linear(temporal_hidden_dim, out_dim)

    def forward(self, pose, pose_conf):
        self._check_inputs(pose, pose_conf)

        pose = pose.float()
        pose_conf = pose_conf.float()

        pose_norm = self._body_center_normalize(pose, pose_conf)

        conf = torch.clamp(pose_conf, min=0.0, max=1.0).pow(self.conf_power)
        conf_expanded = conf.unsqueeze(-1)

        pose_weighted = pose_norm * conf_expanded
        pose_velocity = self._first_order_difference(pose_weighted)

        frame_feat = torch.cat(
            [
                pose_weighted.reshape(pose.shape[0], pose.shape[1], self.num_joints * 2),
                pose_velocity.reshape(pose.shape[0], pose.shape[1], self.num_joints * 2),
                conf,
            ],
            dim=-1,
        )

        frame_feat = self.frame_mlp(frame_feat)
        temporal_feat, _ = self.temporal_encoder(frame_feat)
        pose_feat_seq = self.output_proj(temporal_feat)

        if torch.isnan(pose_feat_seq).any():
            raise ValueError('pose_feat_seq contains NaN values.')
        if torch.isinf(pose_feat_seq).any():
            raise ValueError('pose_feat_seq contains Inf values.')

        return pose_feat_seq

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
