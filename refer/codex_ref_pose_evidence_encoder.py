"""
Reference only.
Goal:
- show the target API for PoseEvidenceEncoder
- remove temporal LSTM
"""

import torch
import torch.nn as nn


class PoseEvidenceEncoder(nn.Module):
    def __init__(self, num_joints=17, frame_hidden_dim=128, out_dim=128, conf_power=1.0, eps=1e-6):
        super().__init__()
        self.num_joints = int(num_joints)
        self.frame_hidden_dim = int(frame_hidden_dim)
        self.out_dim = int(out_dim)
        self.conf_power = float(conf_power)
        self.eps = float(eps)

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

        pose_evidence_seq = self.frame_mlp(frame_feat)

        if torch.isnan(pose_evidence_seq).any():
            raise ValueError("pose_evidence_seq contains NaN values.")
        if torch.isinf(pose_evidence_seq).any():
            raise ValueError("pose_evidence_seq contains Inf values.")
        return pose_evidence_seq

    def _check_inputs(self, pose, pose_conf):
        if pose.ndim != 4:
            raise ValueError(f"pose must have shape B x T x J x 2, got {tuple(pose.shape)}.")
        if pose_conf.ndim != 3:
            raise ValueError(f"pose_conf must have shape B x T x J, got {tuple(pose_conf.shape)}.")
        _, _, num_joints, channels = pose.shape
        if channels != 2:
            raise ValueError(f"pose last dimension must be 2, got {channels}.")
        if num_joints != self.num_joints or pose_conf.shape[-1] != self.num_joints:
            raise ValueError("joint dimension mismatch.")

    def _body_center_normalize(self, pose, pose_conf):
        left_shoulder = 5
        right_shoulder = 6
        left_hip = 11
        right_hip = 12

        conf = torch.clamp(pose_conf, min=0.0, max=1.0).unsqueeze(-1)
        shoulder_weights = conf[:, :, [left_shoulder, right_shoulder], :]
        hip_weights = conf[:, :, [left_hip, right_hip], :]

        shoulder_center = (pose[:, :, [left_shoulder, right_shoulder], :] * shoulder_weights).sum(dim=2) / torch.clamp(shoulder_weights.sum(dim=2), min=self.eps)
        hip_center = (pose[:, :, [left_hip, right_hip], :] * hip_weights).sum(dim=2) / torch.clamp(hip_weights.sum(dim=2), min=self.eps)

        root = hip_center.unsqueeze(2)
        torso_scale = torch.norm(shoulder_center - hip_center, dim=-1, keepdim=True)
        torso_scale = torch.clamp(torso_scale, min=self.eps).unsqueeze(2)
        return (pose - root) / torso_scale

    def _first_order_difference(self, pose_feat):
        motion = torch.zeros_like(pose_feat)
        motion[:, 1:] = pose_feat[:, 1:] - pose_feat[:, :-1]
        return motion
