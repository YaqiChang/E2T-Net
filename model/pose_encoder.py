import torch
import torch.nn as nn

class PoseSequenceEncoder(nn.Module):
    def __init__(self, num_joints, out_dim=128, hidden_dim=128, use_lstm=True):
        super().__init__()
        self.num_joints = num_joints
        self.use_lstm = use_lstm

        frame_dim = num_joints * 2 + num_joints

        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if use_lstm:
            self.temporal = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=out_dim,
                num_layers=1,
                batch_first=True
            )
        else:
            self.temporal = nn.Linear(hidden_dim, out_dim)

    def forward(self, pose, pose_conf):
        B, T, J, C = pose.shape

        center = pose.mean(dim=2, keepdim=True)
        pose_norm = pose - center

        pose_flat = pose_norm.reshape(B, T, J * C)
        conf_flat = pose_conf.reshape(B, T, J)
        x = torch.cat([pose_flat, conf_flat], dim=-1)

        x = self.frame_mlp(x)

        if self.use_lstm:
            x, _ = self.temporal(x)
        else:
            x = self.temporal(x)

        return x