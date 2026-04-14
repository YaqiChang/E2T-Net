# E2T-Net Stage 5 Experiments

当前仓库已进入第一轮 JAAD pose ablation 的可运行阶段。现阶段使用离线缓存的姿态文件，不包含在线 pose provider。

## 当前实验模式

- `baseline`
  - `use_pose=false`
  - `use_fused_decoder_input=false`
- `pose_no_fused`
  - `use_pose=true`
  - `use_fused_decoder_input=false`
- `pose_fused`
  - `use_pose=true`
  - `use_fused_decoder_input=true`

## Flag 含义

- `use_pose`
  - 启用 JAAD 离线 pose 数据读取，并在模型内部计算 `pose_feat_seq`
- `use_fused_decoder_input`
  - 将 `fused_feat_seq` 注入当前 PTINet decoder 的初始 hidden state
  - 该开关只有在 `use_pose=true` 时才有实际作用

## 必需数据

- JAAD 数据根目录：
  - `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego`
- JAAD 离线 pose 文件：
  - `/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz`

## 运行脚本

脚本目录：

- [scripts/experiments/jaad_baseline.sh](/media/cyq/Data/project/PIP/E2T-Net/scripts/experiments/jaad_baseline.sh)
- [scripts/experiments/jaad_pose_no_fused.sh](/media/cyq/Data/project/PIP/E2T-Net/scripts/experiments/jaad_pose_no_fused.sh)
- [scripts/experiments/jaad_pose_fused.sh](/media/cyq/Data/project/PIP/E2T-Net/scripts/experiments/jaad_pose_fused.sh)

运行方式：

```bash
bash scripts/experiments/jaad_baseline.sh
bash scripts/experiments/jaad_pose_no_fused.sh
bash scripts/experiments/jaad_pose_fused.sh
```

如果只想检查命令而不真正启动训练：

```bash
DRY_RUN=1 bash scripts/experiments/jaad_baseline.sh
DRY_RUN=1 bash scripts/experiments/jaad_pose_no_fused.sh
DRY_RUN=1 bash scripts/experiments/jaad_pose_fused.sh
```

## 输出位置

三个脚本都把实验输出写到：

- `/media/cyq/Data/project/PIP/E2T-Net/output/stage5_ablation/`

对应 `log_name`：

- `jaad_baseline`
- `jaad_pose_no_fused`
- `jaad_pose_fused`

因此每次实验的实际输出目录分别是：

- `output/stage5_ablation/jaad_baseline/`
- `output/stage5_ablation/jaad_pose_no_fused/`
- `output/stage5_ablation/jaad_pose_fused/`

## 说明

- 当前实验只使用缓存离线 pose，不做在线姿态估计。
- 在线 pose provider 属于后续工作，不在当前实验范围内。
