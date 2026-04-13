# Fused Decoder Stage 5

## Purpose

在不改输出签名和损失设计的前提下，让 `fused_feat_seq` 以可开关方式影响当前 PTINet decoder。

## New Flag

- `use_fused_decoder_input`

三种行为保持分离：

- `use_pose=False`
- `use_pose=True` 且 `use_fused_decoder_input=False`
- `use_pose=True` 且 `use_fused_decoder_input=True`

## Decoder Integration

- 仅在 `use_pose=True` 且 `use_fused_decoder_input=True` 时生效
- 使用 `fused_feat_seq[:, -1, :]` 作为 observation 末时刻的融合特征
- 通过线性层投影到 decoder hidden size
- 只加到两个 decoder 的初始 hidden state：
  - speed decoder 的 `hds`
  - crossing decoder 的 `hdc`
- 不改 decoder recurrence 结构
- 不改 cell state 初始化逻辑

## Unchanged

- `speed_preds` / `crossing_preds` / `intention_logits` 输出签名不变
- 原有 loss 和训练流不变
- `use_pose=False` 的 baseline 路径保持原样
- `use_pose=True` 且 `use_fused_decoder_input=False` 时，仅计算 pose/fusion 中间特征，不驱动 decoder

## Files Modified

- `model/network_image.py`
- `docs/fused_decoder_stage5.md`

## Commands Run

```bash
sed -n '1,260p' AGENTS.md
sed -n '1,260p' PLAN.md
sed -n '1,360p' model/network_image.py
sed -n '360,760p' model/network_image.py
/home/cyq/anaconda3/envs/py38/bin/python -m py_compile model/network_image.py
/home/cyq/anaconda3/envs/py38/bin/python - <<'PY'
from argparse import Namespace
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.jaad import JAAD
from model.network_image import PTINet

root = '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'
out_dir = '/tmp/e2t_stage5_pose_false'
Path(out_dir).mkdir(parents=True, exist_ok=True)

dataset = JAAD(
    data_dir=root,
    out_dir=out_dir,
    dtype='train',
    input=16,
    output=32,
    stride=16,
    skip=1,
    from_file=False,
    save=True,
    use_images=False,
    use_attribute=True,
    use_opticalflow=False,
    use_pose=False,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))

args = Namespace(
    dataset='jaad',
    hidden_size=128,
    device='cpu',
    use_attribute=True,
    use_image=False,
    image_network='resnet50',
    use_opticalflow=False,
    output=32,
    skip=1,
    hardtanh_limit=100,
    use_pose=False,
    use_fused_decoder_input=False,
)
net = PTINet(args)
net.eval()
with torch.no_grad():
    mloss, speed_preds, crossing_preds, intention_logits = net(
        speed=batch['speed'].float(),
        pos=batch['pos'].float(),
        ped_attribute=batch['ped_attribute'].float(),
        ped_behavior=batch['ped_behavior'].float(),
        scene_attribute=batch['scene_attribute'].float(),
        images=batch['image'].float(),
        optical=batch['optical'].float(),
        average=False,
    )
print('pose_false_speed', tuple(speed_preds.shape))
print('pose_false_crossing', tuple(crossing_preds.shape))
print('pose_false_intention', tuple(intention_logits.shape))
print('pose_false_nan', bool(torch.isnan(speed_preds).any().item() or torch.isnan(crossing_preds).any().item() or torch.isnan(intention_logits).any().item()))
print('pose_false_inf', bool(torch.isinf(speed_preds).any().item() or torch.isinf(crossing_preds).any().item() or torch.isinf(intention_logits).any().item()))
PY
/home/cyq/anaconda3/envs/py38/bin/python - <<'PY'
from argparse import Namespace
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.jaad import JAAD
from model.network_image import PTINet

root = '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'
out_dir = '/tmp/e2t_stage5_pose_true_no_fused'
Path(out_dir).mkdir(parents=True, exist_ok=True)
pose_file = str(Path(root) / 'jaad_pose_annotations_fixed.npz')

dataset = JAAD(
    data_dir=root,
    out_dir=out_dir,
    dtype='train',
    input=16,
    output=32,
    stride=16,
    skip=1,
    from_file=False,
    save=True,
    use_images=False,
    use_attribute=True,
    use_opticalflow=False,
    use_pose=True,
    pose_file=pose_file,
    pose_format='jaad_hrnet_npz',
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))

args = Namespace(
    dataset='jaad',
    hidden_size=128,
    device='cpu',
    use_attribute=True,
    use_image=False,
    image_network='resnet50',
    use_opticalflow=False,
    output=32,
    skip=1,
    hardtanh_limit=100,
    use_pose=True,
    use_fused_decoder_input=False,
)
net = PTINet(args)
net.eval()
with torch.no_grad():
    mloss, speed_preds, crossing_preds, intention_logits = net(
        speed=batch['speed'].float(),
        pos=batch['pos'].float(),
        ped_attribute=batch['ped_attribute'].float(),
        ped_behavior=batch['ped_behavior'].float(),
        scene_attribute=batch['scene_attribute'].float(),
        images=batch['image'].float(),
        optical=batch['optical'].float(),
        pose=batch['pose'].float(),
        pose_conf=batch['pose_conf'].float(),
        average=False,
    )
print('pose_true_no_fused_feat', tuple(net.debug_last_features['fused_feat_seq'].shape))
print('pose_true_no_fused_speed', tuple(speed_preds.shape))
print('pose_true_no_fused_crossing', tuple(crossing_preds.shape))
print('pose_true_no_fused_intention', tuple(intention_logits.shape))
print('pose_true_no_fused_nan', bool(torch.isnan(speed_preds).any().item() or torch.isnan(crossing_preds).any().item() or torch.isnan(intention_logits).any().item()))
print('pose_true_no_fused_inf', bool(torch.isinf(speed_preds).any().item() or torch.isinf(crossing_preds).any().item() or torch.isinf(intention_logits).any().item()))
PY
/home/cyq/anaconda3/envs/py38/bin/python - <<'PY'
from argparse import Namespace
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.jaad import JAAD
from model.network_image import PTINet

root = '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'
out_dir = '/tmp/e2t_stage5_pose_true_fused'
Path(out_dir).mkdir(parents=True, exist_ok=True)
pose_file = str(Path(root) / 'jaad_pose_annotations_fixed.npz')

dataset = JAAD(
    data_dir=root,
    out_dir=out_dir,
    dtype='train',
    input=16,
    output=32,
    stride=16,
    skip=1,
    from_file=False,
    save=True,
    use_images=False,
    use_attribute=True,
    use_opticalflow=False,
    use_pose=True,
    pose_file=pose_file,
    pose_format='jaad_hrnet_npz',
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))

args = Namespace(
    dataset='jaad',
    hidden_size=128,
    device='cpu',
    use_attribute=True,
    use_image=False,
    image_network='resnet50',
    use_opticalflow=False,
    output=32,
    skip=1,
    hardtanh_limit=100,
    use_pose=True,
    use_fused_decoder_input=True,
)
net = PTINet(args)
net.train()
mloss, speed_preds, crossing_preds, intention_logits = net(
    speed=batch['speed'].float(),
    pos=batch['pos'].float(),
    ped_attribute=batch['ped_attribute'].float(),
    ped_behavior=batch['ped_behavior'].float(),
    scene_attribute=batch['scene_attribute'].float(),
    images=batch['image'].float(),
    optical=batch['optical'].float(),
    pose=batch['pose'].float(),
    pose_conf=batch['pose_conf'].float(),
    average=False,
)
loss = mloss + speed_preds.mean() + crossing_preds.mean() + intention_logits.mean()
loss.backward()
print('pose_true_fused_feat', tuple(net.debug_last_features['fused_feat_seq'].shape))
print('pose_true_fused_speed', tuple(speed_preds.shape))
print('pose_true_fused_crossing', tuple(crossing_preds.shape))
print('pose_true_fused_intention', tuple(intention_logits.shape))
print('pose_true_fused_nan', bool(torch.isnan(speed_preds).any().item() or torch.isnan(crossing_preds).any().item() or torch.isnan(intention_logits).any().item()))
print('pose_true_fused_inf', bool(torch.isinf(speed_preds).any().item() or torch.isinf(crossing_preds).any().item() or torch.isinf(intention_logits).any().item()))
PY
```

## Known Risks

- 当前只把 `fused_feat_seq` 的最后一个时间步接入 decoder 初始 hidden state，没有探索更强的时序融合策略。
- `use_fused_decoder_input` 目前通过 `getattr(args, ..., False)` 读取；若后续希望命令行显式控制，还需要在训练参数解析处补 flag。
- 仓库中 `use_attribute=False` 的旧路径仍有独立问题，本次没有扩大范围修复。
