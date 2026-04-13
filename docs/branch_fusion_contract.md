# Branch Fusion Contract

## Purpose

定义 Stage 4 的最小分支特征接口，只建立 pose 分支与现有位置分支的对齐和融合占位，不改变当前预测头行为。

## Branch Features

- `loc_feat_seq`: 位置分支序列特征，shape 为 `B x T x D_loc`
- `app_feat_seq`: 外观分支序列特征，shape 为 `B x T x D_app`，当前可为空
- `pose_feat_seq`: 姿态分支序列特征，shape 为 `B x T x D_pose`

## Input Contract

- 输入形式为 `dict[str, Tensor | None]`
- 每个已提供分支必须满足 `B x T x D`
- 所有已提供分支必须在 batch 维和时间维严格对齐
- 缺失分支允许传 `None`

## Time Alignment

- `T` 必须表示同一观测窗口上的对齐时间步
- 当前 JAAD pose 输入与 `pos` 使用同一个 observation window
- `BranchFeatureFusion` 只做显式 shape 校验，不做重采样或时间对齐修复

## Output Contract

- `fused_feat_seq`: `B x T x D_out`
- 当前最小实现采用 concat 后接一层线性投影
- 当前 `fused_feat_seq` 仅作为内部中间变量和调试输出，不驱动现有 decoder

## Files Modified

- `model/pose_fusion.py`
- `model/network_image.py`
- `docs/branch_fusion_contract.md`

## Commands Run

```bash
sed -n '1,260p' AGENTS.md
sed -n '1,260p' PLAN.md
sed -n '1,260p' model/pose_fusion.py
sed -n '1,260p' model/network_image.py
/home/cyq/anaconda3/envs/py38/bin/python -m py_compile model/pose_fusion.py
/home/cyq/anaconda3/envs/py38/bin/python -m py_compile model/pose_fusion.py model/network_image.py
/home/cyq/anaconda3/envs/py38/bin/python - <<'PY'
import torch
from model.pose_fusion import BranchFeatureFusion

torch.manual_seed(0)
fusion = BranchFeatureFusion(
    branch_dims={
        'loc_feat_seq': 128,
        'app_feat_seq': 128,
        'pose_feat_seq': 128,
    },
    out_dim=128,
)
inputs = {
    'loc_feat_seq': torch.randn(2, 16, 128),
    'app_feat_seq': None,
    'pose_feat_seq': torch.randn(2, 16, 128),
}
out = fusion(inputs)
print('random_loc_shape', tuple(inputs['loc_feat_seq'].shape))
print('random_pose_shape', tuple(inputs['pose_feat_seq'].shape))
print('random_fused_shape', tuple(out.shape))
print('random_nan', bool(torch.isnan(out).any().item()))
print('random_inf', bool(torch.isinf(out).any().item()))
PY
/home/cyq/anaconda3/envs/py38/bin/python - <<'PY'
from argparse import Namespace
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.jaad import JAAD
from model.network_image import PTINet

root = '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'
out_dir = '/tmp/e2t_stage4_pose_true_attr'
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

debug_feats = net.debug_last_features
print('loc_feat_seq_shape', tuple(debug_feats['loc_feat_seq'].shape))
print('pose_feat_seq_shape', tuple(debug_feats['pose_feat_seq'].shape))
print('fused_feat_seq_shape', tuple(debug_feats['fused_feat_seq'].shape))
print('speed_preds_shape', tuple(speed_preds.shape))
print('crossing_preds_shape', tuple(crossing_preds.shape))
print('intention_logits_shape', tuple(intention_logits.shape))
print('debug_nan', bool(torch.isnan(debug_feats['loc_feat_seq']).any().item() or torch.isnan(debug_feats['pose_feat_seq']).any().item() or torch.isnan(debug_feats['fused_feat_seq']).any().item()))
print('debug_inf', bool(torch.isinf(debug_feats['loc_feat_seq']).any().item() or torch.isinf(debug_feats['pose_feat_seq']).any().item() or torch.isinf(debug_feats['fused_feat_seq']).any().item()))
print('pred_nan', bool(torch.isnan(speed_preds).any().item() or torch.isnan(crossing_preds).any().item() or torch.isnan(intention_logits).any().item()))
print('pred_inf', bool(torch.isinf(speed_preds).any().item() or torch.isinf(crossing_preds).any().item() or torch.isinf(intention_logits).any().item()))
PY
```

## Known Risks

- 当前 `app_feat_seq` 只是接口占位，尚未从现有图像分支提取统一的 `B x T x D` 特征序列。
- 当前 `fused_feat_seq` 还没有接入 decoder，这符合 Stage 4 目标，但也意味着本阶段只验证接口可用性，不验证融合收益。
- 仓库里 `use_attribute=False` 的旧路径仍有与本阶段无关的问题，本次未扩大范围修复。
