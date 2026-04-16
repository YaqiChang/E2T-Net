import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
from model.network_image import PTINet


def _check_finite(tensor, name):
    if tensor is None:
        return
    if torch.isnan(tensor).any():
        raise ValueError(f'{name} contains NaN values.')
    if torch.isinf(tensor).any():
        raise ValueError(f'{name} contains Inf values.')


def _load_config():
    with open(REPO_ROOT / 'config.yml', 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _dataset_kwargs(config):
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    cache_file = os.path.join(out_dir, 'jaad_train_5_5_5.csv')
    from_file = os.path.exists(cache_file)

    return {
        'data_dir': config['data_dir'],
        'out_dir': out_dir,
        'dtype': 'train',
        'input': 5,
        'output': 5,
        'stride': 5,
        'skip': 1,
        'from_file': from_file,
        'save': False,
        'use_images': False,
        'use_attribute': True,
        'use_opticalflow': False,
        'use_pose': True,
        'pose_file': config.get('pose_file', ''),
        'pose_format': config.get('pose_format', 'jaad_hrnet_npz'),
    }


def _build_args():
    return SimpleNamespace(
        dataset='jaad',
        hidden_size=128,
        belief_dim=64,
        belief_readout='last',
        device=torch.device('cpu'),
        use_pose=True,
        use_decision_accumulator=True,
        use_fused_decoder_input=False,
        use_attribute=True,
        use_image=False,
        use_opticalflow=False,
        output=5,
        skip=1,
        hardtanh_limit=1.0,
    )


def _random_batch(batch_size=2, time_steps=5):
    return {
        'speed': torch.randn(batch_size, time_steps, 4),
        'pos': torch.randn(batch_size, time_steps, 4),
        'ped_attribute': torch.randn(batch_size, 3),
        'ped_behavior': torch.randn(batch_size, time_steps, 4),
        'scene_attribute': torch.randn(batch_size, time_steps, 10),
        'image': torch.empty(batch_size, 1, 1),
        'optical': torch.empty(batch_size, 1, 1),
        'pose': torch.randn(batch_size, time_steps, 17, 2),
        'pose_conf': torch.rand(batch_size, time_steps, 17),
    }


def _load_real_or_random_batch():
    try:
        config = _load_config()
        dataset = datasets.jaad(**_dataset_kwargs(config))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        return 'real_jaad', batch
    except Exception as exc:
        print(f'Falling back to random batch: {exc}')
        return 'random', _random_batch()


def _to_model_inputs(batch):
    return {
        'speed': batch['speed'].float(),
        'pos': batch['pos'].float(),
        'ped_attribute': batch['ped_attribute'].float(),
        'ped_behavior': batch['ped_behavior'].float(),
        'scene_attribute': batch['scene_attribute'].float(),
        'images': batch['image'].float(),
        'optical': batch['optical'].float(),
        'pose': batch['pose'].float(),
        'pose_conf': batch['pose_conf'].float(),
    }


def _grad_status(param, name):
    if param.grad is None:
        raise AssertionError(f'{name} grad is None.')
    _check_finite(param.grad, f'{name}.grad')
    return {
        'shape': tuple(param.grad.shape),
        'norm': float(param.grad.norm().item()),
        'has_nan': bool(torch.isnan(param.grad).any().item()),
        'has_inf': bool(torch.isinf(param.grad).any().item()),
    }


def main():
    torch.manual_seed(0)

    batch_source, batch = _load_real_or_random_batch()
    model_inputs = _to_model_inputs(batch)

    for key in ('pose', 'pose_conf'):
        _check_finite(model_inputs[key], f'batch.{key}')

    args = _build_args()
    model = PTINet(args)
    model.zero_grad(set_to_none=True)

    outputs = model(
        speed=model_inputs['speed'],
        pos=model_inputs['pos'],
        ped_attribute=model_inputs['ped_attribute'],
        ped_behavior=model_inputs['ped_behavior'],
        scene_attribute=model_inputs['scene_attribute'],
        images=model_inputs['images'],
        optical=model_inputs['optical'],
        pose=model_inputs['pose'],
        pose_conf=model_inputs['pose_conf'],
    )

    crossing_outputs = outputs[2]
    intention_logits = outputs[3]
    pose_evidence_seq = model.debug_last_features['pose_evidence_seq']
    belief_seq = model.debug_last_features['belief_seq']
    belief_last = model.debug_last_features['belief_last']

    _check_finite(crossing_outputs, 'crossing_outputs')
    _check_finite(intention_logits, 'intention_logits')
    _check_finite(pose_evidence_seq, 'pose_evidence_seq')
    _check_finite(belief_seq, 'belief_seq')
    _check_finite(belief_last, 'belief_last')

    loss = crossing_outputs.mean() + intention_logits.mean()
    _check_finite(loss, 'loss')
    loss.backward()

    pose_grad = _grad_status(
        model.pose_encoder.frame_mlp[0].weight,
        'pose_encoder.frame_mlp[0].weight',
    )
    accumulator_grad = _grad_status(
        model.pose_accumulator.evidence_proj.weight,
        'pose_accumulator.evidence_proj.weight',
    )
    head_grad = _grad_status(
        model.fc_intention_belief.weight,
        'fc_intention_belief.weight',
    )

    grads_have_nan = pose_grad['has_nan'] or accumulator_grad['has_nan'] or head_grad['has_nan']
    grads_have_inf = pose_grad['has_inf'] or accumulator_grad['has_inf'] or head_grad['has_inf']

    print(f'batch source: {batch_source}')
    print(f'pose shape: {tuple(model_inputs["pose"].shape)}')
    print(f'pose_conf shape: {tuple(model_inputs["pose_conf"].shape)}')
    print(f'crossing output shape: {tuple(crossing_outputs.shape)}')
    print(f'intention output shape: {tuple(intention_logits.shape)}')
    print(f'pose_evidence_seq shape: {tuple(pose_evidence_seq.shape)}')
    print(f'belief_seq shape: {tuple(belief_seq.shape)}')
    print(f'belief_last shape: {tuple(belief_last.shape)}')
    print(f'loss shape: {tuple(loss.shape)}')
    print(f'pose grad status: {pose_grad}')
    print(f'accumulator grad status: {accumulator_grad}')
    print(f'head grad status: {head_grad}')
    print(f'grad_has_nan: {grads_have_nan}')
    print(f'grad_has_inf: {grads_have_inf}')


if __name__ == '__main__':
    main()
