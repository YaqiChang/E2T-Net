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
    config_path = REPO_ROOT / 'config.yml'
    with open(config_path, 'r', encoding='utf-8') as handle:
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


def _build_args(use_pose, use_decision_accumulator):
    return SimpleNamespace(
        dataset='jaad',
        hidden_size=128,
        device=torch.device('cpu'),
        use_pose=use_pose,
        use_decision_accumulator=use_decision_accumulator,
        use_fused_decoder_input=False,
        use_attribute=True,
        use_image=False,
        use_opticalflow=False,
        output=5,
        skip=1,
        hardtanh_limit=1.0,
    )


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


def _run_mode(name, batch, use_pose, use_decision_accumulator):
    args = _build_args(use_pose=use_pose, use_decision_accumulator=use_decision_accumulator)
    model = PTINet(args)
    model.eval()

    model_inputs = _to_model_inputs(batch)

    forward_kwargs = {
        'speed': model_inputs['speed'],
        'pos': model_inputs['pos'],
        'ped_attribute': model_inputs['ped_attribute'],
        'ped_behavior': model_inputs['ped_behavior'],
        'scene_attribute': model_inputs['scene_attribute'],
        'images': model_inputs['images'],
        'optical': model_inputs['optical'],
    }
    if use_pose:
        forward_kwargs['pose'] = model_inputs['pose']
        forward_kwargs['pose_conf'] = model_inputs['pose_conf']

    with torch.no_grad():
        outputs = model(**forward_kwargs)

    crossing_output = outputs[2]
    intention_output = outputs[3]
    debug = model.debug_last_features

    _check_finite(crossing_output, f'{name}.crossing_output')
    _check_finite(intention_output, f'{name}.intention_output')

    result = {
        'crossing_output_shape': tuple(crossing_output.shape),
        'intention_output_shape': tuple(intention_output.shape),
        'pose_evidence_seq_shape': None,
        'pose_evidence_last_shape': None,
        'belief_seq_shape': None,
        'belief_last_shape': None,
        'has_nan': False,
        'has_inf': False,
    }

    pose_evidence_seq = debug.get('pose_evidence_seq')
    pose_evidence_last = debug.get('pose_evidence_last')
    belief_seq = debug.get('belief_seq')
    belief_last = debug.get('belief_last')

    if pose_evidence_seq is not None:
        _check_finite(pose_evidence_seq, f'{name}.pose_evidence_seq')
        result['pose_evidence_seq_shape'] = tuple(pose_evidence_seq.shape)
    if pose_evidence_last is not None:
        _check_finite(pose_evidence_last, f'{name}.pose_evidence_last')
        result['pose_evidence_last_shape'] = tuple(pose_evidence_last.shape)
    if belief_seq is not None:
        _check_finite(belief_seq, f'{name}.belief_seq')
        result['belief_seq_shape'] = tuple(belief_seq.shape)
    if belief_last is not None:
        _check_finite(belief_last, f'{name}.belief_last')
        result['belief_last_shape'] = tuple(belief_last.shape)

    tensors = [crossing_output, intention_output]
    if pose_evidence_seq is not None:
        tensors.append(pose_evidence_seq)
    if pose_evidence_last is not None:
        tensors.append(pose_evidence_last)
    if belief_seq is not None:
        tensors.append(belief_seq)
    if belief_last is not None:
        tensors.append(belief_last)

    result['has_nan'] = any(torch.isnan(tensor).any().item() for tensor in tensors)
    result['has_inf'] = any(torch.isinf(tensor).any().item() for tensor in tensors)

    if not use_pose:
        if pose_evidence_seq is not None or pose_evidence_last is not None:
            raise AssertionError('baseline should not produce pose evidence features.')
        if belief_seq is not None or belief_last is not None:
            raise AssertionError('baseline should not produce belief features.')
    elif not use_decision_accumulator:
        if pose_evidence_seq is None or pose_evidence_last is None:
            raise AssertionError('pose_direct_last must produce pose evidence features.')
        if belief_seq is not None or belief_last is not None:
            raise AssertionError('pose_direct_last should not produce belief features.')
    else:
        if pose_evidence_seq is None or pose_evidence_last is None:
            raise AssertionError('pose_accumulator must produce pose evidence features.')
        if belief_seq is None or belief_last is None:
            raise AssertionError('pose_accumulator must produce belief features.')

    return result


def main():
    torch.manual_seed(0)

    config = _load_config()
    dataset_kwargs = _dataset_kwargs(config)
    dataset = datasets.jaad(**dataset_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    pose = batch['pose'].float()
    pose_conf = batch['pose_conf'].float()
    _check_finite(pose, 'batch.pose')
    _check_finite(pose_conf, 'batch.pose_conf')

    results = {
        'baseline': _run_mode('baseline', batch, use_pose=False, use_decision_accumulator=False),
        'pose_direct_last': _run_mode('pose_direct_last', batch, use_pose=True, use_decision_accumulator=False),
        'pose_accumulator': _run_mode('pose_accumulator', batch, use_pose=True, use_decision_accumulator=True),
    }

    print(f'batch pose shape: {tuple(pose.shape)}')
    print(f'batch pose_conf shape: {tuple(pose_conf.shape)}')

    for mode_name, result in results.items():
        print(f'{mode_name} crossing output shape: {result["crossing_output_shape"]}')
        print(f'{mode_name} intention output shape: {result["intention_output_shape"]}')
        print(f'{mode_name} pose_evidence_seq shape: {result["pose_evidence_seq_shape"]}')
        print(f'{mode_name} pose_evidence_last shape: {result["pose_evidence_last_shape"]}')
        print(f'{mode_name} belief_seq shape: {result["belief_seq_shape"]}')
        print(f'{mode_name} belief_last shape: {result["belief_last_shape"]}')
        print(f'{mode_name} has_nan: {result["has_nan"]}')
        print(f'{mode_name} has_inf: {result["has_inf"]}')


if __name__ == '__main__':
    main()
