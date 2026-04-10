# add on 17 Mar 2024 by CYQ
import time
import os
import argparse
import sys
from ast import literal_eval

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import datasets
import model.network_image as network
import utils
from utils import binary_classification_metrics
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value

    lowered = str(value).strip().lower()
    if lowered in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if lowered in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PTINet network')

    parser.add_argument('--data_dir', type=str,
                        default='/scratch/project_2007864/PIE/PN/',
                        required=False)
    parser.add_argument('--dataset', type=str,
                        default='pie',
                        required=False)
    parser.add_argument('--out_dir', type=str,
                        default='/projappl/project_2007864/PIE_lstm_vae_clstm/bounding-box-prediction/output_32/',
                        required=False)
    parser.add_argument('--artifact_dir', type=str, default='',
                        help='Directory containing checkpoint, cached sequences, and best_metrics.json.',
                        required=False)
    parser.add_argument('--task', type=str,
                        default='2D_bounding_box-intention',
                        required=False)

    parser.add_argument('--input', type=int,
                        default=16,
                        required=False)
    parser.add_argument('--output', type=int,
                        default=48,
                        required=False)
    parser.add_argument('--stride', type=int,
                        default=16,
                        required=False)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--is_3D', type=parse_bool_arg, default=False)
    parser.add_argument('--dtype', type=str, default='test')
    parser.add_argument('--from_file', type=parse_bool_arg, default=False)
    parser.add_argument('--save', type=parse_bool_arg, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Checkpoint file to evaluate. Accepts .pkl or .pth.')
    parser.add_argument('--save_sample_results', type=parse_bool_arg, default=True,
                        help='Export per-sample prediction rows for downstream analysis.',
                        required=False)
    parser.add_argument('--sample_results_dir', type=str, default='',
                        help='Directory for analysis-ready exported prediction rows.',
                        required=False)
    parser.add_argument('--sample_results_name', type=str, default='',
                        help='Filename for the exported prediction table.',
                        required=False)
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=parse_bool_arg, default=True)
    parser.add_argument('--pin_memory', type=parse_bool_arg, default=False)
    parser.add_argument('--prefetch_factor', type=int, default=3)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--lr_scheduler', type=parse_bool_arg, default=False)
    parser.add_argument('--crossing_pos_weight', type=float, default=1.0)
    parser.add_argument('--intent_pos_weight', type=float, default=1.0)
    parser.add_argument('--threshold_metric', type=str, default='f0.5')
    parser.add_argument('--use_saved_thresholds', type=parse_bool_arg, default=True)

    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--use_image', type=parse_bool_arg, default=True,
                        help='Use input image as a feature',
                        required=False)
    parser.add_argument('--image_network', type=str, default='clstm',
                        help='select backbone',
                        required=False)
    parser.add_argument('--use_attribute', type=parse_bool_arg, default=True,
                        help='Use input attribute as a feature',
                        required=False)
    parser.add_argument('--use_opticalflow', type=parse_bool_arg, default=False,
                        help='Use optical flow as a feature',
                        required=False)

    return parser.parse_args()


def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def create_progress(iterable, desc, total, verbose):
    if not verbose:
        return iterable
    if tqdm is not None:
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            file=resolve_progress_stream(sys.stderr),
        )
    return iterable


def log_batch_progress(verbose, phase, batch_idx, total_batches, **metrics):
    if not verbose or tqdm is not None or total_batches == 0:
        return

    should_print = batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 10) == 0
    if not should_print:
        return

    metric_text = ' | '.join(f'{name}: {value:.4f}' for name, value in metrics.items())
    print(f'[{phase}] batch {batch_idx}/{total_batches}' + (f' | {metric_text}' if metric_text else ''))


def resolve_progress_stream(stream):
    if hasattr(stream, 'streams'):
        for child_stream in stream.streams:
            if getattr(child_stream, 'isatty', lambda: False)():
                return child_stream
        if stream.streams:
            return stream.streams[0]
    return stream


def should_use_distributed():
    try:
        return int(os.environ.get('WORLD_SIZE', '1')) > 1
    except ValueError:
        return False


def resolve_runtime_device(device_arg, local_rank=0):
    if not torch.cuda.is_available():
        return torch.device('cpu')

    if device_arg.startswith('cuda:'):
        return torch.device(device_arg)

    if device_arg == 'cuda':
        return torch.device(f'cuda:{local_rank}')

    return torch.device(device_arg)


def get_run_dir(args):
    if getattr(args, 'artifact_dir', ''):
        return args.artifact_dir
    return os.path.join(args.out_dir, args.log_name) if args.log_name else args.out_dir


def get_checkpoint_dir(args):
    if not getattr(args, 'checkpoint', ''):
        return ''
    if os.path.isabs(args.checkpoint) and os.path.exists(args.checkpoint):
        return os.path.dirname(args.checkpoint)

    direct_path = os.path.abspath(args.checkpoint)
    if os.path.exists(direct_path):
        return os.path.dirname(direct_path)
    return ''


def sequence_cache_name(args):
    return f'{args.dataset}_{args.dtype}_{args.input}_{args.output}_{args.stride}.csv'


def resolve_artifact_dir(args, expected_filename=''):
    candidates = []
    primary = get_run_dir(args)
    checkpoint_dir = get_checkpoint_dir(args)
    for candidate in [primary, checkpoint_dir, args.out_dir]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if expected_filename:
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, expected_filename)):
                return candidate

    return primary


def checkpoint_suffix(args):
    return '_scheduler' if args.lr_scheduler else ''


def default_checkpoint_name(args):
    return f'model_best{checkpoint_suffix(args)}.pkl'


def best_summary_name(args):
    return f'best_metrics{checkpoint_suffix(args)}.json'


def default_sample_results_name(args):
    return f'{args.dataset}_{args.dtype}_sample_predictions.csv'


def default_sample_results_dir(args):
    artifact_dir = resolve_artifact_dir(args)
    return os.path.join(artifact_dir, 'res_analyze')


def resolve_checkpoint_path(args):
    run_dir = resolve_artifact_dir(args, args.checkpoint or default_checkpoint_name(args))
    if not args.checkpoint:
        checkpoint_path = os.path.join(run_dir, default_checkpoint_name(args))
    elif os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        direct_path = os.path.abspath(args.checkpoint)
        checkpoint_path = direct_path if os.path.exists(direct_path) else os.path.join(run_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    return checkpoint_path

# Loads saved thresholds from the best summary file if it exists and contains valid thresholds.
def load_saved_thresholds(args):
    summary_dir = resolve_artifact_dir(args, best_summary_name(args))
    summary_path = os.path.join(summary_dir, best_summary_name(args))
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, 'r', encoding='utf-8') as handle:
        summary = yaml.safe_load(handle)
    if not isinstance(summary, dict):
        return None

    state_threshold = summary.get('state_threshold')
    intent_threshold = summary.get('intent_threshold')
    if state_threshold is None or intent_threshold is None:
        return None

    return float(state_threshold), float(intent_threshold)


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    target_model = model.module if isinstance(model, DistributedDataParallel) else model
    if any(key.startswith('module.') for key in checkpoint.keys()):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
    if 'fc_intention.weight' not in checkpoint and 'fc_attrib.0.weight' in checkpoint:
        checkpoint['fc_intention.weight'] = checkpoint['fc_attrib.0.weight'][:2].clone()
        checkpoint['fc_intention.bias'] = checkpoint['fc_attrib.0.bias'][:2].clone()
    missing_keys, unexpected_keys = target_model.load_state_dict(checkpoint, strict=False)
    if missing_keys:
        print('Missing checkpoint keys:', missing_keys)
    if unexpected_keys:
        print('Unexpected checkpoint keys:', unexpected_keys)


def parse_sequence_cell(value):
    if isinstance(value, str):
        return literal_eval(value)
    return value


def maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_video_id(image_paths):
    if not image_paths:
        return ''
    return os.path.basename(os.path.dirname(image_paths[0]))


def extract_frame_id(path_or_name):
    stem = os.path.splitext(os.path.basename(str(path_or_name)))[0]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    return maybe_int(digits)


def infer_trigger_frame_id(obs_frame_ids, obs_cross, future_cross, skip):
    for frame_id, state in zip(obs_frame_ids, obs_cross):
        if float(state) >= 0.5 and frame_id is not None:
            return frame_id

    valid_obs_frames = [frame_id for frame_id in obs_frame_ids if frame_id is not None]
    if not valid_obs_frames:
        return None
    last_obs_frame = valid_obs_frames[-1]
    for future_index, state in enumerate(future_cross, start=1):
        if float(state) >= 0.5:
            return last_obs_frame + future_index * skip
    return None


def resolve_sample_results_path(args):
    output_dir = args.sample_results_dir or default_sample_results_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    filename = args.sample_results_name or default_sample_results_name(args)
    return os.path.join(output_dir, filename)


def build_sample_rows(
    args,
    eval_set,
    start_index,
    batch_size,
    crossing_probs,
    future_cross_np,
    intent_probs,
    intent_targets_np,
):
    rows = []
    horizon = crossing_probs.shape[1]
    for batch_index in range(batch_size):
        dataset_index = start_index + batch_index
        seq = eval_set.data.iloc[dataset_index]

        image_paths = parse_sequence_cell(seq['imagefolderpath']) if 'imagefolderpath' in seq else []
        filenames = parse_sequence_cell(seq['filename']) if 'filename' in seq else []
        crossing_obs = parse_sequence_cell(seq['crossing_obs']) if 'crossing_obs' in seq else []
        future_cross_seq = parse_sequence_cell(seq['crossing_true']) if 'crossing_true' in seq else future_cross_np[batch_index].tolist()

        obs_frame_ids = [extract_frame_id(path) for path in image_paths] if image_paths else [extract_frame_id(name) for name in filenames]
        last_obs_frame = next((frame_id for frame_id in reversed(obs_frame_ids) if frame_id is not None), None)
        trigger_frame = infer_trigger_frame_id(obs_frame_ids, crossing_obs, future_cross_seq, args.skip)
        video_id = extract_video_id(image_paths)
        ped_id = seq['ID'] if 'ID' in seq else dataset_index
        sequence_key = f'{args.dtype}_{dataset_index}_{ped_id}'

        for step_index in range(horizon):
            frame_id = None if last_obs_frame is None else last_obs_frame + (step_index + 1) * args.skip
            time_to_trigger = None if trigger_frame is None or frame_id is None else frame_id - trigger_frame
            rows.append({
                'split': args.dtype,
                'sequence_key': sequence_key,
                'dataset_index': dataset_index,
                'video_id': video_id,
                'ped_id': ped_id,
                'frame_id': frame_id,
                'step_idx': step_index,
                'state_gt': int(future_cross_np[batch_index, step_index]),
                'intent_gt': int(intent_targets_np[batch_index]),
                'state_score': float(crossing_probs[batch_index, step_index]),
                'intent_score': float(intent_probs[batch_index]),
                'trigger_frame': trigger_frame,
                'time_to_trigger': time_to_trigger,
            })
    return rows


def export_sample_results(args, rows, state_threshold, intent_threshold):
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df['state_pred_at_eval_th'] = (df['state_score'].to_numpy() >= state_threshold).astype(np.int64)
    intent_pred = (df['intent_score'].to_numpy() >= intent_threshold).astype(np.int64)
    df['intent_pred_at_eval_th'] = intent_pred
    df['state_threshold'] = float(state_threshold)
    df['intent_threshold'] = float(intent_threshold)
    output_path = resolve_sample_results_path(args)
    df.to_csv(output_path, index=False)
    print(f'Saved sample analysis table: {output_path}')
    return output_path


def evaluate_2d(args, eval_set):
    print('=' * 100)
    print('Evaluating ...')
    print('Split: ' + str(args.dtype))
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    torch.manual_seed(0)
    use_distributed = should_use_distributed()
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = resolve_runtime_device(args.device, local_rank)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    args.device = str(device)
    args.crossing_pos_weight = 1.0
    args.intent_pos_weight = 1.0

    verbose = dist.get_rank() == 0 if use_distributed else True
    net = network.PTINet(args).to(device)
    if use_distributed:
        net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    checkpoint_path = resolve_checkpoint_path(args)
    print(checkpoint_path)
    load_model_state(net, checkpoint_path, device)
    if args.dtype == 'test':
        if not args.use_saved_thresholds:
            raise ValueError(
                'Test evaluation must use thresholds saved from the validation split. '
                'Set --use_saved_thresholds True or evaluate on val.'
            )
        saved_thresholds = load_saved_thresholds(args)
        if saved_thresholds is None:
            raise FileNotFoundError(
                'Validation thresholds were not found. Run training first so best_metrics.json '
                'contains state_threshold and intent_threshold.'
            )
    else:
        saved_thresholds = load_saved_thresholds(args) if args.use_saved_thresholds else None
    net.eval()
    eval_sampler = DistributedSampler(eval_set) if use_distributed else None
    dataloader_kwargs = {
        'dataset': eval_set,
        'batch_size': args.batch_size,
        'pin_memory': args.pin_memory,
        'sampler': eval_sampler,
        'shuffle': False,
        'num_workers': args.loader_workers,
        'drop_last': True,
    }
    if args.loader_workers > 0:
        dataloader_kwargs['prefetch_factor'] = args.prefetch_factor
    dataloader_eval = torch.utils.data.DataLoader(**dataloader_kwargs)

    huber = torch.nn.HuberLoss(reduction='sum', delta=1.0)
    crossing_ce = nn.CrossEntropyLoss()
    intent_ce = nn.CrossEntropyLoss()

    ade = 0
    fde = 0
    aiou = 0
    fiou = 0

    avg_epoch_eval_s_loss = 0
    avg_epoch_eval_c_loss = 0
    avg_epoch_eval_i_loss = 0

    counter = 0
    state_targets = []
    state_probs = []
    intent_targets = []
    intent_probs = []
    sample_rows = []
    sample_start_index = 0

    start = time.time()
    eval_batches = len(dataloader_eval)
    eval_iterator = create_progress(
        enumerate(dataloader_eval, start=1),
        desc=f'Eval[{args.dtype}]',
        total=eval_batches,
        verbose=verbose,
    )

    for idx, inputs in eval_iterator:
        counter += 1
        speed = inputs['speed'].to(device, non_blocking=True)
        future_speed = inputs['future_speed'].to(device, non_blocking=True)
        pos = inputs['pos'].to(device, non_blocking=True)
        future_pos = inputs['future_pos'].to(device, non_blocking=True)
        future_cross = inputs['future_cross'].to(device, non_blocking=True)
        ped_attribute = inputs['ped_attribute'].to(device, non_blocking=True)
        scene_attribute = inputs['scene_attribute'].to(device, non_blocking=True)
        ped_behavior = inputs['ped_behavior'].to(device, non_blocking=True)
        optical = inputs['optical'].to(device, non_blocking=True)
        images = inputs['image'].to(device, non_blocking=True)
        label_c = inputs['cross_label'].to(device, non_blocking=True)

        with torch.no_grad():
            _, speed_preds, crossing_preds, intention_logits, _ = net(
                speed=speed,
                pos=pos,
                ped_attribute=ped_attribute,
                ped_behavior=ped_behavior,
                scene_attribute=scene_attribute,
                images=images,
                optical=optical,
                average=True,
            )
            speed_loss = huber(speed_preds, future_speed)

            crossing_loss = 0.0
            for i in range(future_cross.shape[1]):
                target_cross = torch.argmax(future_cross[:, i], dim=1)
                crossing_loss = crossing_loss + crossing_ce(crossing_preds[:, i], target_cross)
            crossing_loss = crossing_loss / future_cross.shape[1]
            intention_loss = intent_ce(intention_logits, label_c.long().view(-1))

            avg_epoch_eval_s_loss += float(speed_loss)
            avg_epoch_eval_c_loss += float(crossing_loss)
            avg_epoch_eval_i_loss += float(intention_loss)

            preds_p = utils.speed2pos(speed_preds, pos)
            ade += float(utils.ADE(preds_p, future_pos))
            fde += float(utils.FDE(preds_p, future_pos))
            aiou += float(utils.AIOU(preds_p, future_pos))
            fiou += float(utils.FIOU(preds_p, future_pos))

            future_cross_batch = torch.argmax(future_cross, dim=2).cpu().numpy()
            future_cross = future_cross_batch.reshape(-1)
            crossing_prob = torch.softmax(crossing_preds, dim=2)[:, :, 1].reshape(-1).detach().cpu().numpy()
            crossing_prob_seq = torch.softmax(crossing_preds, dim=2)[:, :, 1].detach().cpu().numpy()

            label_c = label_c.view(-1).cpu().numpy()
            intention_prob = torch.softmax(intention_logits, dim=1)[:, 1].detach().cpu().numpy()

            state_probs.extend(crossing_prob.tolist())
            state_targets.extend(future_cross.tolist())
            intent_probs.extend(intention_prob.tolist())
            intent_targets.extend(label_c.tolist())

            if args.save_sample_results:
                sample_rows.extend(
                    build_sample_rows(
                        args=args,
                        eval_set=eval_set,
                        start_index=sample_start_index,
                        batch_size=future_cross_batch.shape[0],
                        crossing_probs=crossing_prob_seq,
                        future_cross_np=future_cross_batch,
                        intent_probs=intention_prob,
                        intent_targets_np=label_c,
                    )
                )
                sample_start_index += future_cross_batch.shape[0]

            mean_eval_speed_loss = avg_epoch_eval_s_loss / counter
            mean_eval_cross_loss = avg_epoch_eval_c_loss / counter
            mean_eval_intent_loss = avg_epoch_eval_i_loss / counter
            if verbose and tqdm is not None:
                eval_iterator.set_postfix(
                    s=f'{mean_eval_speed_loss:.4f}',
                    c=f'{mean_eval_cross_loss:.4f}',
                    i=f'{mean_eval_intent_loss:.4f}',
                    ade=f'{ade / counter:.4f}',
                )
            else:
                log_batch_progress(
                    verbose,
                    f'eval[{args.dtype}]',
                    idx,
                    eval_batches,
                    speed_loss=mean_eval_speed_loss,
                    cross_loss=mean_eval_cross_loss,
                    intent_loss=mean_eval_intent_loss,
                    ade=ade / counter,
                )

    avg_epoch_eval_s_loss /= counter
    avg_epoch_eval_c_loss /= counter
    avg_epoch_eval_i_loss /= counter

    ade /= counter
    fde /= counter
    aiou /= counter
    fiou /= counter

    if saved_thresholds is not None:
        state_threshold, intent_threshold = saved_thresholds
        state_metrics = binary_classification_metrics(
            (np.asarray(state_probs) >= state_threshold).astype(np.int64),
            np.asarray(state_targets),
        )
        intent_metrics = binary_classification_metrics(
            (np.asarray(intent_probs) >= intent_threshold).astype(np.int64),
            np.asarray(intent_targets),
        )
        print(
            'Using saved validation thresholds: '
            f'state={state_threshold:.2f}, intent={intent_threshold:.2f}'
        )
    else:
        state_threshold, state_metrics = utils.find_best_binary_threshold(
            state_probs,
            state_targets,
            metric=args.threshold_metric,
        )
        intent_threshold, intent_metrics = utils.find_best_binary_threshold(
            intent_probs,
            intent_targets,
            metric=args.threshold_metric,
        )
        print(
            f'Swept thresholds on {args.dtype} split: '
            f'state={state_threshold:.2f}, intent={intent_threshold:.2f}, metric={args.threshold_metric}'
        )

    print(
        'Debug state confusion TP/FP/FN/TN:',
        state_metrics['tp'], state_metrics['fp'], state_metrics['fn'], state_metrics['tn']
    )
    print(
        'Debug intent confusion TP/FP/FN/TN:',
        intent_metrics['tp'], intent_metrics['fp'], intent_metrics['fn'], intent_metrics['tn']
    )
    print(
        '| ade: %.4f' % ade,
        '| fde: %.4f' % fde,
        '| aiou: %.4f' % aiou,
        '| fiou: %.4f' % fiou,
        '| state_acc: %.4f' % state_metrics['accuracy'],
        '| int_acc: %.4f' % intent_metrics['accuracy'],
        '| f1_int: %.4f' % intent_metrics['f1'],
        '| f1_state: %.4f' % state_metrics['f1'],
        '| pre: %.4f' % state_metrics['precision'],
        '| recall_sc: %.4f' % state_metrics['recall'],
        '| bal_sc: %.4f' % state_metrics['balanced_accuracy'],
        '| th_sc: %.2f' % state_threshold,
        '| pre_int: %.4f' % intent_metrics['precision'],
        '| recall_int: %.4f' % intent_metrics['recall'],
        '| bal_int: %.4f' % intent_metrics['balanced_accuracy'],
        '| th_int: %.2f' % intent_threshold,
        '| t:%.4f' % (time.time() - start),
    )

    if args.save_sample_results:
        export_sample_results(args, sample_rows, state_threshold, intent_threshold)


def main():
    args = parse_args()
    config = parse_config_file(os.path.join(os.path.dirname(__file__), 'config.yml'))
    if config.get('use_argument_parser') is False:
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])

    args.dtype = str(args.dtype).lower()

    if args.dataset == 'jaad':
        args.is_3D = False
    elif args.dataset == 'jta':
        args.is_3D = True
    elif args.dataset == 'nuscenes':
        args.is_3D = True
    else:
        print('Unknown dataset entered! Please select from available datasets: jaad, jta, nuscenes...')

    data_out_dir = resolve_artifact_dir(args, sequence_cache_name(args))
    eval_set = eval('datasets.' + args.dataset)(
        data_dir=args.data_dir,
        out_dir=data_out_dir,
        dtype=args.dtype,
        input=args.input,
        output=args.output,
        stride=args.stride,
        skip=args.skip,
        task=args.task,
        from_file=args.from_file,
        save=args.save,
        use_images=args.use_image,
        use_attribute=args.use_attribute,
        use_opticalflow=args.use_opticalflow
    )

    evaluate_2d(args, eval_set)


if __name__ == '__main__':
    main()
