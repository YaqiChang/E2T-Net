import time
import os
import argparse
import json
import sys
from ast import literal_eval

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import WeightedRandomSampler

import yaml
import numpy as np
import pandas as pd
import datetime
import datasets
import model.network_image as network
import utils
from utils import binary_classification_metrics
from torch.utils.tensorboard import SummaryWriter
import visualization.display as viz

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


METRIC_COLUMNS = [
    'epoch',
    'train_loss_s',
    'val_loss_s',
    'train_loss_c',
    'val_loss_c',
    'train_loss_i',
    'val_loss_i',
    'train_loss_m',
    'val_loss_m',
    'train_loss_total',
    'val_loss_total',
    'ade',
    'fde',
    'aiou',
    'fiou',
    'state_acc',
    'state_f1',
    'state_recall',
    'state_bal_acc',
    'state_threshold',
    'state_pred_pos_rate',
    'state_true_pos_rate',
    'intent_acc',
    'intent_f1',
    'intent_recall',
    'intent_bal_acc',
    'intent_threshold',
    'intent_pred_pos_rate',
    'intent_true_pos_rate',
    'selection_score',
]


def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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


def parse_args():
    parser = argparse.ArgumentParser(description='Train PTINet network')

    parser.add_argument('--data_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/JAAD/PN/',
                        required=False)
    parser.add_argument('--dataset', type=str,
                        default='pie',
                        required=False)
    parser.add_argument('--out_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/PIE_bbox_image/bounding-box-prediction/output',
                        required=False)

    parser.add_argument('--input', type=int,
                        default=16,
                        required=False)
    parser.add_argument('--output', type=int,
                        default=32,
                        required=False)
    parser.add_argument('--stride', type=int,
                        default=16,
                        required=False)
    parser.add_argument('--skip', type=int, default=1)

    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default=False)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=16)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--prefetch_factor', type=int, default=3)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--crossing_pos_weight', type=float, default=1.0)
    parser.add_argument('--intent_pos_weight', type=float, default=1.0)
    parser.add_argument('--intention_loss_weight', type=float, default=1.0)
    parser.add_argument('--vae_loss_weight', type=float, default=1e-5)
    parser.add_argument('--auto_class_weights', type=bool, default=False)
    parser.add_argument('--max_class_weight', type=float, default=1.0)
    parser.add_argument('--use_balanced_sampler', type=bool, default=True)
    parser.add_argument('--sampler_target_pos_rate', type=float, default=0.3)
    parser.add_argument('--threshold_metric', type=str, default='f0.5')
    parser.add_argument('--abort_on_collapse', type=bool, default=True)
    parser.add_argument('--collapse_patience_epochs', type=int, default=10)
    parser.add_argument('--collapse_min_state_f1', type=float, default=0.05)
    parser.add_argument('--collapse_min_intent_f1', type=float, default=0.05)
    parser.add_argument('--collapse_max_state_recall', type=float, default=0.01)
    parser.add_argument('--collapse_max_intent_recall', type=float, default=0.01)
    parser.add_argument('--best_metric_f1_weight', type=float, default=1.0)
    parser.add_argument('--best_metric_recall_weight', type=float, default=1.0)
    parser.add_argument('--best_metric_precision_weight', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--use_plateau_scheduler', type=bool, default=True)
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--plateau_patience', type=int, default=4)
    parser.add_argument('--plateau_min_lr', type=float, default=1e-6)
    parser.add_argument('--use_early_stopping', type=bool, default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=12)
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4)

    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--use_image', type=bool, default=False,
                        help='Use input image as a feature',
                        required=False)
    parser.add_argument('--image_network', type=str, default='resnet50',
                        help='select backbone',
                        required=False)
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='Use input attribute as a feature',
                        required=False)
    parser.add_argument('--use_opticalflow', type=bool, default=False,
                        help='Use input emdedding as a feature',
                        required=False)

    args = parser.parse_args()

    return args


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


def log_batch_progress(verbose, phase, epoch, total_epochs, batch_idx, total_batches, **metrics):
    if not verbose or tqdm is not None or total_batches == 0:
        return

    should_print = batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 10) == 0
    if not should_print:
        return

    metric_text = ' | '.join(f'{name}: {value:.4f}' for name, value in metrics.items())
    print(f'[{phase}] epoch {epoch}/{total_epochs} batch {batch_idx}/{total_batches}' + (f' | {metric_text}' if metric_text else ''))


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)


def resolve_progress_stream(stream):
    if hasattr(stream, 'streams'):
        for child_stream in stream.streams:
            if getattr(child_stream, 'isatty', lambda: False)():
                return child_stream
        if stream.streams:
            return stream.streams[0]
    return stream


def get_run_dir(args):
    return os.path.join(args.out_dir, args.log_name) if args.log_name else args.out_dir


def checkpoint_suffix(args):
    return '_scheduler' if args.lr_scheduler else ''


def checkpoint_name(args, kind, epoch=None):
    suffix = checkpoint_suffix(args)
    if kind == 'epoch':
        if epoch is None:
            raise ValueError('epoch is required for epoch checkpoints')
        return f'model_epoch_{epoch:03d}{suffix}.pkl'
    if kind == 'best':
        return f'model_best{suffix}.pkl'
    if kind == 'final':
        return f'model_final{suffix}.pkl'
    raise ValueError(f'Unknown checkpoint kind: {kind}')


def metrics_filename(args):
    return f'train_metrics{checkpoint_suffix(args)}.csv'


def best_summary_filename(args):
    return f'best_metrics{checkpoint_suffix(args)}.json'


def save_model_state(model, path):
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(model_to_save.state_dict(), path)


def save_metrics_table(rows, path):
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df.to_csv(path, index=False)
    return df


def save_best_summary(path, epoch, checkpoint_file, ade, fde, aiou, fiou, state_metrics, intent_metrics, selection_score):
    payload = {
        'best_epoch': int(epoch),
        'checkpoint': checkpoint_file,
        'ade': float(ade),
        'fde': float(fde),
        'aiou': float(aiou),
        'fiou': float(fiou),
        'selection_score': float(selection_score),
        'state_acc': float(state_metrics['accuracy']),
        'state_precision': float(state_metrics['precision']),
        'state_f1': float(state_metrics['f1']),
        'state_f0_5': float(state_metrics.get('f0_5', 0.0)),
        'state_recall': float(state_metrics['recall']),
        'state_bal_acc': float(state_metrics['balanced_accuracy']),
        'state_threshold': float(state_metrics.get('threshold', 0.5)),
        'intent_acc': float(intent_metrics['accuracy']),
        'intent_precision': float(intent_metrics['precision']),
        'intent_f1': float(intent_metrics['f1']),
        'intent_f0_5': float(intent_metrics.get('f0_5', 0.0)),
        'intent_recall': float(intent_metrics['recall']),
        'intent_bal_acc': float(intent_metrics['balanced_accuracy']),
        'intent_threshold': float(intent_metrics.get('threshold', 0.5)),
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def detect_metric_collapse(args, epoch, state_metrics, intent_metrics):
    if not args.abort_on_collapse:
        return None
    if epoch < args.collapse_patience_epochs:
        return None

    state_collapsed = (
        state_metrics['f1'] <= args.collapse_min_state_f1 and
        state_metrics['recall'] <= args.collapse_max_state_recall
    )
    intent_collapsed = (
        intent_metrics['f1'] <= args.collapse_min_intent_f1 and
        intent_metrics['recall'] <= args.collapse_max_intent_recall
    )
    if not (state_collapsed and intent_collapsed):
        return None

    return (
        f'Training collapse detected at epoch {epoch}: '
        f'state_f1={state_metrics["f1"]:.4f}, state_recall={state_metrics["recall"]:.4f}, '
        f'intent_f1={intent_metrics["f1"]:.4f}, intent_recall={intent_metrics["recall"]:.4f}. '
        'Model is still predicting almost no positive samples.'
    )


def safe_positive_class_weight(positive_count, total_count, max_weight):
    positive_count = float(positive_count)
    total_count = float(total_count)
    if total_count <= 0:
        return 1.0
    if positive_count <= 0:
        return float(max_weight)
    negative_count = max(total_count - positive_count, 1.0)
    return float(min(max_weight, max(1.0, negative_count / positive_count)))


def parse_sequence_values(value):
    if isinstance(value, str):
        return literal_eval(value)
    return value


def infer_dataset_class_weights(dataset, max_weight):
    if not hasattr(dataset, 'data'):
        return None, None

    df = dataset.data
    if 'crossing_true' not in df.columns or 'label' not in df.columns:
        return None, None

    crossing_positive = 0.0
    crossing_total = 0.0
    for sequence in df['crossing_true']:
        parsed = np.asarray(parse_sequence_values(sequence), dtype=np.float32).reshape(-1)
        crossing_positive += float(parsed.sum())
        crossing_total += float(parsed.size)

    label_values = df['label'].apply(parse_sequence_values).astype(np.float32).to_numpy()
    intent_positive = float(label_values.sum())
    intent_total = float(label_values.size)

    return (
        safe_positive_class_weight(crossing_positive, crossing_total, max_weight),
        safe_positive_class_weight(intent_positive, intent_total, max_weight),
    )


def build_balanced_sampler(dataset, target_pos_rate=0.5):
    if not hasattr(dataset, 'data') or 'label' not in dataset.data.columns:
        return None

    labels = dataset.data['label'].apply(parse_sequence_values).astype(np.float32).to_numpy()
    positive_count = float(labels.sum())
    negative_count = float(labels.size - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        return None

    target_pos_rate = float(np.clip(target_pos_rate, 1e-3, 1.0 - 1e-3))
    positive_weight = target_pos_rate / positive_count
    negative_weight = (1.0 - target_pos_rate) / negative_count
    sample_weights = np.where(labels > 0, positive_weight, negative_weight).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_selection_score(args, ade, state_metrics, intent_metrics):
    classification_score = (
        args.best_metric_f1_weight * (state_metrics['f0_5'] + intent_metrics['f0_5'])
        + args.best_metric_precision_weight * (
            state_metrics['precision'] + intent_metrics['precision']
        )
        + args.best_metric_recall_weight * (
            state_metrics['balanced_accuracy'] + intent_metrics['balanced_accuracy']
        )
    )
    return classification_score - 0.01 * float(ade)


def train(args, train, val):
    use_distributed = should_use_distributed()
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = resolve_runtime_device(args.device, local_rank)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    args.device = str(device)
    args.crossing_pos_weight = 1.0
    args.intent_pos_weight = 1.0
    args.auto_class_weights = False

    verbose = dist.get_rank() == 0 if use_distributed else True
    run_dir = get_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)

    log_file = None
    original_stdout = None
    original_stderr = None
    if verbose:
        log_file = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(sys.stdout, log_file)
        sys.stderr = TeeStream(sys.stderr, log_file)

    net = network.PTINet(args).to(device)
    if use_distributed:
        net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_plateau_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    if use_distributed:
        train_sampler = DistributedSampler(train)
    elif args.use_balanced_sampler:
        train_sampler = build_balanced_sampler(train, args.sampler_target_pos_rate)
    else:
        train_sampler = None
    dataloader_train = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=0,
        drop_last=True,
        sampler=train_sampler,
        shuffle=args.loader_shuffle if train_sampler is None else False,
    )

    huber = nn.HuberLoss(delta=1.0)
    crossing_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    intent_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    data = []
    best_selection_score = float('-inf')
    epochs_without_improvement = 0
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard')) if verbose else None

    try:
        print('=' * 100)
        print('Training ...')
        print('Run directory: ' + run_dir)
        print('Learning rate: ' + str(args.lr))
        print('Number of epochs: ' + str(args.n_epochs))
        print('Hidden layer size: ' + str(args.hidden_size))
        print('Crossing loss class weight: 1.0000')
        print('Intent loss class weight: 1.0000')
        print('VAE loss weight: %.6f' % args.vae_loss_weight)
        print('Weight decay: %.6f' % args.weight_decay)
        print('Gradient clip norm: %.2f' % args.grad_clip_norm)
        print('Label smoothing: %.3f' % args.label_smoothing)
        print('Plateau scheduler: ' + str(args.use_plateau_scheduler))
        print('Early stopping: ' + str(args.use_early_stopping))
        print('Balanced sampler: ' + str(train_sampler is not None and not use_distributed))
        print('Sampler target positive rate: %.3f' % args.sampler_target_pos_rate)
        print('Threshold metric: ' + args.threshold_metric)
        print('Arguments:', args, '\n')

        for epoch in range(args.n_epochs):
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            start = time.time()
            epoch_label = f'{epoch + 1}/{args.n_epochs}'

            avg_epoch_train_s_loss = 0
            avg_epoch_val_s_loss = 0
            avg_epoch_train_c_loss = 0
            avg_epoch_val_c_loss = 0
            avg_epoch_train_i_loss = 0
            avg_epoch_val_i_loss = 0
            avg_epoch_train_m_loss = 0
            avg_epoch_train_t_loss = 0
            avg_epoch_val_v_loss = 0

            ade = 0
            fde = 0
            aiou = 0
            fiou = 0

            train_state_pred_positive = 0
            train_state_target_positive = 0
            train_state_total = 0
            train_intent_pred_positive = 0
            train_intent_target_positive = 0
            train_intent_total = 0

            counter = 0
            net.train()
            train_batches = len(dataloader_train)
            train_iterator = create_progress(
                enumerate(dataloader_train, start=1),
                desc=f'Train {epoch_label}',
                total=train_batches,
                verbose=verbose,
            )
            for idx, inputs in train_iterator:
                counter += 1
                speed = inputs['speed'].to(device, non_blocking=True)
                future_speed = inputs['future_speed'].to(device, non_blocking=True)
                pos = inputs['pos'].to(device, non_blocking=True)
                future_pos = inputs['future_pos'].to(device, non_blocking=True)
                future_cross = inputs['future_cross'].to(device, non_blocking=True)
                optical = inputs['optical'].to(device, non_blocking=True)
                ped_behavior = inputs['ped_behavior'].to(device, non_blocking=True)
                images = inputs['image'].to(device, non_blocking=True)
                ped_attribute = inputs['ped_attribute'].to(device, non_blocking=True)
                scene_attribute = inputs['scene_attribute'].to(device, non_blocking=True)
                label_c = inputs['cross_label'].to(device, non_blocking=True).long().view(-1)

                net.zero_grad()
                mloss, speed_preds, crossing_preds, intention_logits = net(
                    speed=speed,
                    pos=pos,
                    ped_attribute=ped_attribute,
                    ped_behavior=ped_behavior,
                    scene_attribute=scene_attribute,
                    images=images,
                    optical=optical,
                    average=False,
                )

                speed_loss = huber(speed_preds, future_speed)

                crossing_loss = 0
                for i in range(future_cross.shape[1]):
                    target_cross = torch.argmax(future_cross[:, i], dim=1)
                    crossing_loss += crossing_ce(crossing_preds[:, i], target_cross)

                crossing_loss /= future_cross.shape[1]
                intention_loss = intent_ce(intention_logits, label_c)
                weighted_mloss = args.vae_loss_weight * mloss
                loss = speed_loss + crossing_loss + args.intention_loss_weight * intention_loss + weighted_mloss
                loss.backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
                optimizer.step()

                avg_epoch_train_s_loss += float(speed_loss)
                avg_epoch_train_c_loss += float(crossing_loss)
                avg_epoch_train_i_loss += float(intention_loss)
                avg_epoch_train_m_loss += float(mloss)
                avg_epoch_train_t_loss += float(loss)

                batch_state_targets = torch.argmax(future_cross, dim=2).view(-1)
                batch_state_preds = torch.argmax(crossing_preds.detach(), dim=2).view(-1)
                batch_intent_preds = torch.argmax(intention_logits.detach(), dim=1).view(-1)
                train_state_pred_positive += int(batch_state_preds.sum().item())
                train_state_target_positive += int(batch_state_targets.sum().item())
                train_state_total += int(batch_state_targets.numel())
                train_intent_pred_positive += int(batch_intent_preds.sum().item())
                train_intent_target_positive += int(label_c.sum().item())
                train_intent_total += int(label_c.numel())

                mean_train_speed_loss = avg_epoch_train_s_loss / counter
                mean_train_cross_loss = avg_epoch_train_c_loss / counter
                mean_train_intent_loss = avg_epoch_train_i_loss / counter
                mean_train_m_loss = avg_epoch_train_m_loss / counter
                mean_train_total_loss = avg_epoch_train_t_loss / counter
                if verbose and tqdm is not None:
                    train_iterator.set_postfix(
                        s=f'{mean_train_speed_loss:.4f}',
                        c=f'{mean_train_cross_loss:.4f}',
                        i=f'{mean_train_intent_loss:.4f}',
                        m=f'{mean_train_m_loss:.1f}',
                        t=f'{mean_train_total_loss:.4f}',
                    )
                else:
                    log_batch_progress(
                        verbose,
                        'train',
                        epoch + 1,
                        args.n_epochs,
                        idx,
                        train_batches,
                        speed_loss=mean_train_speed_loss,
                        cross_loss=mean_train_cross_loss,
                        intent_loss=mean_train_intent_loss,
                        vae_loss=mean_train_m_loss,
                        total_loss=mean_train_total_loss,
                    )
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            avg_epoch_train_s_loss /= counter
            avg_epoch_train_c_loss /= counter
            avg_epoch_train_i_loss /= counter
            avg_epoch_train_m_loss /= counter
            avg_epoch_train_t_loss /= counter
            train_state_pred_pos_rate = train_state_pred_positive / train_state_total if train_state_total else 0.0
            train_state_true_pos_rate = train_state_target_positive / train_state_total if train_state_total else 0.0
            train_intent_pred_pos_rate = train_intent_pred_positive / train_intent_total if train_intent_total else 0.0
            train_intent_true_pos_rate = train_intent_target_positive / train_intent_total if train_intent_total else 0.0
            if writer:
                writer.add_scalar("Loss_speed/train", avg_epoch_train_s_loss, epoch)
                writer.add_scalar("Loss_crossing/train", avg_epoch_train_c_loss, epoch)
                writer.add_scalar("Loss_intention/train", avg_epoch_train_i_loss, epoch)
                writer.add_scalar("Loss_vae/train", avg_epoch_train_m_loss, epoch)
                writer.add_scalar("Loss/train", avg_epoch_train_t_loss, epoch)

            if args.save and verbose:
                epoch_checkpoint = checkpoint_name(args, 'epoch', epoch + 1)
                save_model_state(net, os.path.join(run_dir, epoch_checkpoint))
                print(f'Saved epoch checkpoint: {epoch_checkpoint}')

            val_sampler = DistributedSampler(val) if use_distributed else None
            dataloader_val = torch.utils.data.DataLoader(
                val,
                batch_size=args.batch_size,
                pin_memory=args.pin_memory,
                num_workers=0,
                drop_last=True,
                sampler=val_sampler,
                shuffle=False,
            )

            counter = 0
            state_preds = []
            state_targets = []
            state_probs = []
            intent_preds = []
            intent_targets = []
            intent_probs = []

            net.eval()
            val_batches = len(dataloader_val)
            val_iterator = create_progress(
                enumerate(dataloader_val, start=1),
                desc=f'Eval  {epoch_label}',
                total=val_batches,
                verbose=verbose,
            )
            for idx, val_in in val_iterator:
                counter += 1

                speed = val_in['speed'].to(device, non_blocking=True)
                future_speed = val_in['future_speed'].to(device, non_blocking=True)
                pos = val_in['pos'].to(device, non_blocking=True)
                future_pos = val_in['future_pos'].to(device, non_blocking=True)
                future_cross = val_in['future_cross'].to(device, non_blocking=True)
                ped_attribute = val_in['ped_attribute'].to(device, non_blocking=True)
                scene_attribute = val_in['scene_attribute'].to(device, non_blocking=True)
                optical = val_in['optical'].to(device, non_blocking=True)
                ped_behavior = val_in['ped_behavior'].to(device, non_blocking=True)
                images = val_in['image'].to(device, non_blocking=True)
                label_c = val_in['cross_label'].to(device, non_blocking=True)

                with torch.no_grad():
                    vloss, speed_preds, crossing_preds, intention_logits, intentions = net(
                        speed=speed,
                        pos=pos,
                        ped_attribute=ped_attribute,
                        ped_behavior=ped_behavior,
                        scene_attribute=scene_attribute,
                        images=images,
                        optical=optical,
                        average=True,
                    )
                    speed_loss_v = huber(speed_preds, future_speed)

                    crossing_loss_v = 0
                    for i in range(future_cross.shape[1]):
                        target_cross = torch.argmax(future_cross[:, i], dim=1)
                        crossing_loss_v += crossing_ce(crossing_preds[:, i], target_cross)
                    crossing_loss_v /= future_cross.shape[1]
                    intention_loss_v = intent_ce(intention_logits, label_c.long().view(-1))

                    avg_epoch_val_s_loss += float(speed_loss_v)
                    avg_epoch_val_c_loss += float(crossing_loss_v)
                    avg_epoch_val_i_loss += float(intention_loss_v)
                    avg_epoch_val_v_loss += float(vloss)

                    preds_p = utils.speed2pos(speed_preds, pos)
                    ade += float(utils.ADE(preds_p, future_pos))
                    fde += float(utils.FDE(preds_p, future_pos))
                    aiou += float(utils.AIOU(preds_p, future_pos))
                    fiou += float(utils.FIOU(preds_p, future_pos))

                    future_cross = torch.argmax(future_cross, dim=2).view(-1).cpu().numpy()
                    crossing_prob = torch.softmax(crossing_preds, dim=2)[:, :, 1].reshape(-1).detach().cpu().numpy()
                    provisional_state_pred = (crossing_prob >= 0.5).astype(np.int64)
                    batch_state_acc = float(np.mean(provisional_state_pred == future_cross))

                    label_c = label_c.view(-1).cpu().numpy()
                    intention_prob = torch.softmax(intention_logits, dim=1)[:, 1].detach().cpu().numpy()
                    provisional_intent_pred = (intention_prob >= 0.5).astype(np.int64)

                    state_probs.extend(crossing_prob.tolist())
                    state_targets.extend(future_cross.tolist())
                    intent_probs.extend(intention_prob.tolist())
                    intent_targets.extend(label_c.tolist())

                    mean_val_speed_loss = avg_epoch_val_s_loss / counter
                    mean_val_cross_loss = avg_epoch_val_c_loss / counter
                    mean_val_intent_loss = avg_epoch_val_i_loss / counter
                    mean_val_m_loss = avg_epoch_val_v_loss / counter
                    if verbose and tqdm is not None:
                        val_iterator.set_postfix(
                            s=f'{mean_val_speed_loss:.4f}',
                            c=f'{mean_val_cross_loss:.4f}',
                            i=f'{mean_val_intent_loss:.4f}',
                            m=f'{mean_val_m_loss:.1f}',
                            ade=f'{ade / counter:.4f}',
                            sa=f'{batch_state_acc:.4f}',
                        )
                    else:
                        log_batch_progress(
                            verbose,
                            'eval',
                            epoch + 1,
                            args.n_epochs,
                            idx,
                            val_batches,
                            speed_loss=mean_val_speed_loss,
                            cross_loss=mean_val_cross_loss,
                            intent_loss=mean_val_intent_loss,
                            vae_loss=mean_val_m_loss,
                            ade=ade / counter,
                            state_acc=batch_state_acc,
                        )
                    if device.type == 'cuda':
                        torch.cuda.synchronize()

            avg_epoch_val_s_loss /= counter
            avg_epoch_val_c_loss /= counter
            avg_epoch_val_i_loss /= counter
            avg_epoch_val_v_loss /= counter

            ade /= counter
            fde /= counter
            aiou /= counter
            fiou /= counter

            avg_epoch_val_t_loss = (
                avg_epoch_val_s_loss
                + avg_epoch_val_c_loss
                + args.intention_loss_weight * avg_epoch_val_i_loss
                + args.vae_loss_weight * avg_epoch_val_v_loss
            )

            if writer:
                writer.add_scalar("Loss_speed/val", avg_epoch_val_s_loss, epoch)
                writer.add_scalar("Loss_crossing/val", avg_epoch_val_c_loss, epoch)
                writer.add_scalar("Loss_intention/val", avg_epoch_val_i_loss, epoch)
                writer.add_scalar("Loss_vae/val", avg_epoch_val_v_loss, epoch)
                writer.add_scalar("Loss/val", avg_epoch_val_t_loss, epoch)

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
            state_preds = (np.asarray(state_probs) >= state_threshold).astype(np.int64)
            intent_preds = (np.asarray(intent_probs) >= intent_threshold).astype(np.int64)
            selection_score = compute_selection_score(args, ade, state_metrics, intent_metrics)
            state_pred_pos_rate = float(np.mean(np.asarray(state_preds) == 1)) if len(state_preds) else 0.0
            state_true_pos_rate = float(np.mean(np.asarray(state_targets) == 1)) if len(state_targets) else 0.0
            intent_pred_pos_rate = float(np.mean(np.asarray(intent_preds) == 1)) if len(intent_preds) else 0.0
            intent_true_pos_rate = float(np.mean(np.asarray(intent_targets) == 1)) if len(intent_targets) else 0.0

            data.append([
                epoch + 1,
                avg_epoch_train_s_loss,
                avg_epoch_val_s_loss,
                avg_epoch_train_c_loss,
                avg_epoch_val_c_loss,
                avg_epoch_train_i_loss,
                avg_epoch_val_i_loss,
                avg_epoch_train_m_loss,
                avg_epoch_val_v_loss,
                avg_epoch_train_t_loss,
                avg_epoch_val_t_loss,
                ade,
                fde,
                aiou,
                fiou,
                state_metrics['accuracy'],
                state_metrics['f1'],
                state_metrics['recall'],
                state_metrics['balanced_accuracy'],
                state_threshold,
                state_pred_pos_rate,
                state_true_pos_rate,
                intent_metrics['accuracy'],
                intent_metrics['f1'],
                intent_metrics['recall'],
                intent_metrics['balanced_accuracy'],
                intent_threshold,
                intent_pred_pos_rate,
                intent_true_pos_rate,
                selection_score,
            ])

            if args.save and verbose:
                save_metrics_table(data, os.path.join(run_dir, metrics_filename(args)))

            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(avg_epoch_val_t_loss)
                current_lr = optimizer.param_groups[0]['lr']

            if selection_score > best_selection_score and verbose:
                best_selection_score = selection_score
                epochs_without_improvement = 0
                best_checkpoint = checkpoint_name(args, 'best')
                save_model_state(net, os.path.join(run_dir, best_checkpoint))
                save_best_summary(
                    os.path.join(run_dir, best_summary_filename(args)),
                    epoch + 1,
                    best_checkpoint,
                    ade,
                    fde,
                    aiou,
                    fiou,
                    state_metrics,
                    intent_metrics,
                    selection_score,
                )
                print(
                    f'Updated best checkpoint: {best_checkpoint} '
                    f'(epoch {epoch + 1}, score {selection_score:.4f}, ADE {ade:.4f}, '
                    f'state_f1 {state_metrics["f1"]:.4f}@{state_threshold:.2f}, '
                    f'intent_f1 {intent_metrics["f1"]:.4f}@{intent_threshold:.2f})'
                )
            else:
                epochs_without_improvement += 1

            if verbose:
                print(
                    'Debug state confusion TP/FP/FN/TN:',
                    state_metrics['tp'], state_metrics['fp'], state_metrics['fn'], state_metrics['tn']
                )
                print(
                    'Debug intent confusion TP/FP/FN/TN:',
                    intent_metrics['tp'], intent_metrics['fp'], intent_metrics['fn'], intent_metrics['tn']
                )
                print('Epoch:', epoch + 1,
                    '| ade: %.4f' % ade,
                    '| fde: %.4f' % fde,
                    '| aiou: %.4f' % aiou,
                    '| fiou: %.4f' % fiou,
                    '| state_acc: %.4f' % state_metrics['accuracy'],
                    '| intention_acc: %.4f' % intent_metrics['accuracy'],
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
                    '| train_sc_pos: %.4f' % train_state_pred_pos_rate,
                    '| train_sc_true: %.4f' % train_state_true_pos_rate,
                    '| train_int_pos: %.4f' % train_intent_pred_pos_rate,
                    '| train_int_true: %.4f' % train_intent_true_pos_rate,
                    '| val_sc_pos: %.4f' % state_pred_pos_rate,
                    '| val_sc_true: %.4f' % state_true_pos_rate,
                    '| val_int_pos: %.4f' % intent_pred_pos_rate,
                    '| val_int_true: %.4f' % intent_true_pos_rate,
                    '| train_m: %.1f' % avg_epoch_train_m_loss,
                    '| val_m: %.1f' % avg_epoch_val_v_loss,
                    '| score: %.4f' % selection_score,
                    '| lr: %.6f' % current_lr,
                    '| no_improve: %d' % epochs_without_improvement,
                    '| t: %.4f' % (time.time() - start),
                    )

            collapse_message = detect_metric_collapse(args, epoch + 1, state_metrics, intent_metrics)
            if collapse_message:
                print('ERROR:', collapse_message)
                raise RuntimeError(collapse_message)

            if (
                args.use_early_stopping and
                epochs_without_improvement >= args.early_stopping_patience and
                selection_score <= best_selection_score + args.early_stopping_min_delta
            ):
                if verbose:
                    print(
                        'Early stopping triggered at epoch %d after %d epochs without selection score improvement.' %
                        (epoch + 1, epochs_without_improvement)
                    )
                break

        if args.save and verbose:
            save_metrics_table(data, os.path.join(run_dir, metrics_filename(args)))
            final_checkpoint = checkpoint_name(args, 'final')
            save_model_state(net, os.path.join(run_dir, final_checkpoint))
            print(f'Saved final checkpoint: {final_checkpoint}')
            print('Training artifacts saved to {}\n'.format(run_dir))

        print('=' * 100)
        print('Done !')
    finally:
        if writer:
            writer.close()
        if log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == '__main__':

    print("Date and time:", datetime.datetime.now())

    args = parse_args()
    config = parse_config_file(os.path.join(os.path.dirname(__file__), 'config.yml'))
    if config.get('use_argument_parser') == False:
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])
    print(args)

    os.makedirs(get_run_dir(args), exist_ok=True)

    train_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=get_run_dir(args),
                dtype='train',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute,
                use_opticalflow=args.use_opticalflow
                )

    val_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=get_run_dir(args),
                dtype='val',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                from_file=args.from_file,
                save=args.save,
                use_images=args.use_image,
                use_attribute=args.use_attribute,
                use_opticalflow=args.use_opticalflow
                )

    train(args, train_set, val_set)
