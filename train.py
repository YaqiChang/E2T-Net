import time
import os
import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

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
    'train_loss_total',
    'val_loss_total',
    'ade',
    'fde',
    'aiou',
    'fiou',
    'state_acc',
    'state_f1',
    'intent_acc',
    'intent_f1',
]


def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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
    parser.add_argument('--crossing_pos_weight', type=float, default=4.0)

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
        return tqdm(iterable, desc=desc, total=total, leave=False)
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


def save_best_summary(path, epoch, checkpoint_file, ade, fde, aiou, fiou, state_metrics, intent_metrics):
    payload = {
        'best_epoch': int(epoch),
        'checkpoint': checkpoint_file,
        'ade': float(ade),
        'fde': float(fde),
        'aiou': float(aiou),
        'fiou': float(fiou),
        'state_acc': float(state_metrics['accuracy']),
        'state_f1': float(state_metrics['f1']),
        'intent_acc': float(intent_metrics['accuracy']),
        'intent_f1': float(intent_metrics['f1']),
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def train(args, train, val):
    use_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

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

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-7)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    train_sampler = DistributedSampler(train) if use_distributed else None
    dataloader_train = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=0,
        drop_last=True,
        sampler=train_sampler,
        shuffle=args.loader_shuffle if train_sampler is None else False,
    )

    mse = nn.MSELoss()
    class_weights = torch.tensor([1.0, args.crossing_pos_weight], device=device)
    crossing_ce = nn.CrossEntropyLoss(weight=class_weights)
    data = []
    best_ade = float('inf')
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard')) if verbose else None

    try:
        print('=' * 100)
        print('Training ...')
        print('Run directory: ' + run_dir)
        print('Learning rate: ' + str(args.lr))
        print('Number of epochs: ' + str(args.n_epochs))
        print('Hidden layer size: ' + str(args.hidden_size))
        print('Arguments:', args, '\n')

        for epoch in range(args.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            start = time.time()
            epoch_label = f'{epoch + 1}/{args.n_epochs}'

            avg_epoch_train_s_loss = 0
            avg_epoch_val_s_loss = 0
            avg_epoch_train_c_loss = 0
            avg_epoch_val_c_loss = 0
            avg_epoch_train_t_loss = 0
            avg_epoch_val_v_loss = 0

            ade = 0
            fde = 0
            aiou = 0
            fiou = 0

            counter = 0
            net.train()
            train_batches = len(dataloader_train)
            train_iterator = create_progress(
                enumerate(dataloader_train, start=1),
                desc=f'Epoch {epoch_label} [train]',
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

                net.zero_grad()
                mloss, speed_preds, crossing_preds = net(
                    speed=speed,
                    pos=pos,
                    ped_attribute=ped_attribute,
                    ped_behavior=ped_behavior,
                    scene_attribute=scene_attribute,
                    images=images,
                    optical=optical,
                    average=False,
                )

                speed_loss = mse(speed_preds, future_speed) / 100

                crossing_loss = 0
                for i in range(future_cross.shape[1]):
                    target_cross = torch.argmax(future_cross[:, i], dim=1)
                    crossing_loss += crossing_ce(crossing_preds[:, i], target_cross)

                crossing_loss /= future_cross.shape[1]

                loss = speed_loss + crossing_loss + mloss
                loss.backward()
                optimizer.step()

                avg_epoch_train_s_loss += float(speed_loss)
                avg_epoch_train_c_loss += float(crossing_loss)
                avg_epoch_train_t_loss += float(loss)

                mean_train_speed_loss = avg_epoch_train_s_loss / counter
                mean_train_cross_loss = avg_epoch_train_c_loss / counter
                mean_train_total_loss = avg_epoch_train_t_loss / counter
                if verbose and tqdm is not None:
                    train_iterator.set_postfix(
                        speed_loss=f'{mean_train_speed_loss:.4f}',
                        cross_loss=f'{mean_train_cross_loss:.4f}',
                        total_loss=f'{mean_train_total_loss:.4f}',
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
                        total_loss=mean_train_total_loss,
                    )
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            avg_epoch_train_s_loss /= counter
            avg_epoch_train_c_loss /= counter
            avg_epoch_train_t_loss /= counter
            if writer:
                writer.add_scalar("Loss_speed/train", avg_epoch_train_s_loss, epoch)
                writer.add_scalar("Loss_crossing/train", avg_epoch_train_c_loss, epoch)
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
            intent_preds = []
            intent_targets = []

            net.eval()
            val_batches = len(dataloader_val)
            val_iterator = create_progress(
                enumerate(dataloader_val, start=1),
                desc=f'Epoch {epoch_label} [eval]',
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
                    vloss, speed_preds, crossing_preds, intentions = net(
                        speed=speed,
                        pos=pos,
                        ped_attribute=ped_attribute,
                        ped_behavior=ped_behavior,
                        scene_attribute=scene_attribute,
                        images=images,
                        optical=optical,
                        average=True,
                    )
                    speed_loss_v = mse(speed_preds, future_speed) / 100

                    crossing_loss_v = 0
                    for i in range(future_cross.shape[1]):
                        target_cross = torch.argmax(future_cross[:, i], dim=1)
                        crossing_loss_v += crossing_ce(crossing_preds[:, i], target_cross)
                    crossing_loss_v /= future_cross.shape[1]

                    avg_epoch_val_s_loss += float(speed_loss_v)
                    avg_epoch_val_c_loss += float(crossing_loss_v)
                    avg_epoch_val_v_loss += float(vloss)

                    preds_p = utils.speed2pos(speed_preds, pos)
                    ade += float(utils.ADE(preds_p, future_pos))
                    fde += float(utils.FDE(preds_p, future_pos))
                    aiou += float(utils.AIOU(preds_p, future_pos))
                    fiou += float(utils.FIOU(preds_p, future_pos))

                    future_cross = torch.argmax(future_cross, dim=2).view(-1).cpu().numpy()
                    crossing_preds = np.argmax(crossing_preds.view(-1, 2).detach().cpu().numpy(), axis=1)
                    batch_state_acc = float(np.mean(crossing_preds == future_cross))

                    label_c = label_c.view(-1).cpu().numpy()
                    intentions = intentions.view(-1).detach().cpu().numpy()

                    state_preds.extend(crossing_preds)
                    state_targets.extend(future_cross)
                    intent_preds.extend(intentions)
                    intent_targets.extend(label_c)

                    mean_val_speed_loss = avg_epoch_val_s_loss / counter
                    mean_val_cross_loss = avg_epoch_val_c_loss / counter
                    if verbose and tqdm is not None:
                        val_iterator.set_postfix(
                            speed_loss=f'{mean_val_speed_loss:.4f}',
                            cross_loss=f'{mean_val_cross_loss:.4f}',
                            ade=f'{ade / counter:.4f}',
                            state_acc=f'{batch_state_acc:.4f}',
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
                            ade=ade / counter,
                            state_acc=batch_state_acc,
                        )
                    if device.type == 'cuda':
                        torch.cuda.synchronize()

            avg_epoch_val_s_loss /= counter
            avg_epoch_val_c_loss /= counter

            ade /= counter
            fde /= counter
            aiou /= counter
            fiou /= counter

            avg_epoch_val_t_loss = avg_epoch_val_s_loss + avg_epoch_val_c_loss + (avg_epoch_val_v_loss / counter)

            if writer:
                writer.add_scalar("Loss_speed/val", avg_epoch_val_s_loss, epoch)
                writer.add_scalar("Loss_crossing/val", avg_epoch_val_c_loss, epoch)
                writer.add_scalar("Loss/val", avg_epoch_val_t_loss, epoch)

            state_metrics = binary_classification_metrics(state_preds, state_targets)
            intent_metrics = binary_classification_metrics(intent_preds, intent_targets)

            data.append([
                epoch + 1,
                avg_epoch_train_s_loss,
                avg_epoch_val_s_loss,
                avg_epoch_train_c_loss,
                avg_epoch_val_c_loss,
                avg_epoch_train_t_loss,
                avg_epoch_val_t_loss,
                ade,
                fde,
                aiou,
                fiou,
                state_metrics['accuracy'],
                state_metrics['f1'],
                intent_metrics['accuracy'],
                intent_metrics['f1'],
            ])

            if args.save and verbose:
                save_metrics_table(data, os.path.join(run_dir, metrics_filename(args)))

            if args.lr_scheduler:
                scheduler.step(avg_epoch_train_t_loss)

            if ade < best_ade and verbose:
                best_ade = ade
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
                )
                print(f'Updated best checkpoint: {best_checkpoint} (epoch {epoch + 1}, ADE {ade:.4f})')

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
                    '| pre_int: %.4f' % intent_metrics['precision'],
                    '| recall_int: %.4f' % intent_metrics['recall'],
                    '| t: %.4f' % (time.time() - start),
                    )

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
