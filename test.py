import time
import os
import argparse

import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torchvision
# import torchvision.transforms as transforms
    
import numpy as np

# import DataLoader
import datasets
import model.network_image as network
import utils
from utils import data_loader, binary_classification_metrics
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    
    parser.add_argument('--data_dir', type=str,
                        default= '/scratch/project_2007864/PIE/PN/',
                        required=False)
    parser.add_argument('--dataset', type=str, 
                        default='pie',
                        required=False)
    parser.add_argument('--out_dir', type=str, 
                        default= '/projappl/project_2007864/PIE_lstm_vae_clstm/bounding-box-prediction/output_32/',
                        required=False)  
    parser.add_argument('--task', type=str, 
                        default='2D_bounding_box-intention',
                        required=False)
    
    # data configuration
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
    parser.add_argument('--is_3D', type=bool, default=False) 
    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='val')
    parser.add_argument("--from_file", type=bool, default=False)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Checkpoint file to evaluate. Accepts .pkl or .pth.')
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--prefetch_factor', type=int, default=3)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=int, default= 0.001)
    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--crossing_pos_weight', type=float, default=4.0)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)
    parser.add_argument('--use_image', type=bool, default=True,
                        help='Use input image as a feature',
                        required=False)
    parser.add_argument('--image_network', type=str, default='clstm',
                        help='select backbone',
                        required=False)
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='Use input attribute as a feature',
                        required=False)
    parser.add_argument('--use_opticalflow', type=bool, default=False,
                        help='Use optical flow as a feature',
                        required=False)
   

    args = parser.parse_args()

    return args


def get_run_dir(args):
    return os.path.join(args.out_dir, args.log_name) if args.log_name else args.out_dir


def checkpoint_suffix(args):
    return '_scheduler' if args.lr_scheduler else ''


def default_checkpoint_name(args):
    return f'model_best{checkpoint_suffix(args)}.pkl'


def resolve_checkpoint_path(args):
    run_dir = get_run_dir(args)
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


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    target_model = model.module if isinstance(model, DistributedDataParallel) else model
    if any(key.startswith('module.') for key in checkpoint.keys()):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
    target_model.load_state_dict(checkpoint)


# For 2D datasets
def test_2d(args, test):
    print('='*100)
    print('Testing ...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')



    torch.manual_seed(0)
    use_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    verbose = dist.get_rank() == 0 if use_distributed else True
    net = network.PTINet(args).to(device)
    if use_distributed:
        net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    checkpoint_path = resolve_checkpoint_path(args)
    print(checkpoint_path)
    load_model_state(net, checkpoint_path, device)
    net.eval()
    test_sampler = DistributedSampler(test) if use_distributed else None
    dataloader_test = torch.utils.data.DataLoader(
        test,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.loader_workers,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    # mse = nn.MSELoss()
    huber = torch.nn.HuberLoss(reduction='sum', delta=1.0)
    class_weights = torch.tensor([1.0, args.crossing_pos_weight], device=device)
    crossing_ce = nn.CrossEntropyLoss(weight=class_weights)
    val_s_scores   = []
    val_c_scores   = []

    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0
    avg_acc = 0
    avg_rec = 0
    avg_pre = 0
    mAP = 0

    avg_epoch_val_s_loss   = 0
    avg_epoch_val_c_loss   = 0

    counter=0
    state_preds = []
    state_targets = []
    intent_preds = []
    intent_targets = []

    start = time.time()

    for idx, inputs in enumerate(dataloader_test):
        counter+=1
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
            mloss, speed_preds, crossing_preds, intentions = net(
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
            # speed_loss    = mse(speed_preds, future_speed)/100

            crossing_loss = 0.0
            for i in range(future_cross.shape[1]):
                target_cross = torch.argmax(future_cross[:, i], dim=1)
                crossing_loss = crossing_loss + crossing_ce(crossing_preds[:, i], target_cross)
            crossing_loss = crossing_loss / future_cross.shape[1]

            avg_epoch_val_s_loss += float(speed_loss)
            avg_epoch_val_c_loss += float(crossing_loss)

            preds_p = utils.speed2pos(speed_preds, pos)
            ade += float(utils.ADE(preds_p, future_pos))
            fde += float(utils.FDE(preds_p, future_pos))
            aiou += float(utils.AIOU(preds_p, future_pos))
            fiou += float(utils.FIOU(preds_p, future_pos))

            future_cross = torch.argmax(future_cross, dim=2).view(-1).cpu().numpy()
            crossing_preds = np.argmax(crossing_preds.view(-1, 2).detach().cpu().numpy(), axis=1)
            true_cls = future_cross

            label_c = label_c.view(-1).cpu().numpy()
            intentions = intentions.view(-1).detach().cpu().numpy()

            state_preds.extend(crossing_preds)
            state_targets.extend(true_cls)
            intent_preds.extend(intentions)
            intent_targets.extend(label_c)

    avg_epoch_val_s_loss /= counter
    avg_epoch_val_c_loss /= counter
    val_s_scores.append(avg_epoch_val_s_loss)
    val_c_scores.append(avg_epoch_val_c_loss)

    ade  /= counter
    fde  /= counter     
    aiou /= counter
    fiou /= counter

    state_metrics = binary_classification_metrics(state_preds, state_targets)
    intent_metrics = binary_classification_metrics(intent_preds, intent_targets)

    print(
        'Debug state confusion TP/FP/FN/TN:',
        state_metrics['tp'], state_metrics['fp'], state_metrics['fn'], state_metrics['tn']
    )
    print(
        'Debug intent confusion TP/FP/FN/TN:',
        intent_metrics['tp'], intent_metrics['fp'], intent_metrics['fn'], intent_metrics['tn']
    )
    print( '| ade: %.4f'% ade, 
        '| fde: %.4f'% fde,
        '| aiou: %.4f'% aiou,
        '| fiou: %.4f'% fiou,
        '| state_acc: %.4f'% state_metrics['accuracy'],
        '| int_acc: %.4f'% intent_metrics['accuracy'],
        '| f1_int: %.4f'% intent_metrics['f1'],
        '| f1_state: %.4f'% state_metrics['f1'],
        '| pre: %.4f'% state_metrics['precision'],
        '| recall_sc: %.4f'% state_metrics['recall'],
        '| pre_int: %.4f'% intent_metrics['precision'],
        '| recall_int: %.4f'% intent_metrics['recall'],
        '| t:%.4f'%(time.time()-start))




if __name__ == '__main__':
    args = parse_args()

    # create output dir
    # if not args.log_name:
    #     args.log_name = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
    #                             str(args.output), str(args.stride)) 
    # if not os.path.isdir(os.path.join(args.out_dir, args.log_name)):
    #     os.mkdir(os.path.join(args.out_dir, args.log_name))

    # select dataset
    if args.dataset == 'jaad':
        args.is_3D = False
    elif args.dataset == 'jta':
        args.is_3D = True
    elif args.dataset == 'nuscenes':
        args.is_3D = True
    else:
        print('Unknown dataset entered! Please select from available datasets: jaad, jta, nuscenes...')

    # load data
    test_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=get_run_dir(args),
                dtype='val',
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

    # test_loader = data_loader(args, test_set)

    # initiate network
    # net = network.PV_LSTM(args).to(args.device)

    # training
    test_2d(args,  test_set)
