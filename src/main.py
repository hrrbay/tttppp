import argparse
import importlib
import pdb
import os
import time

import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

import torchvision

from data import data_loader, dataset, data_config
# from network import TestNet
import train
import network
import config.utils



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='TestNet')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr-patience', default=6, type=int)
    parser.add_argument('--lr-factor', default=3, type=float)
    parser.add_argument('--lr-min', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=0.0002, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int,  default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-config', type=str, default='base')
    parser.add_argument('--src-fps', type=int, default=120)
    parser.add_argument('--target-fps', type=int, default=30)
    parser.add_argument('--labeled-start', action='store_true', default=False)
    parser.add_argument('--window-size', type=int, default=7)
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--train-config', type=str, default=None)
    parser.add_argument('--has-checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint-freq', type=int, default=5)
    return parser.parse_args()


def main():
    # load args
    args = parse_arguments()
    if args.train_config is not None:
        args = config.utils.update_args(args, args.train_config)  # update args with train config file

    # set device
    if args.gpu < 0:
        print(f'* GPU < 0. Will use CPU instead.')
        device = 'cpu'
    if not torch.cuda.is_available():
        print('* WARNING: cuda is not available. Will use CPU instead.')
        device = 'cpu'
    elif torch.cuda.device_count() < args.gpu:
        print(f'* WARNING: torch.cuda.device_count() < {args.gpu}. Will use GPU 0 instead.')
        device = 'cuda'
        torch.cuda.device(0)
    else:
        device = 'cuda'
        torch.cuda.device(args.gpu)

    # data-config
    data_conf = data_config.data_config[args.data_config]
    transforms = data_conf['transforms']
    transforms = [torchvision.transforms.ToTensor()] + transforms
    data_path = data_conf['path']

    # load network
    if hasattr(network, args.network):
        model = getattr(network, args.network)(model_config=args.model_config)
    elif hasattr(torchvision.models.video, args.network):
        # torchvision models. Assuming last layer to be fc
        model = getattr(torchvision.models.video, args.network)(pretrained=args.pretrained, progress=True)
        head_name, head_var = list(model.named_modules())[-1]
        assert type(head_var) == torch.nn.Linear, 'Fix this.'
        setattr(model, head_name, torch.nn.Linear(in_features=head_var.in_features, out_features=1))
    # TODO: add other models here if needed
    
    model.to(device)
    trn_loader, val_loader, tst_loader = data_loader.load_data(data_path, args.batch_size, args.src_fps, args.target_fps, args.labeled_start, args.window_size, args.seed, transforms=transforms, validation=0.1)
    
    # init logging w/ tensorboard
    run_id = f'{args.network}_{args.lr}' #we need some naming scheme
    if not os.path.exists(f'../runs/{run_id}'):
        os.makedirs(f'../runs/{run_id}')
        os.makedirs(f'../runs/{run_id}/logs')
        os.makedirs(f'../runs/{run_id}/model')
    summary_writer = SummaryWriter(f'../runs/{run_id}/logs')
    
    # optimizer
    optim = SGD(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train.train(trn_loader, val_loader, tst_loader, args.nepochs, model, optim, args.lr_patience, args.lr_factor, args.lr_min, device, summary_writer, args.has_checkpoint, args.checkpoint_freq)
if __name__ == '__main__':
    main()