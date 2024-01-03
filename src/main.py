import argparse
import importlib
import pdb

import torch
from torch.optim import SGD
import torchvision

from data import data_loader, dataset
# from network import TestNet
import train

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='TestNet')
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
    parser.add_argument('--data-path', type=str, default='/mnt/data/datasets/t3p3')
    parser.add_argument('--src-fps', type=int, default=120)
    parser.add_argument('--target-fps', type=int, default=30)
    parser.add_argument('--labeled-start', action='store_true', default=False)
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--sparse-data', action='store_true')
    parser.add_argument('--data-mode', type=str, default='mmap')
    parser.add_argument('--gpu', type=int, default='0')
    # parser.print_help()
    return parser.parse_args()


def main():
    # load args
    args = parse_arguments()

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

    # load network
    NetworkCls = getattr(importlib.import_module('network'), args.network)
    # network_cls = importlib.import_module(f'network.TestNet')
    model = NetworkCls()
    model.to(device)
    transforms=[torchvision.transforms.ToTensor()]

    ''' 
    NOTE: use the next 4 lines to use resnet3d. Just hardcoded it here for now to try
    '''
    # model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
    # model.fc = torch.nn.Linear(model.fc.in_features, out_features=1)
    # model.to(device)
    # transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((112,112)), torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
    
    trn_loader, val_loader, tst_loader = data_loader.load_data(args.data_path, args.batch_size, args.src_fps, args.target_fps, args.labeled_start, args.window_size, args.data_mode, args.seed, transforms=transforms, validation=0.1)
    print(f'got dataloaders')
    # optimizer
    optim = SGD(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train.train(trn_loader, val_loader, tst_loader, args.nepochs, model, optim, args.lr_patience, args.lr_factor, args.lr_min, device)
if __name__ == '__main__':
    main()