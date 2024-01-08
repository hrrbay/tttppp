import argparse
import importlib
import pdb

import torch
from torch.optim import SGD
import torchvision
import numpy as np

from data import data_loader, dataset, data_config
# from network import TestNet
import train
import network


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='TestNet')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr-patience', default=6, type=int)
    parser.add_argument('--lr-factor', default=3, type=float)
    parser.add_argument('--lr-min', default=1e-4, type=float)
    parser.add_argument('--validation', default=0.1, type=float)
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
    parser.add_argument('--freeze-backbone', default=False, action='store_true')
    parser.add_argument('--model-name', default='model', type=str, required=False)
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

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data-config
    data_conf = data_config.data_config[args.data_config]
    transforms = data_conf['transforms']
    transforms = [torchvision.transforms.ToTensor()] + transforms
    data_path = data_conf['path']

    # load network
    if hasattr(network, args.network):
        model = getattr(network, args.network)()
    elif hasattr(torchvision.models.video, args.network):
        # torchvision models. Assuming last layer to be fc
        model = getattr(torchvision.models.video, args.network)(pretrained=args.pretrained, progress=True)
        head_name, head_var = list(model.named_modules())[-1]
        assert type(head_var) == torch.nn.Linear, 'Fix this.'
        setattr(model, head_name, torch.nn.Linear(in_features=head_var.in_features, out_features=1))
    # TODO: add other models here if needed
    model.to(device)

    if args.freeze_backbone:
        layers = list(model.modules())[1:-1]
        for l in layers:
            print(f'freezing {l}')
            if hasattr(l, 'weight'):
                l.weight.requires_grad = False
            if hasattr(l, 'bias') and l.bias is not None:
                l.bias.requires_grad = False

    trn_loader, val_loader, tst_loader = data_loader.load_data(data_path, args.batch_size, args.src_fps, args.target_fps, args.labeled_start, args.window_size, args.seed, transforms=transforms, validation=args.validation)
    
    # optimizer
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f'training {len(train_params)} params')
    optim = SGD(params=train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train.train(trn_loader, val_loader, tst_loader, args.nepochs, model, optim, args.lr_patience, args.lr_factor, args.lr_min, device)

    # save model
    # torch.save(model.state_dict(), f'./models/{args.model_name}.ckpt')
if __name__ == '__main__':
    main()