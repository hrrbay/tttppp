import argparse
import json
import os
import pdb

import network
import numpy as np
import torch
import torchvision
from network import TestNet
import train
from data import data_config, data_loader
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    """
        Parsing a lot of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default=None, type=str, required=False, help='name of the experiment for logging and checkpointing')
    parser.add_argument('--network', type=str, default='TestNet', help='network to use from either src/network or torchvision.models.video')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained weights from torchvision.models.video')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for optimizer')
    parser.add_argument('--lr-patience', default=6, type=int, help='patience for early stopping')
    parser.add_argument('--lr-factor', default=3, type=float, help='factor for reducing learning rate')
    parser.add_argument('--lr-min', default=1e-4, type=float, help='minimum learning rate')
    parser.add_argument('--validation', default=0.1, type=float, help='percentage of data to use for validation')
    parser.add_argument('--validation-vid', default=None, type=int, help='optional: video to use for validation')
    parser.add_argument('--flip-prob', default=0, type=float, required=False, help='probability of flipping a video horizontally')
    parser.add_argument('--weight-decay', default=0.0002, type=float, help='weight decay for optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int,  default=32, help='batch size for training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--data-config', type=str, default='base', help='data configuration to use from src/data/data_config.py')
    parser.add_argument('--src-fps', type=int, default=120, help='fps of source videos')
    parser.add_argument('--target-fps', type=int, default=30, help='fps to downsample to')
    parser.add_argument('--labeled-start', action='store_true', default=False, help='use only rallies and labeled start for videos')
    parser.add_argument('--window-size', type=int, default=7, help='window size for sliding window')
    parser.add_argument('--gpu', type=int, default='0', help='gpu to use')
    parser.add_argument('--model-config', type=str, default=None, help='model configuration to use from src/config, only for Hiera and TTTransformer')
    parser.add_argument('--has-checkpoint', type=bool, default=False, help='whether to start training from a checkpoint')
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='how often to save a checkpoint')
    parser.add_argument('--freeze-backbone', default=False, action='store_true', help='whether to freeze the backbone of the used network')
    parser.add_argument('--model-name', default='model', type=str, required=False, help='name of the model')
    parser.add_argument('--fixed-seq-len', default=0, type=int, required=False, help='whether to use fixed sequence length before end of rally, labeled start has to be false')
    parser.add_argument('--test-model', default=None, type=str, required=False, help='path to model for evaluation, if set no training will be done only a test run')
    parser.add_argument('--use-poses', type=bool, default=False, help='whether to use poses instead of segmentation masks')
    return parser.parse_args()


def main():
    # load args
    args = parse_arguments()

    print(f'Arguments:')
    for arg, val in vars(args).items():
        print(f'\t{arg}: {val}')
    print('-' * 80)
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
        print(f'Using GPU {args.gpu}')
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
        model = getattr(network, args.network)(model_config=args.model_config)
    elif hasattr(torchvision.models.video, args.network):
        # torchvision models. Assuming last layer to be fc
        model = getattr(torchvision.models.video, args.network)(pretrained=args.pretrained, progress=True)
        head_name, head_var = list(model.named_modules())[-1]
        assert type(head_var) == torch.nn.Linear, 'Fix this.'
        setattr(model, head_name, torch.nn.Linear(in_features=head_var.in_features, out_features=1))
    else:
        raise NotImplementedError(f'Network {args.network} not implemented.')
    
    model.to(device)
    if args.freeze_backbone:
        layers = list(model.modules())[1:-1]
        for l in layers:
            print(f'freezing {l}')
            if hasattr(l, 'weight'):
                l.weight.requires_grad = False
            if hasattr(l, 'bias') and l.bias is not None:
                l.bias.requires_grad = False

    trn_loader, val_loader, tst_loader = data_loader.load_data(data_path, args.batch_size, args.src_fps, args.target_fps,
                                                               args.labeled_start, args.window_size, args.seed, transforms=transforms,
                                                               validation=args.validation, fixed_seq_len=args.fixed_seq_len, flip_prob=args.flip_prob,
                                                               validation_vid=args.validation_vid, use_poses=args.use_poses)
    
    if args.test_model:
        best_model = torch.load(args.test_model)
        model.load_state_dict(best_model['model_state_dict'])
        loss, acc, _, _ = train.eval(tst_loader, model, device)
        print(f'tst_loss: {loss:.4f}, tst_acc: {acc*100:.2f}%')
        loss, acc, _, _ = train.eval(trn_loader, model, device)
        print(f'trn_loss: {loss:.4f}, trn_acc: {acc*100:.2f}%')
        loss, acc, _, _ = train.eval(val_loader, model, device)
        print(f'val_loss: {loss:.4f}, val_acc: {acc*100:.2f}%')
        print('Evaluating on test set...')
        vid_accs = train.eval_test_split(tst_loader, model, device)
        exit(0)

    # init logging w/ tensorboard
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = f'{args.network}_{args.lr}'
    run_dir = os.path.join(os.path.dirname(__file__), '..', 'runs')
    print(f'run_dir: {run_dir}')
    run_id = f'{exp_name}/{args.seed}' #we need some naming scheme
    print(f'saving to: {run_dir}/{run_id}')
    if not os.path.exists(f'{run_dir}/{run_id}'):
        os.makedirs(f'{run_dir}/{run_id}')
        os.makedirs(f'{run_dir}/{run_id}/logs')
        os.makedirs(f'{run_dir}/{run_id}/model')
    summary_writer = SummaryWriter(f'{run_dir}/{run_id}/logs')
    # store arguments
    with open(os.path.join(run_dir, run_id, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f'training {sum([p.numel() for p in train_params])} params')
    optim = SGD(params=train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    train.train(trn_loader, val_loader, tst_loader, args.nepochs, model, optim, args.lr_patience, args.lr_factor, args.lr_min, device, summary_writer, args.has_checkpoint, args.checkpoint_freq)

if __name__ == '__main__':
    main()