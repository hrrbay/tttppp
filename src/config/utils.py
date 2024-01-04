import yaml


def load_config(path: str):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config

def update_args(args, config):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['train']
        optimizer = config['optimizer']
        args.nepochs = config['nepochs']
        args.batch_size = config['batch_size']
        args.lr_patience = config['lr_patience']
        args.lr_factor = config['lr_factor']
        args.lr_min = config['lr_min']
        args.gpu = config['gpu']
        args.seed = config['seed']

        optimizer_args = config[optimizer]
        for arg in optimizer_args: # TODO for different optimizers arg might not be in args
            setattr(args, arg, optimizer_args[arg])
    return args
