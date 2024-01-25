import hiera
import torch
from config.utils import load_config
from torch import nn

class HieraB(torch.nn.Module): 
    """
        Wrapper for hiera base (16x224x224, 51M params)
    """
    def __init__(self, model_config: str):
        config = load_config(model_config)['hiera_b']
        # return results after each hiera block, we only use the last intermediate results
        self.return_intermediates = config['return_intermediates']  

        super().__init__()
        final_embed_dim = config['init_embed_dim'] * 2 ** (len(config['stages']) - 1)
        print(f'Final embed dim: {final_embed_dim}')
        self.model = hiera.hiera_base_16x224(pretrained=config['pretrained'], checkpoint=config['checkpoint'],
                                             stages=config['stages'], embed_dim=config['init_embed_dim'])

        if config['finetune']:
            # freeze all layers in hiera and only train head
            for param in self.model.parameters(): 
                param.requires_grad = False

        # same head as in the original model, but for binary classification
        self.norm = nn.LayerNorm(final_embed_dim, eps=1e-6)  
        self.last = torch.nn.Linear(final_embed_dim, 1)
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, x):
        # class output and intermediate results for each block
        _, intermediates = self.model(x, return_intermediates=self.return_intermediates) 
        x = intermediates[-1]  # last block
        shape = x.shape
        x = x.view(shape[0], -1, shape[4])
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.last(x)
        return x