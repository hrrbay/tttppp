from typing import Tuple
import torch, hiera
from functools import partial
from torch import nn


class HieraB(torch.nn.Module): # wrapper for hiera base (16x224x224, 51M params)
    def __init__(
            self,
            pretrained=True,
            checkpoint="mae_k400_ft_k400",
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            init_embed_dim=96,
            stages: Tuple[int, ...] = (2, 3, 16, 3),
            finetune=True,
            ):
        super().__init__()
        depth = len(stages) - 1
        final_embed_dim = init_embed_dim * 2 ** depth
        print(f'Final embed dim: {final_embed_dim}')
        self.model = hiera.hiera_base_16x224(pretrained=pretrained, checkpoint=checkpoint, stages=stages, embed_dim=init_embed_dim)

        if finetune:
            for param in self.model.parameters(): # freeze all layers in hiera and only train head
                param.requires_grad = False

        self.norm = norm_layer(final_embed_dim)  # same head as in the original model, but for binary classification
        self.head = torch.nn.Linear(final_embed_dim, 1) # TODO maybe second linear layer?
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, x):
        _, intermediates = self.model(x, return_intermediates=True)  # class output and intermediate results for each block
        x = intermediates[-1]  # last block
        shape = x.shape
        x = x.view(shape[0], -1, shape[4])
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x