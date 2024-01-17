import torchvision
import os
import pdb

from torchvision import transforms

from . import data_loader

BASE_PATH = '/mnt/shared/datasets'
# BASE_PATH = '/home/jakob/datasets'

data_config = {
    # TODO: add normalization for all
    'base': {
        'path': os.path.join(BASE_PATH, 't3p3'),
        'transforms': [
            transforms.Normalize(mean=0.012362, std=0.076184),
        ]
    },
    'r3d_18': {
        'path': os.path.join(BASE_PATH, 't3p3'),
        'transforms': [
            transforms.Normalize(mean=0.01298, std=0.07275),
            transforms.Resize((112,112)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
        ]
    },
    'hiera': {
            'path': os.path.join(BASE_PATH, 't3p3'),
            'transforms': [
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
            ]
        }
}
