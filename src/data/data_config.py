import torchvision
import os

from torchvision import transforms

BASE_PATH = '/home/jakob/datasets'
data_config = {
    # TODO: add normalization for all
    'base': {
        'path': os.path.join(BASE_PATH, 't3p3'),
        'transforms': [
        ]
    },
    'r3d_18': {
        'path': os.path.join(BASE_PATH, 't3p3'),
        'transforms': [
            transforms.Resize((112,112)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
        ]
    } 
}