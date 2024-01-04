import torchvision
import os

from torchvision import transforms

BASE_PATH = './data'
HDD_PATH = '/media/christian/DATA/tttppp/data'

data_config = {
    # TODO: add normalization for all
    'base': {
        'path': os.path.join(HDD_PATH, 't3p3'),
        'transforms': [
        ]
    },
    'r3d_18': {
        'path': os.path.join(BASE_PATH, 't3p3'),
        'transforms': [
            transforms.Resize((112,112)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
        ]
    },
    'hiera': {
            'path': os.path.join(HDD_PATH, 't3p3'),
            'transforms': [
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
            ]
        }
}