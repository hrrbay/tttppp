import os
import pdb

from torchvision import transforms

BASE_PATH = os.path.join(os.path.dirname(__file__), '..')
BASE_PATH = os.path.abspath(BASE_PATH)
if 'T3P3_PATH' in os.environ:
    BASE_PATH = os.environ['T3P3_PATH']
  
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
            transforms.Normalize(mean=0.012362, std=0.076184),
            transforms.Resize((112,112)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
        ]
    },
    'hiera': {
            'path': os.path.join(BASE_PATH, 't3p3'),
            'transforms': [
                transforms.Normalize(mean=0.012362, std=0.076184),
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # TODO: Check this does the correct thing (repeating channel)
            ]
        },
    'tttransformer': {
            'path': os.path.join(BASE_PATH, 't3p3'),
            'transforms': [
            ]
        }
}
