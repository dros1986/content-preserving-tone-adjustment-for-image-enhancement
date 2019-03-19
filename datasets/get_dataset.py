import torch
import torch.nn as nn
from PlainDataset import PlainDataset


def get_dataset(params, phase):
    # include filenames if regen
    if phase=='regen': params['params'][phase]['include_filenames']=True
    # define dataset
    ds_type = params['type']
    if ds_type == 'plain': return PlainDataset(params['params'][phase])
    else: raise ValueError('Dataset type not available.')
