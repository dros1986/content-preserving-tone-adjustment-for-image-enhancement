import torch
import torch.nn.functional as F

def get_loss(params):
    if params['type'].lower() == 'l2':
        return F.mse_loss
    elif params['type'].lower() == 'l1':
        return F.l1_loss
    else: raise ValueError('Loss not implemented.')
