import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable

SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'MSELoss', 'SmoothL1Loss', 'L1Loss']

class NewLoss(nn.Module):
    def __init__(self, params):
        super(NewLoss, self).__init__()
        pass
    def forward(self, inputs,targets):
        pass

def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")