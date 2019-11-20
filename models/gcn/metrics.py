import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable

SUPPORTED_METRICS = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'MSELoss', 'Accuracy']

class NewMetric:
    def __init__(self, params, **kwargs):
        super(NewMetric, self).__init__()
        pass
    def __call__(self, inputs, targets):
        pass

class Accuracy:
    def __init__(self, **kwargs):
        pass
    def __call__(self, inputs, targets):
        labels = torch.argmax(inputs, dim = 1)
        accuracy = torch.mean((labels == targets).to(torch.double)).detach().cpu()
        return accuracy


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """
    assert 'eval_metric' in config, 'Could not find evalvalutation metric configuration'
    eval_config = config['eval_metric']
    name = eval_config.pop('name')

    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'Accuracy':
        return Accuracy()
    else:
        raise RuntimeError(f"Unsupported metric function: '{name}'. Supported losses: {SUPPORTED_METRICS}")