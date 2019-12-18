import importlib
import argparse

import torch

from tqdm import tqdm 

from utils.data_handler import get_data_loaders
from utils.helper import load_checkpoint, get_batch_size, RunningAverage
from utils.config import load_config

from models.gcn.losses import get_loss_criterion
from models.gcn.metrics import get_evaluation_metric

CONFIG_PATH = "./configs/train_gcn_basic_01.yaml"
MODEL_PATH = "/home/sherry/Dropbox/PhD/dl_graph_connectomes/checkpoints/best_checkpoint.pytorch"

def validate(model, val_loader, loss_criterion, eval_criterion, device):

    val_losses = RunningAverage()
    val_scores = RunningAverage()

    val_iterator = iter(val_loader)
        
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(len(val_loader))):
            t = next(val_iterator)

            target = t.y.to(device)
            input = t.to(device)
            output, _ = model(input)

            # compute loss criterion
            loss = loss_criterion(output, target)
            val_losses.update(loss.item(), get_batch_size(target))

            # compute eval criterion
            eval_score = eval_criterion(output, target)
            val_scores.update(eval_score.item(), get_batch_size(target))

    return val_losses.avg, val_scores.avg

def _get_model(module_path, config):
    def _model_class(module_path, class_name):
        m = importlib.import_module(module_path)
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(module_path, model_config['name'])
    return model_class(**model_config)

def main():
    # Create main logger
    parser = argparse.ArgumentParser(description='GCN training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default = CONFIG_PATH)
    parser.add_argument('--model', type=str, help='Path to the model parameters', default = MODEL_PATH)
    args = parser.parse_args()

    # Load experiment configuration
    config = load_config(args.config)
    print(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    module_path = "models.gcn.model"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(_get_model(module_path, config))
    else:
        model = _get_model(module_path, config)
    
    load_checkpoint(args.model, model)

    # put the model on GPUs
    model = model.to(config['device'])

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_data_loaders(config)

    # Start testing
    val_losses, val_scores = validate(model, loaders['test'], loss_criterion, eval_criterion, config['device'])

    print('testing loss is:', val_losses)
    print('evaluation score is:', val_scores)

if __name__ == '__main__':
    main()