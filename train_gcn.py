import importlib
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.data_handler import get_data_loaders
from utils.helper import get_logger, get_number_of_learnable_parameters
from utils.config import load_config
from utils.trainer import Trainer

from models.gcn.losses import get_loss_criterion
from models.gcn.metrics import get_evaluation_metric

CONFIG_PATH = "./configs/train_gcn_basic_01.yaml"

def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    validate_iters = trainer_config.get('validate_iters', None)

    if resume is not None:
        # continue training from a given checkpoint
        return Trainer.from_checkpoint(resume, model,
                                        optimizer, lr_scheduler, loss_criterion,
                                        eval_criterion, config['device'], loaders,
                                        logger=logger)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return Trainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                        eval_criterion, device=config['device'], loaders=loaders,
                                        max_num_epochs=trainer_config['epochs'],
                                        max_num_iterations=trainer_config['iters'],
                                        validate_after_iters=trainer_config['validate_after_iters'],
                                        log_after_iters=trainer_config['log_after_iters'],
                                        eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                                        logger=logger, validate_iters = validate_iters)
    else:
        # start training from scratch
        return Trainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                            config['device'], loaders, trainer_config['checkpoint_dir'],
                            max_num_epochs=trainer_config['epochs'],
                            max_num_iterations=trainer_config['iters'],
                            validate_after_iters=trainer_config['validate_after_iters'],
                            log_after_iters=trainer_config['log_after_iters'],
                            eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                            logger=logger, validate_iters = validate_iters)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)

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
    logger = get_logger('GCN Trainer')

    parser = argparse.ArgumentParser(description='GCN training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default = CONFIG_PATH)
    args = parser.parse_args()

    # Load and log experiment configuration
    config = load_config(args.config)
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    module_path = "models.gcn.model"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(_get_model(module_path, config))
    else:
        model = _get_model(module_path, config)

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}', using {torch.cuda.device_count()} GPUs...")
    model = model.to(config['device'])
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_data_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
                              logger=logger)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()