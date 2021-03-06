import logging
import os
import pdb

from tqdm import tqdm 
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from utils.helper import RunningAverage, save_checkpoint, load_checkpoint, get_logger, get_batch_size
from utils.visualize import VisdomLinePlotter

class Trainer:
    """Network trainer

    Args:
        model: network model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metricc
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=1000, max_num_iterations=None,
                 validate_after_iters=None, log_after_iters=None,
                 validate_iters=None, num_iterations=0, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None, inference_config = None):
        if logger is None:
            self.logger = get_logger('Trainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.plotter = VisdomLinePlotter('gcn')

        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.inference_config = inference_config
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders, logger=None, inference_config = None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger=logger, inference_config = inference_config)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=1000, max_num_iterations=None,
                        validate_after_iters=None, log_after_iters=None,
                        validate_iters=None, num_iterations=0, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        logger=None, inference_config = None):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   logger=logger, inference_config = inference_config)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()
        self.logger.info(
            f'Training epoch [{self.num_epoch}/{self.max_num_epochs - 1}], iteration per epoch: {len(train_loader)}. ')
        # sets the model in training mode
        self.model.train()
        if self.validate_after_iters is None:
            self.validate_after_iters = len(train_loader)
        if self.log_after_iters is None:
            self.log_after_iters = self.validate_after_iters
        if self.max_num_iterations is None:
            self.max_num_iterations = self.max_num_epochs * len(train_loader)

        for i, t in enumerate(train_loader):
            target = t.y.to(self.device)
            input = t.to(self.device)
            output = self.model(input)

            # compute loss criterion
            loss = self.loss_criterion(output, target)
            train_losses.update(loss.item(), get_batch_size(target))

            # compute eval criterion
            eval_score = self.eval_criterion(output, target)
            train_eval_scores.update(eval_score.item(), get_batch_size(target))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                # log stats and params
                self.logger.info(
                    f'Training iteration [{self.num_iterations}/{self.max_num_iterations - 1}]. Batch [{i}/{len(train_loader) - 1}]. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self.plotter.plot('loss', 'train', 'loss', self.num_iterations, train_losses.avg, xlabel='Iter')
                self.plotter.plot('accuracy', 'train', 'accuracy', self.num_iterations, train_eval_scores.avg, xlabel='Iter')

                train_losses = RunningAverage()
                train_eval_scores = RunningAverage()

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                # save checkpoint
                self._save_checkpoint(is_best)
                self._log_params()

                if self.inference_config is not None:
                    if (self.num_iterations >= self.inference_config['infer_init_iters'] and
                        self.num_iterations % self.inference_config['infer_after_iters'] == 0):
                        self.inference(self.loaders)

            if self.num_iterations >= self.max_num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        return False

    def validate(self, val_loader):

        val_losses = RunningAverage()
        val_scores = RunningAverage()
        
        self.logger.info(f'Validating epoch [{self.num_epoch}/{self.max_num_epochs - 1}]. ')
        if self.validate_iters is None:
            self.validate_iters = len(val_loader)
        val_iterator = iter(val_loader)
        
        try:
            self.model.eval()
            with torch.no_grad():
                for _ in tqdm(range(self.validate_iters)):
                    try:
                        t = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_loader)
                        t = next(val_iterator)

                    target = t.y.to(self.device)
                    input = t.to(self.device)
                    output = self.model(input)

                    # compute loss criterion
                    loss = self.loss_criterion(output, target)
                    val_losses.update(loss.item(), get_batch_size(target))

                    # compute eval criterion
                    eval_score = self.eval_criterion(output, target)
                    val_scores.update(eval_score.item(), get_batch_size(target))

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                self.plotter.plot('loss', 'val', 'loss', self.num_iterations, val_losses.avg, xlabel='Iter')
                self.plotter.plot('accuracy', 'val', 'accuracy', self.num_iterations, val_scores.avg, xlabel='Iter')

                return val_scores.avg
        finally:
            # set back in training mode
            self.model.train()
        
    def inference(self, loaders):
        self.logger.info(f'Infering hidden data... ')
        return
    
    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score
        return is_best

    def _save_checkpoint(self, is_best):
        if torch.cuda.device_count() > 1:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': model_state,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)