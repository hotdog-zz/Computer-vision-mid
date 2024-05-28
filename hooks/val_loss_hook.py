from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from mmengine.utils import is_list_of
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

def parse_losses(losses):
    log_vars = []
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars.append([loss_name, loss_value.mean()])
        elif is_list_of(loss_value, torch.Tensor):
            log_vars.append(
                [loss_name,
                    sum(_loss.mean() for _loss in loss_value)])
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(value for key, value in log_vars if 'loss' in key)
    log_vars.insert(0, ['loss', loss])
    log_vars = OrderedDict(log_vars)  # type: ignore

    return loss, log_vars  # type: ignore

@HOOKS.register_module()
class ValLossHook(Hook):
    def __init__(self, interval=1):
        self.interval = interval

    def after_train_epoch(self, runner) -> None:
        model = runner.model
        model.eval()
        all_losses = []
        for i, data in enumerate(runner.val_dataloader):
            data = model.module.data_preprocessor(data, True)
            losses = model._run_forward(data, mode='loss')  # type: ignore
                
            parsed_losses, log_vars = parse_losses(losses)  # type: ignore
            all_losses.append(log_vars['loss'].item())
    
        avg_loss = sum(all_losses) / len(all_losses)
        runner.logger.info(f'Validation Loss at epoch {runner.epoch}: {avg_loss}')
        runner.visualizer.add_scalar("val_loss", avg_loss, runner.epoch)
        model.train()