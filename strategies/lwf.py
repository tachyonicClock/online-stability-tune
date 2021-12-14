import torch
from typing import Optional, Sequence, List, Union

from torch.nn import Module
from torch.optim import Optimizer
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_logger
from avalanche.training.plugins.lwf import LwFPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy

class LwFPlugin(LwFPlugin):

    def _distillation_loss(self, out, prev_out, active_units):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

class LwF(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 alpha: Union[float, Sequence[float]], temperature: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        lwf = LwFPlugin(alpha, temperature)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)