# Cybernetic Online Stability Tuning
import copy
from abc import ABC, abstractmethod
from typing import List, Optional

import avalanche.training.plugins as avl_plugins
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.training.plugins.evaluation import (EvaluationPlugin,
                                                   default_logger)
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from helper.strategy import Strategy
from strategies.ewc import EWCPlugin
from strategies.lwf import LwFPlugin
from torch.nn import Module
from torch.optim import Optimizer


class MBAccuracy(StrategyPlugin):
    """Plugin for collecting accuracy for each minibatch"""

    def __init__(self):
        self.accuracy = Accuracy()
        super().__init__()

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs) -> None:
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self.accuracy.update(strategy.mb_output, strategy.mb_y, task_labels)

    def result(self):
        """Return the accuracy for the last mini-batch"""
        return self.accuracy.result().get(0, 0)

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        """Callback that resets the metric after the completion of a mini-batch"""
        self.accuracy.reset()


class ReferenceStrategy(StrategyPlugin):
    """Underlying strategy that we compare against"""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy.avl_strategy
        self.mb_accuracy = MBAccuracy()
        self.strategy.plugins.append(self.mb_accuracy)

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        s = self.strategy
        s.is_training = True
        s.model.train()
        s.model.to(s.device)

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.strategy.experience = strategy.experience
        # This is important! Reseting makes the comparison more fair
        # self.strategy.model = copy.deepcopy(strategy.model)
        self.strategy.model_adaptation()
        self.strategy.make_optimizer()

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        assert strategy.device == self.strategy.device, "Reference strategy uses a different compute device"

        self.strategy.clock.train_epoch_iterations = strategy.clock.train_epoch_iterations
        self.strategy.mbatch = strategy.mbatch

        self.strategy.before_training_iteration()

        self.strategy.optimizer.zero_grad()
        self.strategy.loss = 0

        # Forward
        self.strategy.before_forward()
        self.strategy.mb_output = self.strategy.forward()
        self.strategy.after_forward()

        # Backward
        self.strategy.loss += self.strategy.criterion()

        self.strategy.before_backward()
        self.strategy.loss.backward()
        self.strategy.after_backward()

        self.strategy.before_update()
        self.strategy.optimizer.step()
        self.strategy.after_update()
        self.strategy.after_training_iteration()

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self.strategy.after_eval_iteration()


class AbstractCyberneticOST(ABC, BaseStrategy):
    """Cybernetic Online Stability tuner, tunes stability during training"""
    @abstractmethod
    def get_intransigence(self) -> float:
        """Return the intransigence metric"""

    @abstractmethod
    def set_stability(self, stability: float):
        """Set the stability of the strategy"""

    @abstractmethod
    def get_stability(self) -> float:
        """Get the stability of the strategy"""


class SICyberneticOST(AbstractCyberneticOST):
    """
    Cybernetic Online Stability Tuner, tunes stability during training for 
    synaptic intelligence
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 si_lambda: float,
                 reference_strategy: BaseStrategy,
                 cost_params: dict = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        self.reference_strategy = ReferenceStrategy(reference_strategy)

        self.mb_accuracy = MBAccuracy()

        self.si = avl_plugins.SynapticIntelligencePlugin(si_lambda)

        self.controller = COSTControl(**cost_params)

        plugins = [self.mb_accuracy, self.controller,
                   self.reference_strategy, self.si]

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def get_intransigence(self):
        """Intransigence is the difference between the reference and strategy"""
        return self.reference_strategy.mb_accuracy.result() - self.mb_accuracy.result()

    def set_stability(self, stability):
        """Set the stability parameter"""
        self.si.si_lambda[0] = max(stability, 0)

    def get_stability(self):
        """Returns the stability parameter"""
        return self.si.si_lambda[0]


class LwFCyberneticOST(AbstractCyberneticOST):
    """
    Cybernetic Online Stability Tuner, tunes stability during training for 
    learning without forgetting
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 alpha: float,
                 temperature: float,
                 reference_strategy: BaseStrategy,
                 cost_params: dict = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        self.reference_strategy = ReferenceStrategy(reference_strategy)
        self.mb_accuracy = MBAccuracy()

        self.lwf = LwFPlugin(alpha, temperature)
        # assert plugins is (None or []), "No additional plugin should be provided" + str(plugins)

        self.controller = COSTControl(**cost_params)
        plugins = [self.mb_accuracy, self.controller,
                   self.reference_strategy, self.lwf]

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def get_intransigence(self) -> float:
        return self.reference_strategy.mb_accuracy.result() - self.mb_accuracy.result()

    def set_stability(self, stability: float):
        self.lwf.alpha = max(stability, 0)

    def get_stability(self) -> float:
        return max(self.lwf.alpha, 0)


class EWCCyberneticOST(AbstractCyberneticOST):
    """
    Cybernetic Online Stability Tuner, tunes stability during training for 
    elastic weight consolidation
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 ewc_lambda: float,
                 reference_strategy: BaseStrategy,
                 clipping_threshold=None,
                 mode="separate",
                 decay_factor=None,
                 cost_params: dict = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        self.reference_strategy = ReferenceStrategy(reference_strategy)
        self.mb_accuracy = MBAccuracy()

        self.ewc = EWCPlugin(
            ewc_lambda, mode, clipping_threshold, decay_factor)

        self.controller = COSTControl(**cost_params)
        plugins = [self.mb_accuracy, self.controller,
                   self.reference_strategy, self.ewc, *plugins]

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def get_intransigence(self) -> float:
        return self.reference_strategy.mb_accuracy.result() - self.mb_accuracy.result()

    def set_stability(self, stability: float):
        self.ewc.ewc_lambda = max(stability, 0)

    def get_stability(self) -> float:
        return max(self.ewc.ewc_lambda, 0)

class LRCyberneticOST(AbstractCyberneticOST):
    """
    Cybernetic Online Stability Tuner, tunes stability during training for
    learning Rate
    """

    def __init__(self, model: Module,
                 optimizer: Optimizer,
                 criterion,
                 reference_strategy: BaseStrategy,
                 cost_params: dict = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        self.reference_strategy = ReferenceStrategy(reference_strategy)
        self.mb_accuracy = MBAccuracy()

        self.controller = COSTControl(**cost_params)
        plugins = [self.mb_accuracy, self.controller,
                   self.reference_strategy, *plugins]

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def get_intransigence(self) -> float:
        return self.reference_strategy.mb_accuracy.result() - self.mb_accuracy.result()

    def set_stability(self, stability: float):
        stability = max(0, stability)
        for p_groups in self.optimizer.param_groups:
            p_groups["lr"] = stability

    def get_stability(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class PIDController():
    """
    Proportional–integral–derivative controller
    """

    def __init__(self, kp: float, ki: float, kd: float, setpoint: float):
        self.set_tuning(kp, ki, kd)
        self.integral = 0.0
        self.setpoint = setpoint
        self.last_error = 0.0
        self.derivative = 0.0

    def set_tuning(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def update(self, input_signal: float, delta_t: float):
        error = self.setpoint - input_signal
        self.integral += error*delta_t
        self.derivative = (error-self.last_error)/delta_t
        self.last_error = error
        return self.kp*error + self.ki*self.integral + self.kd*self.derivative

class COSTControl(StrategyPlugin):
    """Wraps PIDController to update stability"""

    def __init__(self, kp: float = .0, ki: float = .0, kd: float = .0, setpoint: float = .0,  border_factor: float = 1.0):
        self.pid = PIDController(kp, ki, kd, setpoint)
        # self.pid = LinearControl(kp, ki, kd, setpoint)
        self.border_factor = border_factor
        super().__init__()

    def before_training_exp(self, strategy: 'AbstractCyberneticOST', **kwargs):
        # return super().before_training_exp(strategy, **kwargs)
        strategy.set_stability(strategy.get_stability()*self.border_factor)

    def after_training_iteration(self, strategy: AbstractCyberneticOST, **kwargs):
        if strategy.training_exp_counter == 0:
            return
        control_signal = self.pid.update(strategy.get_intransigence(), 1)
        strategy.set_stability(strategy.get_stability()+control_signal)
