"""
Responsible for creating strategies and logging
"""
import copy
from datetime import datetime
from typing import Any, Type, Union

from algorithms.ost import SemiOnlineStabilityTuning, StabilityPolicy
from avalanche.benchmarks.scenarios.generic_definitions import ScenarioStream
from avalanche.evaluation import metrics as avl_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.replay import ReplayPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer
from avalanche.training.strategies.base_strategy import BaseStrategy
from metrics.cost_metrics import COSTMetric
from metrics.per_class_accuracy import AccuracyOnTrainedClasses
from torch import cuda, nn
from torch.optim.optimizer import Optimizer

from helper.strategy import LoggerWithNotes, Strategy


def new_full_label(label: str, params: dict, has_time:bool = True) -> str:
    """Add time and paramaters to make a full strategy label"""
    param_label = ", ".join([f"{k}={v}" for k, v in params.items()])

    time = ""
    if has_time:
        time = datetime.now().isoformat("_", "seconds")

    return f"{label}({param_label}){time}"

class StrategyRunner():
    """Run many strategies at once on a scenario"""
    scenario: ScenarioStream
    eval_every: int = 1
    first_batch_it: int = 1

    def __init__(self, scenario: ScenarioStream, eval_every:int=1) -> None:
        self.scenario = scenario
        self.eval_every = eval_every

    def run(self, *strategies):
        """Run one or more stratagies on a scenario"""
        n_experiences = self.scenario.n_experiences
        for index, experience in enumerate(self.scenario.train_stream):

            print()
            print("="*50)
            print(f"Start of experience:  {experience.current_experience}/{n_experiences}")
            print(f"Current Classes:      {experience.classes_in_this_experience}")
            print(f'This batch contains:  {len(experience.dataset)} patterns')
            print(f"GPU Memory in use:    {cuda.memory_allocated()/1e9:.5}GB")
            print(f"GPU Max memory:       {cuda.max_memory_allocated()/1e9:.5}GB")



            for strategy in [*strategies]:
                print("-"*50)
                print(strategy.label)
                print()
                strategy.avl_strategy.train(experience)

                if index % self.eval_every == 0 or index+1 == n_experiences:
                    strategy.avl_strategy.eval(self.scenario.test_stream[:index+1])
                strategy.flush()


class StrategyBuilder():
    """
    Use a builder pattern to construct a strategy
    """

    # pylint: disable=too-many-instance-attributes
    _model_kwargs:     Union[dict, None] = None
    _optimizer_kwargs: Union[dict, None] = None
    _training_kwargs:  Union[dict, None] = None
    _strategy_kwargs:  Union[dict, None] = None

    _optimizer_class: Type[Optimizer] = None
    _model_class: Type[nn.Module] = None
    _strategy_class: Type[BaseStrategy] = None

    _tb_log_dir: str = None
    _should_log: bool = False
    _criterion: Any = None
    _label: str = None
    _num_classes: int = None
    _mem_size: int = 0

    _runner: StrategyRunner = None

    _notes: dict = None

    def _copy(self):
        # We shallow copy making this class immutable, which means we don't accidentally
        # shoot ourselves in foot with pass by reference
        # return copy.copy(self)
        return self

    def use_logger(self, tb_log_dir: str):
        """Optionally output tensorboard data"""
        self._tb_log_dir = tb_log_dir
        self._should_log = True
        return self

    def without_logger(self):
        """Disable logger"""
        self._should_log = False
        return self

    def add_replay(self, mem_size):
        """Add a replay method to constructed stratagies"""
        self._mem_size = mem_size
        return self

    def use_model(self, model_class: Type[nn.Module], **model_kwargs):
        """Build a strategy with a model"""
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        return self

    def use_optimizer(self, optimizer_class: Type[Optimizer], **optimizer_kwargs):
        """Build a strategy with an optimizer"""
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs
        return self

    def use_training_args(self, **training_kwargs):
        """Build a strategy with specified training arguments"""
        self._training_kwargs = training_kwargs
        return self

    def use_loss_function(self, criterion):
        """Build a strategy with a specified loss function"""
        self._criterion = criterion
        return self

    def use_strategy(self, strategy_class: Type[BaseStrategy], **strategy_kwargs):
        """Build a strategy using a specified continual learning strategy"""
        self._strategy_class = strategy_class
        self._strategy_kwargs = strategy_kwargs
        return self

    def use_label(self, label):
        """Specify a label for the algorithm"""
        self._label = label
        return self

    def get_hyper_parameters(self) -> dict:
        """Output hyperparamers as a single dict"""
        return {
            "model":     {"class": str(self._model_class),     **self._model_kwargs},
            "optimizer": {"class": str(self._optimizer_class), **self._optimizer_kwargs},
            "ExperienceBalancedBufferSize": self._mem_size,
            "training": self._training_kwargs,
            "strategy": {"label": self._label, "parameters": self._strategy_kwargs}
        }

    def get_label(self) -> str:
        """get the label"""
        return self._label

    def add_runner(self, runner: StrategyRunner):
        """Add a runner to simplify running"""
        self._runner = runner
        return self

    def run(self):
        """Run the strategy with the specified runner"""
        self._runner.run(self.build())

    def fix_cm(self, num_classes: int):
        """Supply the confusion matrix with the number of classes this can fix it for core50"""
        self._num_classes = num_classes

    def _build_logger(self, label):
        """Create a logger with notes about parameters"""
        log = LoggerWithNotes(label, tb_log_dir=self._tb_log_dir)
        for (key, value) in self.get_hyper_parameters().items():
            log.note_paramater(key, value)
        log.save()
        return log

    def copy(self):
        """Duplicate self"""
        # new = copy.copy(self)
        return copy.copy(self)

    def build(self):
        """Build a strategy to specifications"""
        assert self._model_class, f"Builder needs a model to be specified {self._model_class}"
        assert self._optimizer_class, "Builder needs an optimizer to be specified"
        assert self._criterion, "Builder needs a loss function to be specified"
        assert self._strategy_class, "Builder needs a strategy to be specified"

        strategy = Strategy()
        loggers = [InteractiveLogger()]
        strategy.label = new_full_label(self._label, self._strategy_kwargs, False)

        model = self._model_class(**self._model_kwargs)
        optimizer = self._optimizer_class(model.parameters(), **self._optimizer_kwargs)

        eval_plugin = EvaluationPlugin(
            COSTMetric(),
            avl_metrics.confusion_matrix.StreamConfusionMatrix(self._num_classes),
            avl_metrics.accuracy_metrics(
                epoch=True,
                experience=True,
                stream=True,
                trained_experience=True,
                minibatch=True),
            avl_metrics.loss_metrics(
                epoch=True,
                experience=True,
                stream=True,
                minibatch=True),
            avl_metrics.timing_metrics(epoch=True),
            AccuracyOnTrainedClasses(),
            avl_metrics.forgetting_metrics(
                experience=True,
                stream=True),
            loggers=loggers)

        plugins = []
        if self._mem_size != 0:
            plugins.append(ReplayPlugin(self._mem_size, ExperienceBalancedBuffer(self._mem_size, True)))

        strategy.avl_strategy = self._strategy_class(
            model,
            optimizer,
            self._criterion,
            **self._strategy_kwargs,
            **self._training_kwargs,
            plugins=plugins,
            evaluator=eval_plugin)

        # If we need a logger build it
        if self._should_log:
            log = self._build_logger(new_full_label(self._label, self._strategy_kwargs))
            loggers.append(log)
            strategy.log = log

        return strategy

def build_tuning_strategy(
    tuning_strategy: Type[SemiOnlineStabilityTuning],
    strategy: StrategyBuilder,
    reference: StrategyBuilder,
    tuning_policy: StabilityPolicy,
    tb_log_dir: str) -> SemiOnlineStabilityTuning:
    """
    Build a tuning strategy using builders of other underlying strategies
    """

    # Disable loggers of undrlying methods this avoids us having multiple logs of
    # the same things. We do this because the stability tuning strategy is
    # responsible for logging
    reference.without_logger()
    strategy.without_logger()

    reference_built = reference.without_logger().build()
    strategy_built = strategy.build()
    params = {"strategy":strategy_built, **tuning_policy.params()}

    log=None
    if tb_log_dir:
        log = LoggerWithNotes(new_full_label(tuning_policy.name(), params), tb_log_dir)
        log.note_paramater("reference", reference.get_hyper_parameters())
        log.note_paramater("strategy", strategy.get_hyper_parameters())
        log.note_paramater("policy", {"class": tuning_policy.name(), **tuning_policy.params()})
        log.save()

    return tuning_strategy(
        reference_strategy=reference_built,
        strategy=strategy_built,
        policy=tuning_policy,
        log=log,
        label=new_full_label(tuning_policy.name(), params, has_time=False)
    )

