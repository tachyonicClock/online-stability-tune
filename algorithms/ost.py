# Online Stability Tuning
import copy
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
from avalanche.benchmarks.scenarios.generic_definitions import Experience
from moacl.helper.strategy import LoggerWithNotes, Strategy


class StabilityPolicy(ABC):
    """Defines how to change stability and train the model"""
    @abstractmethod
    def train(self,
              ost,
              experiences:  Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[Experience]]]] = None,
              **kwargs):
        """Executes the policy"""

    @abstractmethod
    def error(self, strategy_accuracy: float, reference_accuracy: float) -> float:
        """A policy dependent meaure of error, typically distance to margin"""

    @abstractmethod
    def name(self) -> str:
        """The name of the policy"""

    @abstractmethod
    def params(self) -> dict:
        """Return a dict of parameters"""


class SemiOnlineStabilityTuning(ABC):
    """Generic class for per-experience hyper-parameter optimization"""

    metric = "Top1_Acc_Epoch/train_phase/train_stream/Task000"

    def __init__(
            self,
            reference_strategy: Strategy,
            strategy: Strategy,
            policy: StabilityPolicy,
            log: LoggerWithNotes,
            label: str):
        """
        Online stability decay, decays stability until performance is comparable
        to a reference strategy (typically naive)

        Args:
            reference_strategy (Strategy): Strategy to use as a baseline for the
                expected performance
            strategy (Strategy): Strategy that has its stability updated
            update_stability_metric (Callable[[Strategy, float], None]): Callback
                to update stability
            decay_factor (float): How much should the stability be decayed by
            accuracy_drop_margin (float): Used to determine when to decay
        """

        self.label = label
        self.avl_strategy = self
        self.log = log
        self.ref_strategy = reference_strategy
        self.strategy = strategy
        self.counter = 0
        self.policy = policy

        if log:
            self.strategy.avl_strategy.evaluator.loggers.append(self.log)
            self.add_scalar = self.log.writer.add_scalar
        else:
            self.add_scalar = lambda x: x
        self.setup()

    @abstractmethod
    def setup(self):
        """Run after __init__"""
        return NotImplemented

    @abstractmethod
    def set_stability(self, stability: float):
        """Set the underlying strategy's stability"""
        return NotImplemented

    @abstractmethod
    def get_stability(self) -> float:
        """Get the underlying strategy's stability"""
        return NotImplemented

    @abstractmethod
    def get_parameters(self):
        """Return the underlying strategy's parameters"""
        return NotImplemented

    @abstractmethod
    def use_parameters(self, parameters):
        """Set the underlying strategy's parameters"""
        return NotImplemented

    def reset_ref_strategy(self):
        """Sync the reference strategy with the constrained one"""
        del self.ref_strategy.avl_strategy.model
        self.ref_strategy.avl_strategy.model = copy.deepcopy(
            self.strategy.avl_strategy.model)

    def sync_experience_counter(self, experience_counter: int):
        self.ref_strategy.avl_strategy.clock.train_exp_counter = experience_counter
        self.strategy.avl_strategy.clock.train_exp_counter = experience_counter

    def output_ost_metrics(self, ref_accuracy: float, accuracy: float):
        """Output useful metrics to understand how stability is being decay"""

        i = self.counter
        exp_counter = self.strategy.avl_strategy.training_exp_counter

        if not self.log:
            return

        self.add_scalar(
            f"Stability/exp_{exp_counter}", self.get_stability(), i)
        self.add_scalar(f"OST_Accuracy/exp_{exp_counter}", accuracy, i)
        self.add_scalar(
            f"Intransigence/exp_{exp_counter}", ref_accuracy - accuracy, i)
        self.add_scalar(
            f"OST_Error/exp_{exp_counter}",  self.policy.error(accuracy, ref_accuracy), i)
        self.add_scalar(
            f"OST_Margin/exp_{exp_counter}",  self.policy.error(0, ref_accuracy), i)

        self.add_scalar("OST/ref_accuracy", ref_accuracy, i)
        self.add_scalar("OST/accuracy", accuracy, i)
        self.add_scalar("OST/error", ref_accuracy - accuracy, i)
        self.add_scalar("OST/experience", exp_counter, i)
        self.add_scalar("OST/stability", self.get_stability(), i)
        self.add_scalar("OST/experience", exp_counter, i)
        self.log.flush()
        self.counter += 1

    def train(self,
              experiences:  Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[Experience]]]] = None,
              **kwargs):
        """Use a policy to train the underlying stratergies"""
        self.policy.train(self, experiences, eval_streams, **kwargs)

    def eval(self,
             exp_list: Union[Experience, Sequence[Experience]],
             **kwargs):
        """eval, evaluate the underlying strategies"""

        print(f"Eval Underlying {self.ref_strategy.label}")
        self.ref_strategy.avl_strategy.eval(exp_list, **kwargs)
        print(f"Eval Underlying {self.strategy.label}")
        self.strategy.avl_strategy.eval(exp_list, **kwargs)

    def flush(self):
        """Flush logs"""
        if self.log:
            self.log.flush()
        self.ref_strategy.flush()
        self.strategy.flush()


class StabilityDecay(StabilityPolicy):
    """Hyper-parameter optimization by decaying stability until performance is acceptable"""

    def __init__(self, decay_factor: float, drop_margin: float) -> None:
        self.decay_factor = decay_factor
        self.drop_margin = drop_margin

    def train(self,
              ost: SemiOnlineStabilityTuning,
              experiences:  Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[Experience]]]] = None,
              **kwargs):
        # Save parameters so that we can undo learning
        old_params = ost.get_parameters()
        ost.reset_ref_strategy()

        def _train(strategy: Strategy):
            ost.use_parameters(old_params)
            print(f"Training {strategy.label}")
            ost.sync_experience_counter(experiences.current_experience)
            return strategy.avl_strategy.train(experiences, eval_streams, **kwargs)[ost.metric]

        # Maximal Plasticity "Search" using a reference strategy
        ref_accuracy = _train(ost.ref_strategy)

        while True:
            accuracy = _train(ost.strategy)
            ost.output_ost_metrics(
                ref_accuracy, accuracy)

            if experiences.current_experience == 0:
                break

            # Update stability
            if accuracy < (1-self.drop_margin) * ref_accuracy:
                new_stability = ost.get_stability() * self.decay_factor
                print(
                    f"Decaying Stability {ost.get_stability()} -> {new_stability}")
                ost.set_stability(new_stability)
            else:
                ost.output_ost_metrics(ref_accuracy, accuracy)
                return

    def error(self, strategy_accuracy: float, reference_accuracy: float) -> float:
        return (1-self.drop_margin) * reference_accuracy - strategy_accuracy

    def params(self):
        return {"decay_factor": self.decay_factor, "drop_margin": self.drop_margin}

    def name(self) -> str:
        return "Decay"


class StabilityTune(StabilityPolicy):
    """Hyper-parameter optimization through decay and then growth"""

    def __init__(self, change_factor: float, drop_margin: float) -> None:
        self.change_factor = change_factor
        # assert 0 < change_factor < 1, "Change factor must be between 0 and 1"
        self.decay_factor = (1 - change_factor)
        self.growth_factor = 1 / self.decay_factor
        print(f"growth: {self.growth_factor}, decay: {self.decay_factor} ")
        self.drop_margin = drop_margin

    def train(self,
              ost: SemiOnlineStabilityTuning,
              experiences:  Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[Experience]]]] = None,
              **kwargs):
        # Save parameters so that we can undo learning
        old_parameters = ost.get_parameters()
        ost.reset_ref_strategy()

        def _train(strategy: Strategy):
            print(f"Training {strategy.label}")
            ost.use_parameters(old_parameters)
            ost.sync_experience_counter(experiences.current_experience)
            return strategy.avl_strategy.train(experiences, eval_streams, **kwargs)[ost.metric]

        # Find reference accuracy
        ref_accuracy = _train(ost.ref_strategy)

        # Wrap _train to output metrics automatically
        def train(strategy: Strategy):
            accuracy = _train(strategy)
            ost.output_ost_metrics(ref_accuracy, accuracy)
            return accuracy

        # Determin if the given accuracy is in the acceptable margin
        def in_margin(accuracy: float) -> bool:
            return accuracy > (1-self.drop_margin) * ref_accuracy

        accuracy = train(ost.strategy)

        # We do not do any tuning on the first experience
        if experiences.current_experience == 0:
            ost.output_ost_metrics(ref_accuracy, accuracy)
            return

        if in_margin(accuracy):
            # Grow stability until we leave the margin
            while in_margin(accuracy):
                best_parameters = ost.get_parameters()
                best_accuracy = accuracy

                new_stability = ost.get_stability() * self.growth_factor
                print(
                    f"Growing Stability {ost.get_stability()} -> {new_stability}")
                ost.set_stability(new_stability)
                accuracy = train(ost.strategy)

            # Use the best parameters that were barely inside the margin
            ost.set_stability(new_stability*self.decay_factor)
            ost.output_ost_metrics(ref_accuracy, best_accuracy)
            ost.use_parameters(best_parameters)
        else:
            # Decay stability until we enter the margin
            while not in_margin(accuracy):
                new_stability = ost.get_stability() * self.decay_factor
                print(
                    f"Decaying Stability {ost.get_stability()} -> {new_stability}")
                ost.set_stability(new_stability)
                accuracy = train(ost.strategy)

            # We use the paramaters that we found that are barely inside the margin

    def error(self, strategy_accuracy: float, reference_accuracy: float) -> float:
        return (1-self.drop_margin) * reference_accuracy - strategy_accuracy

    def params(self):
        return {"change_factor": round(self.change_factor, 2), "drop_margin": round(self.drop_margin, 2)}

    def name(self) -> str:
        return "Tune"
