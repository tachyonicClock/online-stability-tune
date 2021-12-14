import torch
from torch.functional import Tensor
from avalanche.evaluation import Metric
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name


class PerClassAccuracy(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self.correct = dict() # Number correct
        self.seen = dict()    # Number seen

    def update(self, predicted_y: Tensor, true_y: Tensor):
        # predicted_y = predicted_y.to("cpu")
        # true_y = true_y.to("cpu")

        for pred, actual in zip(predicted_y, true_y):
            pred, actual = torch.argmax(pred, dim=0, keepdim=True), int(actual)
            # Add key if it does not exist
            if actual not in self.correct:
                self.correct[actual] = 0
            if actual not in self.seen:
                self.seen[actual] = 0

            self.seen[actual] += 1
            if pred == actual:
                self.correct[actual] += 1

    def reset(self):
        """
        Reset your metric
        """
        self.correct = dict()
        self.seen = dict()

class AccuracyOnTrainedClasses(PluginMetric[float]):
    """
Calculate accuracy only for classes that the model has been trained on
    """
    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        self._accuracy_metric = PerClassAccuracy()
        self._trained_classes = set()
    def reset(self) -> None:
        """Reset the metric"""
        self._accuracy_metric.reset()

    def result(self) -> float:
        """Calculate accuracy only for classes that the model has been trained on"""
        total_correct = sum([self._accuracy_metric.correct[label] for label in self._trained_classes])
        total_seen = sum([self._accuracy_metric.seen[label] for label in self._trained_classes])
        return total_correct/total_seen

    def after_eval_iteration(self, strategy) -> None:
        self._accuracy_metric.update(strategy.mb_output, strategy.mb_y)
    
    def before_eval(self, strategy):
        self._accuracy_metric.reset()

    def after_eval(self, strategy):
        return self._package_result(strategy)

    def after_training_iteration(self, strategy) -> None:
        """Update the classes the model has been trained on"""
        self._trained_classes.update(strategy.experience.classes_in_this_experience)

    def _package_result(self, strategy):
        """Package the result to be output"""
        return [MetricValue(self, get_metric_name(self, strategy), self.result(),
                                strategy.clock.train_iterations)]
    def __str__(self):
        """Name the metric"""
        return "Accuracy_On_Trained_Classes"