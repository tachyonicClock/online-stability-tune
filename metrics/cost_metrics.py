from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from algorithms.cost import AbstractCyberneticOST

class COSTMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self.x_coord = 0

    def reset(self) -> None:
        self._current_penalty = 0.0

    def result(self) -> float:
        return self._current_penalty

    def after_training_iteration(self, strategy) -> None:
        
        if not isinstance(strategy, AbstractCyberneticOST):
            return

        self.x_coord += 1
        reference_accuracy = strategy.reference_strategy.mb_accuracy.result()
        strategy_accuracy = strategy.mb_accuracy.result()
        intransigence = (reference_accuracy - strategy_accuracy)
        stability = strategy.get_stability()

        exp = strategy.training_exp_counter

        return [
            MetricValue(self, f'COST_Intransigence/exp_{exp}', float(intransigence), self.x_coord),
            MetricValue(self, f'COST_Stability/exp_{exp}', float(stability), self.x_coord),
            MetricValue(self, f'COST_Error/exp_{exp}', float(strategy.controller.pid.last_error), self.x_coord),

            MetricValue(self, "COST/Reference", float(reference_accuracy), self.x_coord),
            MetricValue(self, "COST/Strategy", float(strategy_accuracy), self.x_coord),
            MetricValue(self, 'COST/Intransigence', float(intransigence), self.x_coord),
            MetricValue(self, 'COST/Stability', float(stability), self.x_coord),
            MetricValue(self, 'COST/Experience', float(strategy.training_exp_counter), self.x_coord),
            MetricValue(self, 'COST/Integral', float(strategy.get_intransigence()), self.x_coord),
            MetricValue(self, 'COST/Derivative', float(strategy.controller.pid.derivative), self.x_coord),
            MetricValue(self, 'COST/Error', float(strategy.controller.pid.last_error), self.x_coord),
        ]

    def __str__(self):
        return "COSTMetric"