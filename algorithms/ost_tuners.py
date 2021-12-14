import copy
from typing import Sequence, Union
from avalanche.benchmarks.scenarios.generic_definitions import Experience
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.lwf import LwFPlugin
from avalanche.training.plugins.synaptic_intelligence import SynapticIntelligencePlugin
from avalanche.training.strategies.strategy_wrappers import Naive, SynapticIntelligence
from moacl.algorithms.ost import SemiOnlineStabilityTuning
from moacl.strategies.ewc import EWC, EWCPlugin
from moacl.strategies.lwf import LwF


def get_tuner(strategy_type):
    tuners = {
        EWC: EWCTuner,
        SynapticIntelligence: SITuner,
        LwF: LWFTuner
    }
    return tuners[strategy_type]

class LWFTuner(SemiOnlineStabilityTuning):
    """Learning Without Forgetting Semi-Online Stability Tuning"""

    def setup(self):
        strategy = self.strategy
        self.lwf = strategy.avl_strategy
        self.lwf_plugin = next(
            p for p in self.lwf.plugins if isinstance(p, LwFPlugin))

    def set_stability(self, stability: float):
        self.lwf_plugin.alpha = stability

    def get_stability(self) -> float:
        return self.lwf_plugin.alpha

    def use_parameters(self, parameters):
        del self.lwf_plugin.prev_model
        del self.lwf.model
        del self.lwf_plugin.prev_classes
        self.lwf_plugin.prev_classes = copy.deepcopy(parameters["prev_classes"])
        self.lwf_plugin.prev_model = copy.deepcopy(parameters["prev_model"])
        self.lwf.model = copy.deepcopy(parameters["model"])

    def get_parameters(self):
        return {
            "prev_classes": copy.deepcopy(self.lwf_plugin.prev_classes),
            "prev_model": copy.deepcopy(self.lwf_plugin.prev_model),
            "model": copy.deepcopy(self.lwf.model)}


class EWCTuner(SemiOnlineStabilityTuning):
    """Elastic Weight Consolidation Semi-Online Stability Tuning"""

    def setup(self):
        strategy = self.strategy
        assert isinstance(strategy.avl_strategy,
                          EWC), "Strategy should be ewc"

        self.ewc = strategy.avl_strategy
        self.ewc_plugin = next(p for p in self.ewc.plugins if isinstance(p, EWCPlugin))
        self.clock = next(p for p in self.ewc.plugins if isinstance(p, Clock))
        return super().setup()

    def set_stability(self, stability: float):
        self.ewc_plugin.ewc_lambda = stability

    def get_stability(self) -> float:
        return self.ewc_plugin.ewc_lambda

    def get_parameters(self):
        return {
            "model": copy.deepcopy(self.ewc.model),
            "importances": copy.deepcopy(self.ewc_plugin.importances),
            "saved_params": copy.deepcopy(self.ewc_plugin.saved_params),
        }

    def use_parameters(self, parameters):
        del self.ewc.model
        del self.ewc_plugin.importances
        del self.ewc_plugin.saved_params
        self.ewc.model = copy.deepcopy(parameters["model"])
        self.ewc_plugin.importances = copy.deepcopy(parameters["importances"])
        self.ewc_plugin.saved_params = copy.deepcopy(parameters["saved_params"])



class SITuner(SemiOnlineStabilityTuning):
    """Synaptic Intelligence Semi-Online Stability Tuning"""

    def setup(self):
        strategy = self.strategy
        assert isinstance(
            strategy.avl_strategy, SynapticIntelligence), \
            "Strategy should be SynapticIntelligence"
            
        self.si = strategy.avl_strategy
        self.si_plugin: SynapticIntelligencePlugin \
            = next(p for p in self.si.plugins if isinstance(p, SynapticIntelligencePlugin))

    def set_stability(self, stability: float):
        self.si_plugin.si_lambda = [stability]

    def get_stability(self) -> float:
        return self.si_plugin.si_lambda[0]

    def get_parameters(self):
        return {
            "syn_data": copy.deepcopy(self.si_plugin.syn_data),
            "model": copy.deepcopy(self.si.model),
            "ewc_data": copy.deepcopy(self.si_plugin.ewc_data)
        }

    def use_parameters(self, parameters):
        del self.si_plugin.syn_data
        del self.si.model
        del self.si_plugin.ewc_data  
        self.si_plugin.syn_data = copy.deepcopy(parameters["syn_data"])
        self.si.model = copy.deepcopy(parameters["model"])
        self.si_plugin.ewc_data = copy.deepcopy(parameters["ewc_data"])
