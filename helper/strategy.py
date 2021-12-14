
from os import path
import json
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.training.strategies.base_strategy import BaseStrategy


class LoggerWithNotes(TensorboardLogger):
    """Wrap tensorboard logger so that we can add notes about paramaters"""

    def __init__(self,
                 strategy_label: str,
                 tb_log_dir: str = "tb_data") -> None:

        self.parameters = dict()
        self.strategy_label = strategy_label

        super().__init__(tb_log_dir=path.join(tb_log_dir, self.get_run_name()))

    def note_paramater(self, param_type: str, kwargs: dict):
        """Makes a note of a paramater as a JSON note"""
        self.parameters[param_type] = kwargs

    def save(self):
        """Save parameters to json in tensorboard"""
        dump = json.dumps(self.parameters, indent=1, default=str)
        self.writer.add_text("params", "```\n"+dump+"\n```", global_step=0)
        self.writer.flush()

    def get_run_name(self) -> str:
        """get the name of the run"""
        return self.strategy_label

    def flush(self):
        """Flush tensorboard"""
        self.writer.flush()

class Strategy():
    """Container for my strategies"""
    label: str
    avl_strategy: BaseStrategy
    log: LoggerWithNotes

    def __init__(self, label=None, avl_strategy=None, log=None) -> None:
        self.label = label
        self.avl_strategy = avl_strategy
        self.log = log

    def __str__(self):
        return str(self.label)

    def flush(self):
        """flush logs"""
        if self.log is not None:
            self.log.flush()
