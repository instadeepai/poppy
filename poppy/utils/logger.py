from acme.utils import loggers as acme_loggers
from typing import Any, Dict
import yaml


class TerminalLogger(acme_loggers.TerminalLogger):
    def __init__(self, label: str, time_delta: float, **kwargs: Any):
        super(TerminalLogger, self).__init__(label=label, time_delta=time_delta, print_fn=print, **kwargs)

    def write_config(self, config: Dict) -> None:
        self._print_fn(f"Config:\n {yaml.dump(config)}")
