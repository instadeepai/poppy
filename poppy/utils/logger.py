from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Union

import neptune.new as neptune
import numpy as np
from acme.utils import loggers as acme_loggers

# An artifact is a mapping between a string and the path of a file to log
Path = str
Artifact = Mapping[str, Union[Path, Any]]  # recursive structure


class PoppyLogger(ABC):
    @abstractmethod
    def write(self, data: acme_loggers.LoggingData) -> None:
        pass

    @abstractmethod
    def write_artifact(self, artifact: Artifact) -> None:
        pass

    @abstractmethod
    def close(self):
        pass


class TerminalLogger(acme_loggers.TerminalLogger, PoppyLogger):
    def __init__(self, label: str, time_delta: float, **kwargs: Any):
        super(TerminalLogger, self).__init__(
            label=label, time_delta=time_delta, print_fn=print, **kwargs
        )

    def write_artifact(self, artifact: Artifact) -> None:
        pass


class NeptuneLogger(acme_loggers.Logger, PoppyLogger):
    def __init__(self, **kwargs: Any):
        super(NeptuneLogger, self).__init__()
        self.run = neptune.init(**{k: kwargs[k] for k in ["name", "project", "tags"]})
        self.run["parameters"] = kwargs["parameters"]

    def write(self, data: acme_loggers.LoggingData) -> None:
        for key, value in data.items():
            key_with_label = f"{key}"
            if not np.isscalar(value):
                value = float(value)
            self.run[key_with_label].log(value)

    def write_artifact(self, artifact: Artifact) -> None:
        for key, value in artifact.items():
            self.run[key].upload(value)

    def close(self) -> None:
        self.run.stop()


class EnsembleLogger(acme_loggers.Logger, PoppyLogger):
    def __init__(self, loggers: List):
        self.loggers = loggers

    def write(self, data: acme_loggers.LoggingData) -> None:
        for logger in self.loggers:
            logger.write(data)

    def write_artifact(self, artifact: Artifact) -> None:
        for logger in self.loggers:
            logger.write_artifact(artifact)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
