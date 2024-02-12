from abc import ABC, abstractmethod


class PoppyEnv(ABC):
    @abstractmethod
    def get_problem_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_min_start(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_max_start(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_episode_horizon(self) -> int:
        raise NotImplementedError
