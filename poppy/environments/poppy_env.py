from abc import ABC, abstractmethod
from typing import Union

from chex import Array

from poppy.environments.cvrp.types import Observation as CVRPObservation
from poppy.environments.knapsack.types import Observation as KnapsackObservation
from poppy.environments.tsp.types import Observation as TSPObservation

PoppyObservation = Union[TSPObservation, CVRPObservation, KnapsackObservation]


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

    @staticmethod
    @abstractmethod
    def generate_problem() -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def make_observation(*args, **kwargs) -> PoppyObservation:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_reward_negative() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_reward_string() -> str:
        return "return"
