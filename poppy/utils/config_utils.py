from typing import Dict, Tuple, Type, TYPE_CHECKING
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

from poppy.environments.cvrp.environment import PoppyCVRP
from poppy.environments.knapsack.environment import PoppyKnapsack
from poppy.environments.poppy_env import PoppyEnv
from poppy.environments.tsp.environment import PoppyTSP
from poppy.trainers import TrainerCVRP, TrainerKnapsack, TrainerTSP
from poppy.trainers.trainer_base import TrainerBase


@dataclass
class EnvironmentConfig:  # type: ignore
    name: str
    params: Dict


def make_env_trainer(config: EnvironmentConfig) -> Tuple[PoppyEnv, Type[TrainerBase]]:
    if config.name == "cvrp":
        return (
            PoppyCVRP(
                num_nodes=config.params["num_nodes"],
                norm_factor=config.params["norm_factor"],
            ),
            TrainerCVRP,
        )
    elif config.name == "knapsack":
        return (
            PoppyKnapsack(
                problem_size=config.params["num_items"],
                total_budget=config.params["total_budget"],
            ),
            TrainerKnapsack,
        )
    elif config.name == "tsp":
        return (
            PoppyTSP(
                problem_size=config.params["num_cities"],
            ),
            TrainerTSP,
        )
    else:
        raise RuntimeError(f"Error: Unknown environment '{config.name}'.")
