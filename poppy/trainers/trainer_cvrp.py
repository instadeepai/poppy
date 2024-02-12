from typing import Type

import haiku as hk
import jax.numpy as jnp
from chex import Array, PRNGKey

from poppy.environments.cvrp.types import Observation
from poppy.environments.cvrp.utils import DEPOT_IDX, generate_problem, get_augmentations
from poppy.nets import DecoderBase, EncoderBase
from poppy.trainers.trainer_base import TrainerBase


class CVRPEncoder(EncoderBase):
    # Modified for CVRP according to original source code: https://github.com/yd-kwon/POMO/blob/master/NEW_py_ver/CVRP/POMO/CVRPModel.py (~line 116)
    def get_problem_projection(self, problem: Array) -> Array:
        proj_depot = hk.Linear(128, name="depot_encoder")
        proj_nodes = hk.Linear(128, name="nodes_encoder")
        return jnp.where(
            jnp.zeros((problem.shape[0], 1)).at[DEPOT_IDX].set(1),
            proj_depot(problem),
            proj_nodes(problem),
        )


class CVRPDecoder(DecoderBase):
    def get_context(self, observation: Observation, embeddings: Array) -> Array:
        return jnp.concatenate(
            [
                embeddings.mean(0),
                embeddings[observation.position],
                observation.capacity[None],
            ],
            axis=0,
        )[
            None
        ]  # [1, 2*128+1=257,]

    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        return jnp.where(attention_mask, 0, 1)


class TrainerCVRP(TrainerBase):
    @staticmethod
    def get_reward_name() -> str:
        return "tour_length"

    @staticmethod
    def is_reward_positive() -> bool:
        return False

    @staticmethod
    def get_observation_type() -> Type[Observation]:
        return Observation

    def init_encoder(self, num_layers, name) -> EncoderBase:
        return CVRPEncoder(num_layers, name)

    def init_decoder(self, name) -> DecoderBase:
        return CVRPDecoder(name)

    @staticmethod
    def generate_problem(key: PRNGKey, problem_size: jnp.int32) -> Array:
        return generate_problem(key, problem_size)

    @staticmethod
    def use_augmentations() -> bool:
        return True

    @staticmethod
    def get_augmentations(problem: Array) -> Array:
        return get_augmentations(problem)

    def has_symmetric_starting_points(self) -> bool:
        return False
