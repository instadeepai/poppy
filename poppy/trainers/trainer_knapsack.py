from chex import Array, PRNGKey
import haiku as hk
import jax.numpy as jnp
from jumanji.environments.combinatorial.knapsack.utils import generate_problem
from typing import Type

from poppy.environments.knapsack.types import Observation
from poppy.nets import DecoderBase, EncoderBase
from poppy.trainers.trainer_base import TrainerBase


class KnapsackEncoder(EncoderBase):
    def get_problem_projection(self, problem: Array) -> Array:
        proj = hk.Linear(128, name="encoder")
        return proj(problem)


class KnapsackDecoder(DecoderBase):
    def get_context(self, observation: Observation, embeddings: Array) -> Array:
        return jnp.concatenate(
            [
                embeddings.mean(0),
                embeddings[observation.position],
                embeddings[observation.start_position],
            ],
            axis=0,
        )[
            None
        ]  # [1, 3*128=384,]

    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        return jnp.where(attention_mask, 0, 1)


class TrainerKnapsack(TrainerBase):
    @staticmethod
    def get_reward_name() -> str:
        return "value_items"

    @staticmethod
    def is_reward_positive() -> bool:
        return True

    @staticmethod
    def get_observation_type() -> Type[Observation]:
        return Observation

    def init_encoder(self, num_layers, name) -> EncoderBase:
        return KnapsackEncoder(num_layers, name)

    def init_decoder(self, name) -> DecoderBase:
        return KnapsackDecoder(name)

    @staticmethod
    def generate_problem(key: PRNGKey, problem_size: jnp.int32) -> Array:
        return generate_problem(key, problem_size)

    @staticmethod
    def use_augmentations() -> bool:
        return False

    @staticmethod
    def get_augmentations(problem: Array) -> Array:
        return problem[None]

    def has_symmetric_starting_points(self) -> bool:
        return False
