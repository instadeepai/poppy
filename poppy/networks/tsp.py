import haiku as hk
import jax.numpy as jnp
from chex import Array

from poppy.environments.tsp.types import Observation as TSPObservation
from poppy.networks.base import DecoderBase, EncoderBase


class TSPEncoder(EncoderBase):
    def get_problem_projection(self, problem: Array) -> Array:
        proj = hk.Linear(self.model_size, name="encoder")
        return proj(problem)


class TSPDecoder(DecoderBase):
    def get_context(self, observation: TSPObservation, embeddings: Array) -> Array:  # type: ignore[override]
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
        return attention_mask
