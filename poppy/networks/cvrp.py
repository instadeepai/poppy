import haiku as hk
import jax.numpy as jnp
from chex import Array

from poppy.environments.cvrp.types import Observation as CVRPObservation
from poppy.environments.cvrp.utils import DEPOT_IDX
from poppy.networks.base import DecoderBase, EncoderBase


class CVRPEncoder(EncoderBase):
    # Modified for CVRP according to original source code: https://github.com/yd-kwon/POMO/blob/master/NEW_py_ver/CVRP/POMO/CVRPModel.py (~line 116)
    def get_problem_projection(self, problem: Array) -> Array:
        proj_depot = hk.Linear(self.model_size, name="depot_encoder")
        proj_nodes = hk.Linear(self.model_size, name="nodes_encoder")
        return jnp.where(
            jnp.zeros((problem.shape[0], 1)).at[DEPOT_IDX].set(1),
            proj_depot(problem),
            proj_nodes(problem),
        )


class CVRPDecoder(DecoderBase):
    def get_context(self, observation: CVRPObservation, embeddings: Array) -> Array:  # type: ignore[override]
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
