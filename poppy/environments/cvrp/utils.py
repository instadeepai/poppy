from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random
from jumanji.environments.combinatorial.tsp.utils import get_augmentations as get_augmentations_tsp

DEPOT_IDX = 0
MIN_NORM_FACTOR = 10


def generate_problem(problem_key: PRNGKey, num_nodes: jnp.int32) -> Array:
    coords = random.uniform(problem_key, (num_nodes + 1, 2), minval=0, maxval=1)
    costs = random.randint(problem_key, (num_nodes + 1, 1), minval=1, maxval=10)
    problem = jnp.hstack((coords, costs))
    problem = problem.at[DEPOT_IDX, 2].set(0.0)
    return problem


def generate_start_node(start_key: PRNGKey, num_nodes: jnp.int32) -> jnp.int32:
    return random.randint(start_key, (), minval=1, maxval=num_nodes + 1)


def get_augmentations(problem: Array) -> Array:
    """
    Returns the 8 augmentations of a given instance problem described in [1]. This function leverages the existing
    augmentation method for TSP and appends the costs/demands used in CVRP.
    [1] https://arxiv.org/abs/2010.16011

    Args:
        problem: Array of coordinates and demands/costs for all nodes [num_nodes, 3]

    Returns:
        augmentations: Array with 8 augmentations [8, num_nodes, 3]
    """
    coord_augmentations = get_augmentations_tsp(problem[:, :2])

    num_nodes = problem.shape[0]
    num_augmentations = coord_augmentations.shape[0]

    costs_per_aug = jnp.tile(problem[:, 2], num_augmentations).reshape(
        num_augmentations, num_nodes, 1
    )
    return jnp.concatenate((coord_augmentations, costs_per_aug), axis=2)
