from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array


@dataclass
class State:
    """
    problem: array with the coordinates of all nodes (+ depot) and their cost
    position: index of the current node
    capacity: current capacity of the vehicle
    visited_mask:
    order: array of node indices denoting route (-1 --> not filled yet)
    num_total_visits: number of performed visits (it can count depot multiple times)
    """

    problem: Array  # (num_nodes + 1, 3)
    position: jnp.int32
    capacity: jnp.float32
    visited_mask: Array  # (num_nodes + 1,)
    order: Array  # (2 * num_nodes,) - the size is worst-case (going back to depot after visiting each node)
    num_total_visits: jnp.int32


class Observation(NamedTuple):
    """
    problem: array with the coordinates of all nodes (+ depot) and their cost
    position: index of the current node
    capacity: current capacity of the vehicle
    invalid_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: Array  # (num_nodes + 1, 3)
    position: jnp.int32
    capacity: jnp.float32
    action_mask: Array  # (num_nodes + 1,)
    is_done: bool
