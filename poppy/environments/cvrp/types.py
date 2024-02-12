from typing import NamedTuple
import jax.numpy as jnp
from chex import Array


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
    is_done: jnp.int32
