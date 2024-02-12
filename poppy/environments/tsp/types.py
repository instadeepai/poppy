from chex import Array
import jax.numpy as jnp
from typing import NamedTuple


class Observation(NamedTuple):
    """
    problem: array of coordinates for all cities
    start_position: index of starting city
    position: index of current city
    action_mask: binary mask (0/1 <--> legal/illegal)
    """
    problem: Array
    start_position: jnp.int32
    position: jnp.int32
    action_mask: Array
    is_done: bool
