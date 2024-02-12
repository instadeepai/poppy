from chex import Array
import jax.numpy as jnp
from typing import NamedTuple


class Observation(NamedTuple):
    """
    problem: array of weights/values of the items
    start_position: index of first added item (useless, but Pomo does it to match TSP environment)
    position: index of the last added item
    action_mask: binary mask (0/1 <--> legal/illegal)
    """
    problem: Array
    start_position: jnp.int32
    position: jnp.int32
    action_mask: Array
    is_done: bool
