from typing import NamedTuple
import chex
import jax.numpy as jnp


class Observation(NamedTuple):
    """
    coordinates: array of 2D coordinates for all cities.
    position: index of current city.
    trajectory: array of city indices defining the route (-1 --> not filled yet).
    action_mask: binary mask (False/True <--> illegal/legal).
    """

    problem: chex.Array  # (num_cities, 2)
    start_position: jnp.int32
    position: chex.Numeric  # ()
    trajectory: chex.Array  # (num_cities,)
    action_mask: chex.Array  # (num_cities,)
    is_done: jnp.int32
