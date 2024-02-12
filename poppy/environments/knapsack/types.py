from typing import NamedTuple, TYPE_CHECKING
if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
import chex


@dataclass
class State:
    """
    weights: array of weights of the items.
    values: array of values of the items.
    packed_items: binary mask denoting which items are already packed into the knapsack.
    remaining_budget: the budget currently remaining.
    key: random key used for auto-reset.
    """

    weights: chex.Array  # (num_items,)
    values: chex.Array  # (num_items,)
    packed_items: chex.Array  # (num_items,)
    remaining_budget: chex.Array  # ()
    key: chex.PRNGKey  # (2,)
    num_steps: jnp.int32  # ()


class Observation(NamedTuple):
    """
    problem: array of weights/values of the items
    start_position: index of first added item (useless, but Pomo does it to match TSP environment)
    position: index of the last added item
    action_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: chex.Array
    start_position: jnp.int32
    position: jnp.int32
    action_mask: chex.Array
    is_done: jnp.int32
