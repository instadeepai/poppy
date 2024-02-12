from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import random
from jax.random import PRNGKey

from poppy.environments.poppy_env import PoppyEnv


def get_start_positions(
    environment, start_key, num_start_positions, num_problems, num_agents
):
    """Generate the starting positions for each problem-agent pair.

    Args:
        environment: The environment to prepare problems for.
        start_key: The key for generating the starting positions.
        num_start_positions: The number of start positions per problem (M).  If <0
          then all possible positions are used, i.e. M=N.
        num_problems: The number of problems to generate (N).
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        num_start_positions: The number of start positions per problem.
        starting_positions: M starting positions for each problem-agent pair ([N, K, M]).
    """
    if num_start_positions < 0:
        start_positions = jnp.arange(
            environment.get_min_start(), environment.get_max_start() + 1
        )
        start_positions = (
            start_positions[None, None].repeat(num_problems, 0).repeat(num_agents, 1)
        )
        num_start_positions = environment.get_problem_size()
    else:
        start_positions = random.randint(
            start_key,
            (num_problems, 1, num_start_positions),
            minval=environment.get_min_start(),
            maxval=environment.get_max_start() + 1,
        ).repeat(
            num_agents, axis=1
        )  # make sure agents have same starting keys

    return num_start_positions, start_positions


def get_acting_keys(act_key, num_start_positions, num_problems, num_agents):
    """Get the acting keys

    Args:
        act_key: The key for generating the acting keys.
        num_start_positions: The number of start positions per problem.
        num_problems: The number of problems to generate (N).
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        acting_key: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    acting_keys = random.split(
        act_key, num_problems * num_agents * num_start_positions
    ).reshape((num_problems, num_agents, num_start_positions, -1))

    return acting_keys


def prepare_problem_batch(
    prng_key: PRNGKey,
    environment: PoppyEnv,
    num_problems: int,
    num_agents: int,
    num_start_positions: int,
    duplicate_problems_on_each_device: bool = False,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Prepare a batch of problems.

    Args:
        prng_key: The key for generating this problem set.
        environment: The environment to prepare problems for.
        num_problems: The number of problems to generate (N).
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).
        num_start_positions: The number of start positions per problem (M).  If <0
          then all possible positions are used, i.e. M=N.
        duplicate_problems_on_each_device: Ensure the same problems are generated on each device.

    Returns:
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    problem_key, start_key, act_key = random.split(prng_key, 3)
    if duplicate_problems_on_each_device:
        # Distribute the problem key from the first device to all devices.
        problem_key = jax.lax.all_gather(problem_key, "i", axis=0)[0]
    problems = jax.vmap(environment.generate_problem, in_axes=(0, None))(
        random.split(problem_key, num_problems), environment.get_problem_size()
    )

    num_start_positions, start_positions = get_start_positions(
        environment, start_key, num_start_positions, num_problems, num_agents
    )

    acting_keys = get_acting_keys(
        act_key, num_start_positions, num_problems, num_agents
    )

    return problems, start_positions, acting_keys
