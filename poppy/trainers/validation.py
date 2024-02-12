import functools
import os
import pickle

import chex
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf

import poppy.trainers.trainer as trainer
from poppy.utils.data import (
    get_acting_keys,
    get_start_positions,
    prepare_problem_batch,
)
from poppy.utils.utils import spread_over_devices
from poppy.utils.metrics import get_metrics


def get_params(cfg: omegaconf.DictConfig):
    """Load the encoder and decoder parameters from checkpoint.

    Args:
        cfg: The config containing the checkpointing information.

    Returns: The encoder and decoder parameters.
    """
    cfg.checkpoint_fname_load = os.path.splitext(cfg.checkpoint_fname_load)[0]
    if cfg.restore_path:
        with open(
            os.path.join(cfg.restore_path, cfg.checkpoint_fname_load + ".pkl"), "rb"
        ) as f:
            saved_state = pickle.load(f)
        return saved_state.params
    else:
        raise ValueError(
            "Set 'checkpointing.restore_path' in config to the path containing the checkpoint"
        )


def load_instances(cfg, key, environment, num_start_positions, num_agents):
    """Load problems instances from the given file and generate start positions and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: The PRNGKey for generating the starting positions and acting keys.
        environment: The environment to generate the starting positions on.
        num_start_positions: The number of starting positions to generate.
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    with open(cfg.load_path, "rb") as f:
        problems = jnp.array(pickle.load(f))

    start_key, act_key = random.split(key, 2)
    num_start_positions, start_positions = get_start_positions(
        environment, start_key, num_start_positions, problems.shape[0], num_agents
    )
    acting_keys = get_acting_keys(
        act_key, num_start_positions, problems.shape[0], num_agents
    )

    return problems, start_positions, acting_keys


def get_instances(cfg, key, environment, params, num_start_positions):
    """Get the problem instances, start positions, and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: A PRNGKey.
        environment: The environment to generate the starting positions on.
        params: The encoder and decoder parameters.
        num_start_positions: The number of starting positions to generate.

    Returns:
        problems: A batch of N problems divided over D devices ([D, N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair divided over D devices
        ([D, N, K, M]).
        acting_keys: M acting keys for each problem-agent pair divided over D devices
        ([D, N, K, M, 2]).
    """
    _, decoder = hk.data_structures.partition(lambda m, n, p: "encoder" in m, params)
    num_agents = jax.tree_util.tree_leaves(decoder)[0].shape[0]

    if cfg.load_problem:
        problems, start_positions, acting_keys = load_instances(
            cfg, key, environment, num_start_positions, num_agents
        )
    else:
        problems, start_positions, acting_keys = prepare_problem_batch(
            key, environment, cfg.num_problems, num_agents, num_start_positions
        )

    problems = spread_over_devices(problems)
    start_positions = spread_over_devices(start_positions)
    acting_keys = spread_over_devices(acting_keys)

    return problems, start_positions, acting_keys


def validate(
    cfg: omegaconf.DictConfig,
    params: chex.ArrayTree = None,
) -> dict:
    """Run validation on input problems.

    Args:
        cfg: The config for validation.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.

    Returns:
        metrics: A dictionary of metrics from the validation.
    """
    if cfg.rollout.decoder_pmap_axis == "pop":
        # TODO: Handle metric collection in this case.
        raise NotImplementedError

    @functools.partial(jax.pmap, axis_name="i")
    def run_validate(problems, start_positions, acting_keys):
        """Run the rollout on a batch of problems and return the episode return.

        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).

        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """
        # 1. Split problems, start_positions and acting_keys into chunks of size batch_size.
        # 2. Zip batches into list of inputs:
        #   [(problems[0],start_positions[0],acting_keys[0]),
        #    (problems[1],start_positions[1],acting_keys[1]),
        #    ...]
        num_batches = int(round(len(problems) / cfg.batch_size, 0))
        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)
        start_positions = jnp.stack(jnp.split(start_positions, num_batches, axis=0))
        acting_keys = jnp.stack(jnp.split(acting_keys, num_batches, axis=0))
        num_problems = problems.shape[1]

        if cfg.use_augmentations:
            problems = jax.vmap(jax.vmap(environment.get_augmentations))(problems)
            problems = problems.reshape(
                num_batches, num_problems * 8, environment.get_problem_size(), -1
            )
            # Note, the starting positions and acting keys are duplicated here.
            start_positions = jnp.repeat(start_positions, 8, axis=1)
            acting_keys = jnp.repeat(acting_keys, 8, axis=1)

        def body(_, x):
            problems, start_positions, acting_keys = x
            _, (traj, info) = trainer.rollout(
                cfg=cfg.rollout,
                environment=environment,
                params=params,
                networks=networks,
                problems=problems,
                start_positions=start_positions,
                acting_keys=acting_keys,
            )
            info.metrics["rewards"] = traj.reward
            return None, info.metrics

        _, metrics = jax.lax.scan(
            body, init=None, xs=(problems, start_positions, acting_keys)
        )

        if cfg.use_augmentations:
            num_agents, num_start_positions = (
                start_positions.shape[-2],
                start_positions.shape[-1],
            )
            metrics = jax.tree_map(
                lambda x: x.reshape(
                    num_batches,
                    num_problems,
                    8,
                    num_agents,
                    num_start_positions,
                    -1,
                ).max(2),
                metrics,
            )

        # Flatten batch dimension of metrics.
        metrics = jax.tree_map(lambda x: x.reshape(*(-1,) + x.shape[2:]), metrics)
        episode_return = metrics["rewards"].sum(-1)  # [N, K, M]
        return episode_return

    networks = trainer.get_networks(cfg.networks)
    environment = hydra.utils.instantiate(cfg.environment)
    if not params:
        params = get_params(cfg.checkpointing)

    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems, key, environment, params, cfg.num_starting_points
    )

    episode_return = run_validate(problems, start_positions, acting_keys)
    episode_return = jnp.concatenate(episode_return, axis=0)

    if environment.is_reward_negative():
        ret_sign = -1
    else:
        ret_sign = 1
    return_str = environment.get_reward_string()

    # Make new metrics dictionary which will be all the returned statistics.
    metrics = {
        f"{return_str}": ret_sign * episode_return.max((-1, -2)).mean(),
        f"{return_str}_rand_agent": ret_sign * episode_return.max(-1).mean(),
        f"{return_str}_rand_start": ret_sign * episode_return.max(-2).mean(),
        f"{return_str}_rand_agent+start": ret_sign * episode_return.mean(),
    }

    metrics = get_metrics(
        metrics, episode_return, compute_expensive_metrics=cfg.compute_expensive_metrics
    )

    return metrics
