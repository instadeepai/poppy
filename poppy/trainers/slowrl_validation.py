import functools
from typing import Any, Optional, Tuple

import chex
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf
from jumanji.types import TimeStep

import poppy.trainers.trainer as trainer
from poppy.environments.poppy_env import PoppyEnv
from poppy.networks import Networks
from poppy.trainers.trainer import ActingState, Information, generate_trajectory
from poppy.trainers.validation import get_instances, get_params
from poppy.utils.data import get_acting_keys
from poppy.utils.metrics import get_metrics
from poppy.utils.utils import spread_over_devices


def slowrl_rollout(
    cfg: omegaconf.DictConfig,
    environment: PoppyEnv,
    params: chex.ArrayTree,
    networks: Networks,
    problems: jnp.ndarray,
    start_positions: jnp.ndarray,
    acting_keys: jnp.ndarray,
    agent_indices: jnp.ndarray,
) -> Tuple[ActingState, Tuple[TimeStep, Information]]:
    """Rollout a batch of agents on a batch of problems and starting points.
    Args:
        cfg: The rollout config.
        environment: The environment to rollout.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
            across all agents. There is only one decoder in the case of conditioned decoder. A population
            is implicitely created by the use of several behavior markers as input to the decoder.
        networks: The required networks.
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
        agent_indices: indices of the agents to use (N, K)
    Returns:
        # TODO
    """

    # split the params in encoder and decoder - those a merged in the training state
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # initialise the embeddings for each problem
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    print("Agent indices - in slowrl rollout: ", agent_indices.shape)
    print("Problems - in slowrl rollout: ", problems.shape)

    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0, 0))  # over N problems
    def update_and_generate_trajectory(
        problem, embeddings, decoder_params, start_position, acting_key, agent_indices
    ):
        # update the decoder params according to the agent indices
        jax.tree_util.tree_map(
            lambda x: print("Layer shape: ", x.shape), decoder_params
        )

        print("Agent indices - just before being used: ", agent_indices.shape)

        decoder_params = jax.tree_util.tree_map(
            lambda x: jnp.take(x, agent_indices, axis=0), decoder_params
        )
        jax.tree_util.tree_map(
            lambda x: print("Layer shape: ", x.shape), decoder_params
        )

        @functools.partial(
            jax.vmap, in_axes=(None, None, 0, 0, 0)
        )  # over K agents - behaviors
        @functools.partial(
            jax.vmap, in_axes=(None, None, None, 0, 0)
        )  # M starting pos.
        def generate_trajectory_fn(
            problem,
            embeddings,
            decoder_params,
            start_position,
            acting_key,
        ):
            return generate_trajectory(
                networks.decoder_fn.apply,
                cfg.policy.temperature,
                environment,
                problem,
                embeddings,
                decoder_params,
                start_position,
                acting_key,
            )

        return generate_trajectory_fn(
            problem, embeddings, decoder_params, start_position, acting_key
        )

    # generate the traj
    acting_state, (traj, info) = update_and_generate_trajectory(
        problems,
        embeddings,
        decoder_params,
        start_positions,
        acting_keys,
        agent_indices,
    )

    return acting_state, (traj, info)


def slowrl_validate(
    random_key,
    cfg: omegaconf.DictConfig,
    params: chex.ArrayTree = None,
    logger: Any = None,
) -> dict:
    """Run validation on input problems.
    Args:
        cfg: The config for validation.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.
    Returns:
        metrics: A dictionary of metrics from the validation.
    """

    def log(metrics, used_budget, logger, key=None):
        metrics["used_budget"] = used_budget
        if logger:
            if key:
                metrics = {f"{key}/{k}": v for (k, v) in metrics.items()}
            logger.write(metrics)

    if cfg.rollout.decoder_pmap_axis == "pop":
        # TODO: Handle metric collection in this case.
        raise NotImplementedError

    @functools.partial(jax.pmap, axis_name="i")
    def run_validate(problems, start_positions, acting_keys, agent_indices):
        """Run the rollout on a batch of problems and return the episode return.
        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
            agent_indices: indices of the agent to use in the population [N, K]
        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """
        # split problems, start_positions and acting_keys into chunks of size batch_size.
        num_batches = int(round(len(problems) / cfg.batch_size, 0))

        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)
        start_positions = jnp.stack(jnp.split(start_positions, num_batches, axis=0))
        acting_keys = jnp.stack(jnp.split(acting_keys, num_batches, axis=0))
        agent_indices = jnp.stack(jnp.split(agent_indices, num_batches, axis=0))

        num_problems = problems.shape[1]

        if cfg.use_augmentations:
            problems = jax.vmap(jax.vmap(environment.get_augmentations))(problems)

            problems = problems.reshape(
                num_batches, num_problems * 8, environment.get_problem_size(), -1
            )

            # Note, the starting positions and acting keys are duplicated here.
            start_positions = jnp.repeat(start_positions, 8, axis=1)
            acting_keys = jnp.repeat(acting_keys, 8, axis=1)
            agent_indices = jnp.repeat(agent_indices, 8, axis=1)

        def body(_, x):
            problems, start_positions, acting_keys, agent_indices = x
            _, (traj, info) = slowrl_rollout(
                cfg=cfg.rollout,
                environment=environment,
                params=params,
                networks=networks,
                problems=problems,
                start_positions=start_positions,
                acting_keys=acting_keys,
                agent_indices=agent_indices,
            )
            info.metrics["rewards"] = traj.reward
            return None, info.metrics

        _, metrics = jax.lax.scan(
            body,
            init=None,
            xs=(problems, start_positions, acting_keys, agent_indices),
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
                ).max(
                    2
                ),  # max on the pb augmentation dimension
                metrics,
            )

        # flatten batch dimension of metrics
        metrics = jax.tree_map(lambda x: x.reshape(*(-1,) + x.shape[2:]), metrics)
        episode_return = metrics["rewards"].sum(-1)  # [N, K, M]

        return episode_return

    # instantiate networks and environments
    networks = trainer.get_networks(cfg.networks)
    environment = hydra.utils.instantiate(cfg.environment)
    if not params:
        params = get_params(cfg.checkpointing)

    # define the number of starting points
    if cfg.num_starting_points < 0:
        num_starting_points = environment.get_problem_size()
    else:
        num_starting_points = cfg.num_starting_points

    # get a set of instances
    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
    )

    num_agents = acting_keys.shape[2]  # (num_devices, N/num_devices, K, M, 2)
    agent_indices = jnp.expand_dims(
        jnp.arange(start=0, stop=num_agents), axis=0
    ).repeat(repeats=cfg.problems.num_problems, axis=0)

    print("Agent indices: ", agent_indices.shape)

    # from now one, we want to use optimally a given budget
    budget = cfg.budget * num_starting_points

    # replicate them over the devices
    devices = jax.local_devices()

    # get shape
    shp = agent_indices.shape
    # split the parameters to put them on the different devices
    agent_indices = list(
        agent_indices.reshape(cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:])
    )

    agent_indices = jax.device_put_sharded(agent_indices, devices)

    print("Agent indices: ", agent_indices.shape)
    print("Problems: ", problems.shape)

    # put while loop here - keep all values
    used_budget = 0
    best_episode_return = None

    use_poppy_strategy = cfg.use_poppy_strategy

    while used_budget < budget:
        # update the acting keys - for stochasticity
        new_acting_keys = cfg.new_acting_keys
        if new_acting_keys:
            random_key, subkey = jax.random.split(random_key)
            acting_keys = get_acting_keys(
                subkey,
                num_starting_points,
                cfg.problems.num_problems,
                num_agents,
            )
            acting_keys = spread_over_devices(acting_keys, devices=devices)

        # run the validation episodes
        episode_return = run_validate(
            problems, start_positions, acting_keys, agent_indices
        )
        episode_return = jnp.concatenate(episode_return, axis=0)

        if use_poppy_strategy and used_budget == 0:
            # flatten all the returns obtained (for each episode)
            problems_episode_return = jax.vmap(jnp.ravel)(episode_return)  # [N, K*M]

            # sort agent/starting points pairs from the first evaluation
            problems_sorted_indices = jax.vmap(jnp.argsort)(
                -problems_episode_return
            )  # [N, K*M]

            # unravel those indices - (gives them asif the returns had not been flattened)
            sorted_indices = jax.vmap(
                functools.partial(jnp.unravel_index, shape=episode_return.shape[-2:])
            )(
                problems_sorted_indices
            )  # ([N, K*M], [N, K*M])

            # get the sorted indices
            sorted_indices_agent, sorted_indices_start_position = sorted_indices

            max_length = sorted_indices_agent.shape[1]

            # extract only the K best out of K*M
            sorted_indices_agent = sorted_indices_agent[:, :num_agents, ...]  # [N, K]
            sorted_indices_start_position = sorted_indices_start_position[
                :, :num_agents, ...
            ]  # [N, K]

            repeats = int(jnp.ceil(max_length / num_agents))  # = M

            # repeat best starting positions to get the same number of rollouts as before
            sorted_indices_start_position_repeated = jnp.repeat(
                sorted_indices_start_position, repeats=repeats, axis=1
            )[
                :, :max_length, ...
            ]  # make sure not to overlap - [N, K*M]

            # put start position in same shape as the sorted indices
            noshard_start_positions = jnp.concatenate(list(start_positions), axis=0)
            flat_start_positions = jax.vmap(jnp.ravel)(noshard_start_positions)

            # functions to extract the corresponding agents and starting points
            take_array_indices_2 = lambda arr, x: arr[x]

            # extract the starting points that got the best perfs (w/ those behaviors)
            desired_start_positions = jax.vmap(take_array_indices_2)(
                flat_start_positions,
                sorted_indices_start_position_repeated,
            )  # [N, K*M] (with repeated starting positions)

            # reshape start position
            desired_start_positions = jnp.reshape(
                desired_start_positions, noshard_start_positions.shape
            )  # [N, K, M]

            # # re-arrange for devices
            # get shape
            shp = sorted_indices_agent.shape

            # split the parameters to put them on the different devices
            sorted_indices_agent = list(
                sorted_indices_agent.reshape(
                    cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]
                )
            )

            # get final behavior and starting point that will be used til the
            # end of the given budget
            agent_indices = jax.device_put_sharded(
                sorted_indices_agent, devices
            )  # [D, N/D, K, M]

            start_positions = spread_over_devices(
                desired_start_positions
            )  # [D, N/D, K, M]

        latest_batch_best = episode_return.max((-1, -2))
        if used_budget == 0:
            best_episode_return = latest_batch_best
        else:
            # get latest best

            best_episode_return = jnp.concatenate(
                [best_episode_return[:, None], latest_batch_best[:, None]], axis=1
            ).max(-1)

        if environment.is_reward_negative():
            ret_sign = -1
        else:
            ret_sign = 1
        return_str = environment.get_reward_string()

        # get latest batch min, mean, max and std
        latest_batch_best_sp = episode_return.max(-1)
        latest_batch_min = latest_batch_best_sp.min(-1)
        latest_batch_mean = latest_batch_best_sp.mean(-1)
        latest_batch_std = latest_batch_best_sp.std(-1)

        # Make new metrics dictionary which will be all the returned statistics.
        metrics = {
            f"{return_str}_latest_batch": ret_sign * latest_batch_best.mean(),
            f"{return_str}": ret_sign * best_episode_return.mean(),
            f"{return_str}_latest_min": ret_sign * latest_batch_min.mean(),
            f"{return_str}_latest_mean": ret_sign * latest_batch_mean.mean(),
            f"{return_str}_latest_std": latest_batch_std.mean(),
        }

        # update the used budget
        used_budget += num_agents * num_starting_points

        log(metrics, used_budget, logger, "slowrl")

    return metrics
