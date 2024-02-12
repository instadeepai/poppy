from dataclasses import field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import chex
import omegaconf

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

import functools
import time

import acme
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jmp
import optax
import rlax
from chex import Array, PRNGKey
from jax import random
from jumanji.environments.packing.knapsack.types import State as StateKnapsack
from jumanji.environments.routing.cvrp.types import State as StateCVRP
from jumanji.environments.routing.tsp.types import State as StateTSP
from jumanji.types import TimeStep

import poppy.trainers.validation as validation
from poppy.environments.cvrp.types import Observation as ObservationCVRP
from poppy.environments.knapsack.types import Observation as ObservationKnapsack
from poppy.environments.poppy_env import PoppyEnv
from poppy.environments.tsp.types import Observation as ObservationTSP
from poppy.networks import DecoderBase, EncoderBase, Networks
from poppy.utils.checkpoint import (
    create_checkpoint_directory,
    load_checkpoint,
    save_checkpoint,
)
from poppy.utils.data import prepare_problem_batch
from poppy.utils.utils import (
    fetch_from_first_device,
    generate_zeros_from_spec,
    reduce_from_devices,
)

State = Union[StateTSP, StateKnapsack, StateCVRP]
Observation = Union[ObservationTSP, ObservationKnapsack, ObservationCVRP]


@dataclass
class TrainingState:  # type: ignore
    """Container for data used during the acting in the environment."""

    params: hk.Params
    optimizer_state: optax.OptState
    num_steps: jnp.int32
    key: PRNGKey
    extras: Optional[dict] = field(default_factory=dict)


@dataclass
class ActingState:  # type: ignore
    """Container for data used during the acting in the environment."""

    state: State
    timestep: TimeStep
    key: PRNGKey


@dataclass
class Information:  # type: ignore
    extras: Optional[dict] = field(default_factory=dict)
    metrics: Optional[dict] = field(default_factory=dict)
    logging: Optional[dict] = field(default_factory=dict)


def get_optimizer(cfg: omegaconf.DictConfig) -> optax.GradientTransformation:
    encoder_mask_fn = functools.partial(
        hk.data_structures.map, lambda m, n, p: "encoder" in m
    )
    decoder_mask_fn = functools.partial(
        hk.data_structures.map, lambda m, n, p: "encoder" not in m
    )

    optimizer = optax.chain(
        optax.masked(
            optax.adamw(
                learning_rate=cfg.encoder.lr,
                weight_decay=cfg.encoder.l2_regularization,
            ),
            encoder_mask_fn,
        ),
        optax.masked(
            optax.adamw(
                learning_rate=cfg.decoder.lr,
                weight_decay=cfg.decoder.l2_regularization,
            ),
            decoder_mask_fn,
        ),
    )
    optimizer = optax.MultiSteps(optimizer, cfg.num_gradient_accumulation_steps)

    return optimizer


def get_networks(cfg) -> Networks:
    def encoder_fn(problem: chex.Array):
        encoder = hydra.utils.instantiate(cfg.encoder, name="shared_encoder")
        return encoder(problem)

    def decoder_fn(observation: Observation, embeddings: Array):
        decoder = hydra.utils.instantiate(cfg.decoder, name="decoder")
        return decoder(observation, embeddings)

    return Networks(
        encoder_fn=hk.without_apply_rng(hk.transform(encoder_fn)),
        decoder_fn=hk.without_apply_rng(hk.transform(decoder_fn)),
    )


def init_training_state(
        cfg: omegaconf.DictConfig, networks: Networks, environment: PoppyEnv
) -> TrainingState:
    key = random.PRNGKey(cfg.seed)
    encoder_key, decoder_key, training_key = random.split(key, 3)

    (
        encoder_params,
        decoder_params,
        optimizer_state,
        keys,
        num_steps,
        extras,
    ) = load_checkpoint(cfg)

    environment_spec = acme.make_environment_spec(environment)
    _dummy_obs = environment.make_observation(
        *jax.tree_map(
            generate_zeros_from_spec,
            environment_spec.observations.generate_value(),
        )
    )

    if not encoder_params:
        encoder_params = networks.encoder_fn.init(encoder_key, _dummy_obs.problem)

    if not decoder_params:
        embeddings = networks.encoder_fn.apply(encoder_params, _dummy_obs.problem)
        decoder_params = jax.vmap(networks.decoder_fn.init, in_axes=(0, None, None))(
            random.split(decoder_key, cfg.pop_size), _dummy_obs, embeddings
        )

    if not keys:
        keys = list(random.split(training_key, cfg.num_devices))

    # Distribute parameters over devices as required.
    devices = jax.local_devices()
    encoder_params = jax.device_put_replicated(encoder_params, devices)
    if cfg.rollout.decoder_pmap_axis == "batch":
        # Decoding is parallelised over the batch --> every agent needs to be on every device.
        decoder_params = jax.device_put_replicated(decoder_params, devices)
    elif cfg.rollout.decoder_pmap_axis == "pop":
        # Decoding is parallelised over the population --> subset of agent needs to be on every device.
        assert (
                cfg.pop_size >= cfg.num_devices
        ), f"Population of size {cfg.pop_size} too small for distribution over {cfg.num_devices} devices."
        assert (
                cfg.pop_size % cfg.num_devices == 0
        ), f"Population of size {cfg.pop_size} isn't divisibile by number of devices ({cfg.num_devices})."

        def distribute_params(p):
            shp = p.shape
            p = list(p.reshape(cfg.num_devices, shp[0] // cfg.num_devices, *shp[1:]))

            return jax.device_put_sharded(p, devices)

        decoder_params = jax.tree_map(distribute_params, decoder_params)
    else:
        raise ValueError(
            f"config.rollout.decoder_pmap_axis of {cfg.rollout.decoder_pmap_axis} not recognised"
        )

    params = hk.data_structures.merge(encoder_params, decoder_params)
    if not optimizer_state:
        optimizer_state = get_optimizer(cfg.optimizer).init(
            fetch_from_first_device(params)
        )

    training_state = TrainingState(
        params=params,
        optimizer_state=jax.device_put_replicated(optimizer_state, devices),
        num_steps=jax.device_put_replicated(num_steps, devices),
        key=jax.device_put_sharded(keys, devices),
        extras=jax.device_put_replicated(extras, devices),
    )

    return training_state


def generate_trajectory(
        decoder_apply_fn,
        policy_temperature,
        environment,
        problem,
        embeddings,
        params,
        start_position,
        acting_key,
):
    """Decode a single agent, from a single starting position on a single problem.

    With decorators, the expected input dimensions are:
        problems: [N, problem_size, 2]
        embeddings: [N, problem_size, 128]
        params (decoder only): {key: [K, ...]}
        start_position: [N, K, M]
        acting_key: [N, K, M, 2]
    """

    def policy(
            observation: Observation,
            key,
    ) -> Array:
        logits = decoder_apply_fn(params, observation, embeddings)
        logits -= 1e30 * observation.action_mask
        if policy_temperature > 0:
            action = rlax.softmax(temperature=policy_temperature).sample(key, logits)
        else:
            action = rlax.greedy().sample(key, logits)
        logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)
        return action, logprob

    def take_step(acting_state):
        # TODO when the environment is done, a dummy step should be used to save computation time.
        #  Especially useful for knapsack environment where real number of steps << max number of steps
        #  theoretically possible.
        key, act_key = random.split(acting_state.key, 2)
        action, logprob = policy(acting_state.timestep.observation, act_key)
        state, timestep = environment.step(acting_state.state, action)
        info = Information(extras={"logprob": logprob}, metrics={}, logging={})
        acting_state = ActingState(state=state, timestep=timestep, key=key)
        return acting_state, (timestep, info)

    state, timestep = environment.reset_from_state(problem, start_position)

    acting_state = ActingState(state=state, timestep=timestep, key=acting_key)

    acting_state, (traj, info) = jax.lax.scan(
        lambda acting_state, _: take_step(acting_state),
        acting_state,
        xs=None,
        length=environment.get_episode_horizon(),
    )

    return acting_state, (traj, info)


def rollout(
        cfg: omegaconf.DictConfig,
        environment: PoppyEnv,
        params: chex.ArrayTree,
        networks: Networks,
        problems: jnp.ndarray,
        start_positions: jnp.ndarray,
        acting_keys: jnp.ndarray,
) -> Tuple[ActingState, Tuple[TimeStep, Information]]:
    """Rollout a batch of agents on a batch of problems and starting points.

    Args:
        cfg: The rollout config.
        environment: The environment to rollout.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.
        networks: The required networks.
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).

    Returns:
        # TODO
    """

    # Initialise the embeddings for each problem.
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0))  # over N problems
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))  # over K agents
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0))  # M starting pos.
    def generate_trajectory_fn(
            problem, embeddings, decoder_params, start_position, acting_key
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

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems, embeddings, decoder_params, start_positions, acting_keys
    )

    return acting_state, (traj, info)


def calculate_loss(traj, info, use_poppy='pomo') -> chex.Array:
    returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]
    logprob_traj = info.extras["logprob"].sum(-1)  # [N, K, M, t] --> [N, K, M]

    # Calculate advantages.
    if returns.shape[-1] > 1 and use_poppy != 'poppy':
        advantages = returns - returns.mean(-1, keepdims=True)
    else:
        advantages = returns

    if use_poppy == 'poppy_with_pomo_baseline':
        train_idxs = returns.argmax(axis=1, keepdims=True)
        advantages = jnp.take_along_axis(advantages, train_idxs, axis=1)
        logprob_traj = jnp.take_along_axis(logprob_traj, train_idxs, axis=1)

    elif use_poppy == 'poppy':
        sorted_idxs = jnp.argsort(returns, axis=1)  # [N, K, M]

        # Get the indices of the maximum and second maximum return
        max_idxs = sorted_idxs[:, -1:, :]  # [N, 1, M]
        second_max_idxs = sorted_idxs[:, -2:-1, :]  # [N, 1, M]

        # Get the maximum and second maximum returns
        max_returns = jnp.take_along_axis(returns, max_idxs, axis=1)  # [N, 1, M]
        second_max_returns = jnp.take_along_axis(returns, second_max_idxs, axis=1)  # [N, 1, M]

        # Calculate advantages as the difference between the maximum and second maximum returns
        advantages = max_returns - second_max_returns  # [N, 1, M]

        # Here, we still take the log probabilities along the max return path
        logprob_traj = jnp.take_along_axis(logprob_traj, max_idxs, axis=1)  # [N, 1, M]

    loss = -jnp.mean(advantages * logprob_traj)
    return loss


def get_policy(use_half=True, is_norm_layer=False):
    """Get a jmp.Policy.

    Args:
        use_half: Whether we are configured to use half (bf16) precision.
        is_norm_layer: Whether this policy should be that for a normalisation layer.

    Returns: A configured jmp.Policy.
    """

    half = jnp.bfloat16  # only support A100 GPU and TPU for now
    full = jnp.float32
    if use_half:
        if is_norm_layer:
            # Compute normalisation layers (batch/layer etc) in full precision to avoid instabilities.
            policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        else:
            policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)
    else:
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)
    return policy


def set_policy(modules: Union[List[hk.Module], hk.Module], use_half: bool = True):
    """Set the jmp.Policy of the passed modules.

    Args:
        modules: A list of (or single) haiku modules.
        use_half: Whether we are configured to use half (bf16) precision.

    Returns: None
    """
    if type(modules) is not list:
        modules = [modules]
    for mod in modules:
        policy = get_policy(use_half, is_norm_layer=False)
        hk.mixed_precision.set_policy(mod, policy)
    if use_half:
        # Ensure some layers are always in full precision.
        policy_norm = get_policy(use_half=True, is_norm_layer=True)
        hk.mixed_precision.set_policy(hk.BatchNorm, policy_norm)
        hk.mixed_precision.set_policy(hk.LayerNorm, policy_norm)


class Trainer:
    def __init__(
            self,
            cfg: omegaconf.DictConfig,
            logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.environment = hydra.utils.instantiate(cfg.environment)
        self.networks = get_networks(cfg.networks)
        create_checkpoint_directory(cfg, self.logger)
        self.training_state = init_training_state(cfg, self.networks, self.environment)

        self.cfg.validation.num_devices = self.cfg.num_devices

        if (
                self.cfg.validation.use_augmentations
                and "Knapsack" in cfg.environment._target_
        ):
            raise ValueError(
                "Knapsack's problem instances cannot be augmented, set "
                "'use_augmentations' in config.validate to False."
            )

        def sgd_step(training_state):
            def loss_and_output(params, problems, start_positions, acting_keys):
                state, (traj, info) = rollout(
                    self.cfg.rollout,
                    self.environment,
                    params,
                    self.networks,
                    problems,
                    start_positions,
                    acting_keys,
                )

                # Mask logprob's for steps where the environement was done.
                #  - traj.observation.is_done = [0,0,...,0,1,1,...] with the first 1 at the terminal step.
                #  - we want to mask everything *after* the last step, hence the roll & setting the
                #    first step to always (obviously) not be done.
                # TODO: should this be done inside of the rollout function by default?
                is_done = (
                    jnp.roll(traj.observation.is_done, 1, axis=-1).at[..., 0].set(0)
                )
                info.extras["logprob"] *= 1 - is_done

                loss = calculate_loss(
                    traj, info, use_poppy=self.cfg.loss_objective
                )

                # Log loss and returns.
                info.metrics["loss"] = loss

                episode_return = traj.reward.sum(-1)  # [N, K, M]
                if self.environment.is_reward_negative():
                    ret_sign = -1
                else:
                    ret_sign = 1
                return_str = self.environment.get_reward_string()

                info.metrics[f"{return_str}"] = (
                        ret_sign * episode_return.max((-1, -2)).mean()
                )
                if self.cfg.pop_size > 1:
                    info.metrics[f"{return_str}_rand_agent"] = (
                            ret_sign * episode_return.max(-1).mean()
                    )
                if self.cfg.num_starting_positions != 1:
                    info.metrics[f"{return_str}_rand_start"] = (
                            ret_sign * episode_return.max(-2).mean()
                    )
                if (self.cfg.pop_size > 1) and (self.cfg.num_starting_positions != 1):
                    info.metrics[f"{return_str}_rand_agent+start"] = (
                            ret_sign * episode_return.mean()
                    )

                return loss, (state, (traj, info))

            # Prepare batch of problems, start positions and acting keys.
            key, problem_key = random.split(training_state.key, 2)

            num_problems = self.cfg.batch_size // self.cfg.num_devices
            duplicate_problems_on_each_device = False

            if self.cfg.rollout.decoder_pmap_axis == "pop":
                num_agents = self.cfg.pop_size // self.cfg.num_devices
            else:
                num_agents = self.cfg.pop_size

            problem_key, start_key, act_key = random.split(problem_key, 3)
            if duplicate_problems_on_each_device:
                # Distribute the problem key from the first device to all devices.
                problem_key = jax.lax.all_gather(problem_key, "i", axis=0)[0]

            problems, start_positions, acting_keys = prepare_problem_batch(
                prng_key=problem_key,
                environment=self.environment,
                num_problems=num_problems,
                num_agents=num_agents,
                num_start_positions=self.cfg.num_starting_positions,
                duplicate_problems_on_each_device=duplicate_problems_on_each_device,
            )

            params = training_state.params
            optimizer_state = training_state.optimizer_state

            grads, (state, (traj, info)) = jax.grad(
                loss_and_output,
                has_aux=True,
            )(
                params,
                problems,
                start_positions,
                acting_keys,
            )

            if self.cfg.num_devices > 1:
                # Taking the mean across all devices to keep params in sync.
                grads = jax.lax.pmean(grads, axis_name="i")

            # TODO: mask optimizer updates for non-trained decoder heads.
            updates, optimizer_state = get_optimizer(self.cfg.optimizer).update(
                grads, optimizer_state, params=params
            )

            params = optax.apply_updates(params, updates)

            training_state = TrainingState(
                params=params,
                optimizer_state=optimizer_state,
                key=key,
                num_steps=training_state.num_steps + 1,
                extras=training_state.extras,
            )

            return training_state, info.metrics

        @functools.partial(jax.pmap, axis_name="i")
        def n_sgd_steps(training_state):
            training_state, metrics = jax.lax.scan(
                lambda state, xs: sgd_step(state),
                init=training_state,
                xs=None,
                length=self.cfg.num_jit_steps,
            )

            # Average metrics over all jit-ted steps.
            metrics = jax.tree_map(lambda x: x.mean(0), metrics)

            return training_state, metrics

        self.n_sgd_steps = n_sgd_steps

    def train(self):  # noqa: CCR001
        def get_n_steps():
            if self.cfg.num_devices > 1:
                n_steps = fetch_from_first_device(self.training_state.num_steps)
            else:
                n_steps = self.training_state.num_steps
            return n_steps

        def log(metrics, key=None):
            metrics["step"] = get_n_steps()
            if self.logger:
                if key:
                    metrics = {f"{key}/{k}": v for (k, v) in metrics.items()}
                self.logger.write(metrics)

        set_train_policy = lambda: set_policy(
            modules=[EncoderBase, DecoderBase, hk.MultiHeadAttention],
            use_half=self.cfg.use_half_precision,
        )
        set_validation_policy = lambda: set_policy(
            modules=[EncoderBase, DecoderBase, hk.MultiHeadAttention],
            use_half=self.cfg.validation.use_half_precision,
        )

        set_train_policy()

        while get_n_steps() <= self.cfg.num_steps:
            if get_n_steps() % self.cfg.validation_freq == 0:
                set_validation_policy()
                t = time.time()
                training_state = fetch_from_first_device(self.training_state)
                metrics = validation.validate(
                    self.cfg.validation, training_state.params
                )
                jax.tree_map(
                    lambda x: x.block_until_ready(), metrics
                )  # For accurate timings.
                metrics["total_time"] = time.time() - t
                if self.cfg.num_devices > 1:
                    metrics = reduce_from_devices(metrics, axis=0)
                log(metrics, "validate")
                set_train_policy()

                reward_str = self.environment.get_reward_string()
                if self.cfg.checkpointing.save_checkpoint:
                    training_state = fetch_from_first_device(
                        self.training_state
                    ).replace(key=self.training_state.key)
                    save_checkpoint(
                        self.cfg,
                        training_state,
                        self.logger,
                    )

                    if (
                            metrics[reward_str] > training_state.extras["best_reward"]
                            and self.cfg.checkpointing.keep_best_checkpoint
                    ):
                        save_checkpoint(
                            self.cfg,
                            training_state,
                            self.logger,
                            fname_prefix="best_",
                        )

                        extras = self.training_state.extras
                        extras.update(
                            {
                                "best_reward": jnp.ones_like(extras["best_reward"])
                                               * metrics[reward_str]
                            }
                        )

                        self.training_state = TrainingState(
                            params=self.training_state.params,
                            optimizer_state=self.training_state.optimizer_state,
                            num_steps=self.training_state.num_steps,
                            key=self.training_state.key,
                            extras=extras,
                        )

                    print(f"Saved checkpoint at step {get_n_steps()}")

            t = time.time()
            self.training_state, metrics = self.n_sgd_steps(self.training_state)
            jax.tree_map(
                lambda x: x.block_until_ready(), metrics
            )  # For accurate timings.

            if self.cfg.num_devices > 1:
                metrics = reduce_from_devices(metrics, axis=0)
            metrics["step_time"] = (time.time() - t) / self.cfg.num_jit_steps
            log(metrics, "train")
