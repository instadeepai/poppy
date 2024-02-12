import json
import operator
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import field
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Type, Union


if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

import functools

import acme
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from chex import Array, PRNGKey
from jax import random
from jumanji.environments.combinatorial.knapsack.types import State as StateKnapsack
from jumanji.types import Action, TimeStep

from poppy.environments.cvrp.types import Observation as ObservationCVRP
from poppy.environments.cvrp.types import State as StateCVRP
from poppy.environments.knapsack.types import Observation as ObservationKnapsack
from poppy.environments.poppy_env import PoppyEnv
from poppy.environments.tsp.types import Observation as ObservationTSP
from jumanji.environments.combinatorial.tsp.types import State as StateTSP
from poppy.nets import DecoderBase, EncoderBase
from poppy.utils.load_utils import robust_load
from poppy.utils.logger import TerminalLogger
from poppy.utils.metrics import (
    compute_cheap_metrics,
    compute_expensive_metrics,
    get_contribution_agent,
)
from poppy.utils.plot_utils import save_matrix_img
from poppy.utils.utils import (
    dataclass_to_dict,
    fetch_from_devices,
    fetch_from_first_device,
    make_log_name,
    reduce_from_devices,
    generate_zeros_from_spec,
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


@dataclass
class TrainingConfig:  # type: ignore

    learning_rate_encoder: float = 1e-4
    learning_rate_decoder: float = 1e-4
    batch_size: int = 64  # batch size *per device*, *per agent*
    minibatch_train: int = 64  # minibatch size *per device*
    pop_size: int = 1  # population size
    train_best: bool = True  # train only the best agent for each instance
    pomo_size: int = -1  # -1 --> use all positions <--> pomo_size=problem_size
    l2_regularization: float = 0.0  # weights of the l2 loss
    seed: int = 0
    validation_freq: int = 100
    num_validation_problems: int = 1000  # num validation problems *per device*
    minibatch_validation: int = (
        500  # minibatch size for validation problems *per device*
    )
    num_devices: int = -1  # -1 --> auto-detect
    save_checkpoint: bool = True  # save the model after training
    save_best: bool = True  # save best model only (early stopping)
    load_checkpoint: Optional[str] = None  # optional path to the model directory
    load_decoder: Optional[bool] = True  # load also the decoders or reset new ones
    load_optimizer: Optional[
        bool
    ] = True  # load also the optimizer state or reset new one
    compute_expensive_metrics: bool = (
        True  # compute expensive metrics during validation step
    )
    save_matrix_freq: int = (
        -1
    )  # save the perf matrix every save_matrix_freq validation steps
    use_augmentation_validation: bool = (
        False  # use the 8 POMO augmentations during validation step
    )


class TrainerBase(ABC):
    """Training loop designed for PoppyEnv environments. It both acts in an environment on a batch
    of states and learns from them. The loop compiles and vmap sequences of steps.
    """

    def __init__(
        self,
        environment: PoppyEnv,
        config: TrainingConfig,
        logger: Optional[TerminalLogger] = None,
    ):
        self.environment = environment
        self.encoder = self.init_encoder_fn()
        self.decoder = self.init_decoder_fn()
        self.config = config
        self.logger = logger

        self.training_state = None
        self.validation_problems = None
        reward_str = self.get_reward_name()
        reward_pos = self.is_reward_positive()
        mask_from_obs = self.mask_from_observation

        available_devices = len(jax.local_devices())
        if config.num_devices < 0:
            config.num_devices = available_devices
            print(f"Using {available_devices} available device(s).")
        else:
            assert (
                available_devices >= config.num_devices
            ), f"{config.num_devices} devices requested but only {available_devices} available."

        config.use_augmentation_validation = (
            self.use_augmentations() and config.use_augmentation_validation
        )

        if config.save_checkpoint or config.save_matrix_freq > 0:
            dir_name = os.path.join("checkpoints", make_log_name())
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            self.dir_name = dir_name
            self.dir_img = os.path.join(dir_name, "images")
            os.mkdir(self.dir_img)
            self.checkpoint_path = f"{dir_name}/training_state.pkl"
            with open(os.path.join(dir_name, "config"), "w") as f:
                json.dump(dataclass_to_dict(config), f, indent=4)
            self.agent_order = None
        else:
            self.checkpoint_path = ""

        def policy_train(
            params,
            observation: Observation,
            embeddings: Array,
            key,
        ) -> Action:
            logits = self.decoder.apply(params, observation, embeddings)
            logits -= 1e6 * mask_from_obs(observation)  # mask visited locations
            action = rlax.softmax(temperature=1).sample(key, logits)
            logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)
            return action, logprob

        def policy_validation(
            params,
            observation: Observation,
            embeddings: Array,
            key,
        ) -> Action:
            logits = self.decoder.apply(params, observation, embeddings)
            logits -= 1e6 * mask_from_obs(observation)  # mask visited locations
            action = rlax.greedy().sample(key, logits)
            logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)
            return action, logprob

        @functools.partial(jax.vmap, in_axes=(0, None, None, 0, 0, 0))
        def rollout(embeddings, params, policy, problem, start_positions, acting_keys):
            """Rollout a batch of M*N trajectories on N problems.

            Without the decorator, this function:
                - takes a single set of embeddings ([problem_size, 128])
                - shares a single set decoder_params and a policy across all trajectories.
                - takes a single problem instance ([problem_size, 2])
                - takes M start positions ([M]) and acting keys ([M,2]).

            With the vmap decorator:
                - takes a N sets of embeddings ([N, problem_size, 128])
                - shares a single set of params and a policy across all trajectories.
                - takes a N problem instance ([N, problem_size, 2])
                - takes M start positions ([N,M]) and acting keys ([N,M,2]).
            """

            def run_epoch(problem, start_position, acting_key, embeddings):
                def take_step(acting_state):
                    # TODO when the environment is done, a dummy step should be used to save computation time.
                    #  Especially useful for knapsack environment where real number of steps << max number of steps
                    #  theoretically possible.
                    key, act_key = random.split(acting_state.key, 2)
                    action, logprob = policy(
                        params, acting_state.timestep.observation, embeddings, act_key
                    )
                    state, timestep = self.environment.step(
                        acting_state.state, action
                    )
                    info = Information(
                        extras={"logprob": logprob}, metrics={}, logging={}
                    )
                    acting_state = ActingState(state=state, timestep=timestep, key=key)
                    return acting_state, (timestep, info)

                state, timestep = self.environment.reset_from_state(
                    problem, start_position
                )

                acting_state = ActingState(
                    state=state, timestep=timestep, key=acting_key
                )

                acting_state, (tradj, info) = jax.lax.scan(
                    lambda acting_state, _: take_step(acting_state),
                    acting_state,
                    xs=None,
                    length=self.environment.get_episode_horizon(),
                )

                return acting_state, (tradj, info)

            state, (tradj, info) = jax.vmap(run_epoch, in_axes=(None, 0, 0, None))(
                problem, start_positions, acting_keys, embeddings
            )
            return state, (tradj, info)

        self.rollout = rollout

        def rollout_population(params, policy, problem, start_positions, acting_keys):
            """Rollout a batch of K*M*N trajectories on N problems.

            With the vmap decorator:
                - shares a single set of shared_params and a policy across all trajectories.
                - K sets of parameters unique to the K agents {'param':[K,...]}.
                - takes a N problem instance ([N, problem_size, 2])
                - takes M start positions ([N,M]) and acting keys ([N,M,2]).
            """

            @functools.partial(jax.vmap, in_axes=(None, 0, None, None, None, None))
            def _rollout_population(
                embeddings, params, policy, problem, start_positions, acting_keys
            ):
                return rollout(
                    embeddings, params, policy, problem, start_positions, acting_keys
                )

            encoder_params, decoder_params = hk.data_structures.partition(
                lambda m, n, p: "shared_encoder" in m, params
            )
            embeddings = jax.vmap(self.encoder.apply, in_axes=(None, 0))(
                encoder_params, problem
            )
            # if the learning rate of the encoder is 0, no grad is computed.
            if config.learning_rate_encoder <= 0:
                embeddings = jax.lax.stop_gradient(embeddings)

            return _rollout_population(
                embeddings,
                decoder_params,
                policy,
                problem,
                start_positions,
                acting_keys,
            )

        self.rollout_population = rollout_population

        # optimizer with different learning rates for encoder and decoder
        encoder_mask_fn = functools.partial(
            hk.data_structures.map, lambda m, n, p: "shared_encoder" in m
        )
        decoder_mask_fn = functools.partial(
            hk.data_structures.map, lambda m, n, p: "shared_encoder" not in m
        )

        def get_optimizer():
            optimizer = optax.chain(
                optax.masked(
                    optax.adam(learning_rate=config.learning_rate_encoder),
                    encoder_mask_fn,
                ),
                optax.masked(
                    optax.adam(learning_rate=config.learning_rate_decoder),
                    decoder_mask_fn,
                ),
            )
            optimizer = optax.MultiSteps(
                optimizer, config.batch_size // config.minibatch_train
            )
            return optimizer

        self.get_optimizer = get_optimizer

        if config.pomo_size <= 0:
            make_start_positions = lambda key: self.make_all_start_positions(
                self.environment, config.minibatch_train
            )
            config.pomo_size = self.environment.get_problem_size()
        else:
            make_start_positions = lambda key: random.randint(
                key,
                (config.minibatch_train, config.pomo_size),
                minval=self.environment.get_min_start(),
                maxval=self.environment.get_max_start() + 1,
            )

        # Independent baselines
        if config.pomo_size > 1:
            calc_advantages = lambda returns: returns - returns.mean(
                -1, keepdims=True
            )
        else:
            calc_advantages = lambda returns: returns

        def get_mask_done(tradj):
            mask_not_done = ~tradj.observation.is_done
            mask_not_done = jnp.roll(mask_not_done, 1, axis=-1)
            mask_not_done = mask_not_done.at[..., 0].set(1)
            return mask_not_done

        def run_training_epoch(
            training_state: TrainingState,
        ) -> Tuple[TrainingState, dict]:
            def calc_loss(params, tradj, info):
                returns = tradj.reward.sum(-1)  # [POP_SIZE, BATCH_SIZE, POMO_SIZE]
                logprob_tradj = info.extras["logprob"].sum(
                    -1
                )  # [POP_SIZE, BATCH_SIZE, POMO_SIZE]

                adv = calc_advantages(returns)

                if config.train_best:
                    if self.has_symmetric_starting_points():
                        n_agents, batch_size, pomo_size = returns.shape
                        grad_weights = jnp.einsum("ijk->jki", returns).reshape(
                            batch_size, -1
                        )
                        grad_weights = (
                                grad_weights.argsort(1).argsort(1) >= n_agents * pomo_size - 1
                        )
                        grad_weights = grad_weights.reshape(batch_size, pomo_size, -1)
                        grad_weights = jnp.einsum("ijk->kij", grad_weights)
                    else:
                        grad_weights = returns.argsort(0).argsort(0) >= config.pop_size - 1
                else:
                    grad_weights = jnp.ones_like(returns)

                loss_rl = -(adv * logprob_tradj * grad_weights)
                loss_rl = loss_rl.mean()
                info.metrics["loss_rl"] = loss_rl

                if config.l2_regularization <= 0:
                    return loss_rl

                weights_mask_fn = lambda m, n, p: n == "w"
                weights, _ = hk.data_structures.partition(weights_mask_fn, params)
                weights_2 = jax.tree_map(jnp.square, weights)
                sums = jax.tree_map(jnp.sum, weights_2)
                l2_norm = jax.tree_util.tree_reduce(operator.add, sums)
                loss_l2 = config.l2_regularization * l2_norm
                info.metrics["loss_l2"] = loss_l2
                loss = loss_rl + loss_l2
                return loss

            def loss_and_output(params, problems, start_positions, acting_keys):
                state, (tradj, info) = rollout_population(
                    params, policy_train, problems, start_positions, acting_keys
                )
                mask_not_done = get_mask_done(tradj)
                info.extras["logprob"] *= mask_not_done

                loss = calc_loss(params, tradj, info)
                episode_reward = tradj.reward.sum(
                    -1
                )  # [POP_SIZE, BATCH_SIZE, POMO_SIZE]

                info.metrics[f"{reward_str}_mean"] = episode_reward.mean()
                info.metrics[f"{reward_str}_single_agent"] = episode_reward.max(
                    -1
                ).mean()
                info.metrics[f"{reward_str}"] = episode_reward.max(-1).max(0).mean()
                if not reward_pos:
                    info.metrics[f"{reward_str}_mean"] *= -1
                    info.metrics[f"{reward_str}_single_agent"] *= -1
                    info.metrics[f"{reward_str}"] *= -1

                return loss, (state, (tradj, info))

            batch_size = config.minibatch_train
            pomo_size = config.pomo_size
            base_key, problem_key, start_key, act_key = random.split(
                training_state.key, 4
            )

            problems = jax.vmap(self.generate_problem, in_axes=(0, None))(
                random.split(problem_key, batch_size), environment.get_problem_size()
            )  # [batch_size, problem_size, 2]
            start_positions = make_start_positions(start_key)  # [batch_size, pomo_size]
            acting_keys = random.split(act_key, batch_size * pomo_size).reshape(
                batch_size, pomo_size, -1
            )  # [batch_size, pomo_size, 2]

            params = training_state.params
            optimizer_state = training_state.optimizer_state

            grads, (state, (tradj, info)) = jax.grad(loss_and_output, has_aux=True,)(
                params,
                problems,
                start_positions,
                acting_keys,
            )

            if config.num_devices > 1:
                # Taking the mean across all devices to keep params in sync.
                grads = jax.lax.pmean(grads, axis_name="i")

            updates, optimizer_state = get_optimizer().update(grads, optimizer_state)

            params = optax.apply_updates(params, updates)

            training_state = TrainingState(
                params=params,
                optimizer_state=optimizer_state,
                key=base_key,
                num_steps=training_state.num_steps + 1,
            )

            return training_state, info.metrics

        def run_validation_epoch(training_state: TrainingState, problems: Array):
            batch_size = problems.shape[0]
            pomo_size = self.environment.get_problem_size()
            start_positions = self.make_all_start_positions(
                self.environment, config.minibatch_validation
            )
            acting_keys = random.split(
                training_state.key, batch_size * pomo_size
            ).reshape(
                batch_size // config.minibatch_validation,
                config.minibatch_validation,
                pomo_size,
                -1,
            )

            problems = problems.reshape(
                batch_size // config.minibatch_validation,
                -1,
                problems.shape[1],
                problems.shape[-1],
            )

            if config.use_augmentation_validation:
                problems_aug = jax.vmap(jax.vmap(self.get_augmentations))(
                    problems
                )  # [mini_batch_idx, mini_batch_size, 8, problem_size, 2]

                state, (tradj, info) = jax.lax.map(
                    lambda problem_acting_key: jax.vmap(
                        rollout_population, in_axes=(None, None, 1, None, None)
                    )(
                        jax.lax.stop_gradient(training_state.params),
                        policy_validation,
                        problem_acting_key[0],
                        start_positions,
                        problem_acting_key[1],
                    ),
                    (problems_aug, acting_keys),
                )

                episode_reward_aug = tradj.reward.sum(
                    -1
                )  # [MINIBATCH_IDX, AUGMENTATION, POP_SIZE, MINIBATCH, POMO_SIZE]
                episode_reward_aug = episode_reward_aug.max(
                    1
                )  # [MINIBATCH_IDX, POP_SIZE, MINIBATCH, POMO_SIZE]
                episode_reward_aug = jnp.swapaxes(episode_reward_aug, 1, 2)
                episode_reward_aug = episode_reward_aug.reshape(
                    -1, *episode_reward_aug.shape[2:]
                )
                # [BATCH_SIZE, POP_SIZE, POMO_SIZE]
                episode_reward_aug = episode_reward_aug.swapaxes(
                    0, 1
                )  # [POP_SIZE, BATCH_SIZE, POMO_SIZE]

                info.metrics["reward_aug"] = episode_reward_aug.max(-1).max(0).mean()
                info.metrics[f"{reward_str}_aug"] = info.metrics["reward_aug"]
                info.metrics[f"{reward_str}_aug_per_agent"] = (
                    episode_reward_aug.max(-1).mean(0).mean()
                )
                if not reward_pos:
                    info.metrics[f"{reward_str}_aug"] *= -1
                    info.metrics[f"{reward_str}_aug_per_agent"] *= -1

                episode_reward = tradj.reward.sum(-1)[
                    :, 0, ...
                ]  # [MINIBATCH_IDX, POP_SIZE, MINIBATCH, POMO_SIZE]
            else:
                state, (tradj, info) = jax.lax.map(
                    lambda problem_acting_key: rollout_population(
                        jax.lax.stop_gradient(training_state.params),
                        policy_validation,
                        problem_acting_key[0],
                        start_positions,
                        problem_acting_key[1],
                    ),
                    (problems, acting_keys),
                )

                episode_reward = tradj.reward.sum(
                    -1
                )  # [MINIBATCH_IDX, POP_SIZE, MINIBATCH, POMO_SIZE]

            episode_reward = jnp.swapaxes(episode_reward, 1, 2)
            episode_reward = episode_reward.reshape(
                -1, episode_reward.shape[-2], episode_reward.shape[-1]
            )  # [BATCH_SIZE, POP_SIZE, POMO_SIZE]
            episode_reward = episode_reward.swapaxes(
                0, 1
            )  # [POP_SIZE, BATCH_SIZE, POMO_SIZE]

            n_agents = episode_reward.shape[0]
            info.metrics[f"{reward_str}_mean"] = episode_reward.mean()
            info.metrics[f"{reward_str}_single_agent"] = episode_reward.max(-1).mean()
            info.metrics[f"{reward_str}"] = episode_reward.max(-1).max(0).mean()
            info.metrics["reward"] = info.metrics[f"{reward_str}"]
            info.metrics[f"{reward_str}_std_between_agents_single_instance"] = (
                episode_reward.max(-1).std(0).mean()
            )
            info.metrics[f"{reward_str}_std_between_agents_mean"] = (
                episode_reward.max(-1).mean(-1).std()
            )
            info.logging["score_matrix"] = episode_reward.max(-1)
            info.logging["score_matrix_with_start"] = episode_reward
            info.logging["contribution_agents"] = get_contribution_agent(
                episode_reward.max(-1)
            )

            for i in range(n_agents):
                info.metrics[f"contribution_agent_{i}"] = info.logging[
                    "contribution_agents"
                ][i]

            if not reward_pos:
                info.metrics[f"{reward_str}_mean"] *= -1
                info.metrics[f"{reward_str}_single_agent"] *= -1
                info.metrics[f"{reward_str}"] *= -1

            if config.compute_expensive_metrics and n_agents >= 4:
                expensive_metrics = compute_expensive_metrics(episode_reward)
                info.metrics.update(expensive_metrics)

            if n_agents >= 4:
                cheap_metrics = compute_cheap_metrics(
                    episode_reward, training_state.key
                )
                info.metrics.update(cheap_metrics)

            return info.metrics, info.logging

        # Where the magic happens...
        if config.num_devices > 1:
            self.run_training_epoch = jax.pmap(run_training_epoch, axis_name="i")
            self.run_validation_epoch = jax.pmap(run_validation_epoch, axis_name="i")
        else:
            self.run_training_epoch = jax.jit(run_training_epoch)
            self.run_validation_epoch = jax.jit(run_validation_epoch)

    @staticmethod
    @abstractmethod
    def get_reward_name() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def is_reward_positive() -> bool:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_observation_type() -> Type[Observation]:
        raise NotImplementedError()

    @staticmethod
    def mask_from_observation(observation: Observation) -> Array:
        return observation.action_mask

    @abstractmethod
    def has_symmetric_starting_points(self) -> bool:
        raise NotImplementedError()

    def init_encoder_fn(self) -> hk.Transformed:
        def encoder_fn(problem: Array):
            encoder = self.init_encoder(num_layers=6, name="shared_encoder")
            return encoder(problem)

        return hk.without_apply_rng(hk.transform(encoder_fn))

    @abstractmethod
    def init_encoder(self, num_layers, name) -> EncoderBase:
        pass

    def init_decoder_fn(self) -> hk.Transformed:
        def decoder_fn(observation: Observation, embeddings: Array):
            decoder = self.init_decoder(name="decoder")
            return decoder(observation, embeddings)

        return hk.without_apply_rng(hk.transform(decoder_fn))

    @abstractmethod
    def init_decoder(self, name) -> DecoderBase:
        pass

    @staticmethod
    @abstractmethod
    def generate_problem(key: PRNGKey, problem_size: jnp.int32) -> Array:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def use_augmentations() -> bool:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_augmentations(problem: Array) -> Array:
        raise NotImplementedError()

    def make_all_start_positions(
        self, environment: PoppyEnv, minibatch_size: jnp.int32
    ) -> Callable[[PRNGKey], Array]:
        return jnp.arange(environment.get_min_start(), environment.get_max_start() + 1)[
            None
        ].repeat(minibatch_size, axis=0)

    def init(self):
        if self.training_state:
            # Training state has already been initialised - handle this somehow...
            pass

        key = random.PRNGKey(self.config.seed)
        encoder_key, decoder_key, training_key, validation_key = random.split(key, 4)

        saved_keys = None
        if self.config.load_checkpoint:
            with open(
                os.path.join(self.config.load_checkpoint, "training_state.pkl"), "rb"
            ) as f:
                saved_state = robust_load(f)
                if self.config.load_decoder:  # If loading the decoder, we also load the random keys
                    keys_shape = saved_state.key.shape
                    if (
                        (len(keys_shape) == 2 and keys_shape[0] == self.config.num_devices) or
                        (len(keys_shape) == 1 and self.config.num_devices == 1)
                    ):
                        # If we use the same number of devices
                        saved_keys = saved_state.key
                saved_encoder, saved_decoder = hk.data_structures.partition(
                    lambda m, n, p: "shared_encoder" in m, saved_state.params
                )
        environment_spec = acme.make_environment_spec(self.environment)
        _dummy_obs = self.get_observation_type()(
            *jax.tree_map(
                generate_zeros_from_spec,
                environment_spec.observations.generate_value(),
            )
        )
        if self.config.load_checkpoint:
            encoder_params = saved_encoder
        else:
            encoder_params = self.encoder.init(encoder_key, _dummy_obs.problem)

        if self.config.load_checkpoint and self.config.load_decoder:
            decoder_params = jax.tree_map(
                lambda x: x[: self.config.pop_size], saved_decoder
            )
        else:
            embeddings = self.encoder.apply(encoder_params, _dummy_obs.problem)
            decoder_params = jax.vmap(self.decoder.init, in_axes=(0, None, None))(
                random.split(decoder_key, self.config.pop_size), _dummy_obs, embeddings
            )

        params = hk.data_structures.merge(encoder_params, decoder_params)

        if self.config.load_checkpoint and self.config.load_decoder:
            num_steps = saved_state.num_steps
        else:
            num_steps = 0

        if self.config.load_checkpoint and self.config.load_optimizer:
            optimizer_state = saved_state.optimizer_state
        else:
            optimizer_state = self.get_optimizer().init(params)

        def prepare_state(params, training_key, optimizer_state, num_steps):
            training_state = TrainingState(
                params=params,
                optimizer_state=optimizer_state,
                num_steps=num_steps,
                key=training_key,
            )
            return training_state

        def prepare_validation_problems(validation_key):
            validation_keys = random.split(
                validation_key, self.config.num_validation_problems
            )
            return jax.vmap(
                lambda val_key: self.generate_problem(
                    val_key, self.environment.get_problem_size()
                )
            )(validation_keys)

        if self.config.num_devices > 1:
            validation_key = random.split(validation_key, self.config.num_devices)
            self.validation_problems = jax.pmap(
                prepare_validation_problems,
                in_axes=0,
            )(validation_key)
        else:
            self.validation_problems = prepare_validation_problems(validation_key)

        if self.config.num_devices > 1:
            training_key = random.split(training_key, self.config.num_devices)
            self.training_state = jax.pmap(
                prepare_state, in_axes=(None, 0, None, None)
            )(params, training_key, optimizer_state, num_steps)

        else:
            self.training_state = prepare_state(
                params, training_key, optimizer_state, num_steps
            )

        # fancy: use the same random keys as in the checkpoint state
        if saved_keys is not None:
            self.training_state.key = saved_keys

    def train(self, num_steps=100):  # noqa: CCR001
        def get_n_steps():
            if self.config.num_devices > 1:
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

        if not self.training_state:
            self.init()

        if self.config.save_checkpoint:
            best_reward = -1e6

        performance_str = (
            "reward_aug" if self.config.use_augmentation_validation else "reward"
        )

        while get_n_steps() <= num_steps:

            if get_n_steps() % self.config.validation_freq == 0:
                t = time.time()
                n_validation_steps = get_n_steps() // self.config.validation_freq
                metrics, logging = self.run_validation_epoch(
                    self.training_state, self.validation_problems
                )
                jax.tree_map(
                    lambda x: x.block_until_ready(), metrics
                )  # For accurate timings.
                metrics["total_time"] = time.time() - t
                if self.config.num_devices > 1:
                    metrics = reduce_from_devices(metrics, axis=0)
                    logging = fetch_from_devices(logging, as_numpy=True)
                    logging["score_matrix"] = np.concatenate(
                        logging["score_matrix"], axis=-1
                    )
                    logging["contribution_agents"] = np.sum(
                        logging["contribution_agents"], axis=0
                    )

                log(metrics, "validate")
                if self.config.save_checkpoint:
                    if (
                        metrics[performance_str] > best_reward
                        or not self.config.save_best
                    ):
                        with open(self.checkpoint_path, "wb") as f:
                            if self.config.num_devices > 1:
                                # Just save a checkpoint for the first device (avoid useless copies) but keep the
                                # random key in case a new execution starts from the set number of devices.
                                first_device_training_state = fetch_from_first_device(self.training_state)
                                first_device_training_state = first_device_training_state.replace(
                                    key=self.training_state.key
                                )
                                pickle.dump(first_device_training_state, f)
                            else:
                                pickle.dump(self.training_state, f)
                        with open(os.path.join(self.dir_name, "performance"), "w") as f:
                            performance_dict = {
                                "reward": float(metrics[performance_str]),
                                "step": int(get_n_steps()),
                            }
                            json.dump(performance_dict, f)
                        best_reward = metrics[performance_str]
                        print(f"Saved checkpoint at step {get_n_steps()}")

                if (
                    self.config.save_matrix_freq > 0
                    and n_validation_steps % self.config.save_matrix_freq == 0
                ):
                    img_name = os.path.join(
                        self.dir_img,
                        f"matrix_{n_validation_steps // self.config.save_matrix_freq}.pdf",
                    )
                    self.agent_order = save_matrix_img(
                        order_agent=self.agent_order,
                        score_matrix=logging["score_matrix"],
                        filename=img_name,
                        agent_score=logging["contribution_agents"],
                    )

            t = time.time()
            self.training_state, metrics = self.run_training_epoch(self.training_state)
            jax.tree_map(
                lambda x: x.block_until_ready(), metrics
            )  # For accurate timings.
            if self.config.num_devices > 1:
                metrics = reduce_from_devices(metrics, axis=0)
            metrics["step_time"] = time.time() - t
            log(metrics, "train")
