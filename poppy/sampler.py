from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import functools

import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from chex import Array
from jax import random
from jumanji.types import Action
from poppy.trainers.trainer_base import ActingState, Information, Observation, TrainingState, TrainerBase

from poppy.utils.utils import spread_over_devices


class Sampler:
    """Wrapper around the trainer designed to sample from the policy."""

    def __init__(
            self,
            trainer: TrainerBase,
            problems: Array,
            batch_size: int = 1,
    ):
        """
        Args:
        :param trainer: the trainer to sample from
        :param problems: the problem instances to solve
        :param batch_size: the number of problems to solve in parallel
        """

        self.environment = trainer.environment
        self.config = trainer.config
        config = self.config
        encoder = trainer.encoder
        decoder = trainer.decoder
        environment = self.environment
        num_best_pairs = (
            config.pop_size
        )  # number of best (agent, start point) pairs to sample from

        self.training_state = trainer.training_state
        self.validation_problems = None
        mask_from_obs = trainer.mask_from_observation

        available_devices = len(jax.local_devices())
        if config.num_devices < 0:
            config.num_devices = available_devices
            print(f"Using {available_devices} available device(s).")
        else:
            assert (
                    available_devices >= config.num_devices
            ), f"{config.num_devices} devices requested but only {available_devices} available."

        num_problems = config.num_validation_problems
        batch_size = batch_size or config.minibatch_validation
        # get the scores
        if config.num_devices > 1:
            self.problems = spread_over_devices(problems)
        else:
            self.problems = problems
        metrics, logging = trainer.run_validation_epoch(
            trainer.training_state, self.problems
        )
        self.scores = logging["score_matrix_with_start"]

        def policy_train(
                params: hk.Params,
                observation: Observation,
                embeddings: Array,
                key,
        ) -> Action:
            logits = decoder.apply(params, observation, embeddings)
            logits -= 1e6 * mask_from_obs(observation)  # mask visited locations
            action = rlax.softmax(temperature=1).sample(key, logits)
            logprob = rlax.softmax(temperature=1).logprob(sample=action, logits=logits)
            return action, logprob

        def rollout_sampling(
                embeddings: Array,
                params: hk.Params,
                problem: Array,
                start_positions: Array,
                acting_keys,
        ):
            policy = policy_train

            def run_epoch(problem, start_position, acting_key, embeddings):
                def take_step(acting_state):
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

            state, (tradj, info) = run_epoch(
                problem, start_positions, acting_keys, embeddings
            )
            return state, (tradj, info)

        def _run_sampling_on_best_start(
                training_state: TrainingState, problems, scores
        ):
            @functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))  # BATCH
            @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))  # POMO
            def rollout_best(
                    embeddings, problem, start_positions, acting_key, best_params
            ):
                # instead of 1 acting_keys, use (200 - population) for sampling
                acting_keys = random.split(
                    acting_key,
                    (200 - config.pop_size) * self.environment.get_problem_size() // num_best_pairs,
                )
                acting_state, (tradj, info) = jax.vmap(
                    rollout_sampling, in_axes=(None, None, None, None, 0)
                )(embeddings, best_params, problem, start_positions, acting_keys)
                return acting_state, (tradj, info)

            n_agents, batch_size, start_pos = scores.shape
            best_idx = jnp.einsum("ijk->jki", scores).reshape(batch_size, -1)

            best_idx = best_idx.argsort(1)[:, -num_best_pairs:]

            best_agent_idx = best_idx % n_agents
            best_start_positions = best_idx // n_agents
            best_start_positions = jnp.take_along_axis(
                trainer.make_all_start_positions(environment, batch_size),
                best_start_positions,
                axis=1,
            )
            acting_keys = random.split(
                training_state.key, batch_size * num_best_pairs
            ).reshape(batch_size, num_best_pairs, -1)
            params = training_state.params

            encoder_params, decoder_params = hk.data_structures.partition(
                lambda m, n, p: "shared_encoder" in m, params
            )
            embeddings = jax.vmap(encoder.apply, in_axes=(None, 0))(
                jax.lax.stop_gradient(encoder_params), problems
            )

            best_params = jax.tree_map(lambda w: w[best_agent_idx], decoder_params)

            state, (tradj, info) = rollout_best(
                embeddings, problems, best_start_positions, acting_keys, best_params
            )
            episode_rewards = tradj.reward.sum(
                -1
            )  # [batch_size, num_best_pairs, num_trajs]

            return episode_rewards.max(-1).max(-1)  # [batch_size]

        def run_sampling_on_best_start(
                training_state: TrainingState, problems: Array, scores: Array
        ):

            # use minibatch
            problems = problems.reshape(
                num_problems // batch_size,
                -1,
                problems.shape[1],
                problems.shape[-1],
            )
            # scores: [POP, num_problems, POMO_SIZE]
            scores = scores.reshape(
                scores.shape[0],
                scores.shape[1] // batch_size,
                -1,
                scores.shape[-1],
            )

            scores = jnp.swapaxes(scores, 0, 1)
            episode_rewards = jax.lax.map(
                lambda problem_score: _run_sampling_on_best_start(
                    training_state, problem_score[0], problem_score[1]
                ),
                (problems, scores),
            )

            return episode_rewards

        # Where the magic happens...
        if config.num_devices > 1:
            self.run_sampling_on_best_start = jax.pmap(
                run_sampling_on_best_start, axis_name="i"
            )
        else:
            self.run_sampling_on_best_start = jax.jit(run_sampling_on_best_start)

    def run_sampling(self):
        return self.run_sampling_on_best_start(self.training_state, self.problems, self.scores)
