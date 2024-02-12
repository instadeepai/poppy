from chex import Array
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.combinatorial.knapsack.env import Knapsack
from jumanji.environments.combinatorial.knapsack.types import State
from jumanji.environments.combinatorial.knapsack.utils import compute_value_items
from jumanji.types import Action, TimeStep, restart, termination, transition
from typing import Any, Tuple

from poppy.environments.poppy_env import PoppyEnv
from poppy.environments.knapsack.specs import ObservationSpec
from poppy.environments.knapsack.types import Observation


class PoppyKnapsack(Knapsack, PoppyEnv):
    def reset_from_state(self, problem: Array, first_item: jnp.int32) -> Tuple[State, TimeStep]:
        state = State(
            problem=problem,
            last_item=-1,
            first_item=first_item,
            used_mask=jnp.zeros(self.problem_size, dtype=jnp.int8),
            num_steps=jnp.int32(0),
            remaining_budget=jnp.float32(self.total_budget),
        )
        state = self._update_state(state, first_item)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics. The difference with the Jumanji
        environment is that here we always increment the number of steps after performing
        each action, regardless of whether the action was valid or not (like in TSP, we
        assume the agent is using the action mask correctly and thus not taking any invalid
        actions).

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of next item to take.

        Returns:
            state: next state of the environment.
            timestep: the timestep to be observed.
        """
        state = self._update_state(state, action)
        timestep = self._state_to_timestep(state, True)
        return state, timestep

    def _update_state(self, state: State, next_item: int) -> State:
        fit_budget = state.remaining_budget >= state.problem[next_item, 0]
        new_item = state.used_mask[next_item] == 0
        is_valid = fit_budget & new_item

        next_item, next_mask, next_budget = jax.lax.cond(
            pred=is_valid,
            true_fun=lambda _: (
                next_item,
                state.used_mask.at[next_item].set(1),
                state.remaining_budget - state.problem[next_item, 0],
            ),
            false_fun=lambda _: (
                state.last_item,
                state.used_mask,
                state.remaining_budget,
            ),
            operand=[],
        )

        return State(
            problem=state.problem,
            last_item=next_item,
            first_item=state.first_item,
            used_mask=next_mask,
            num_steps=state.num_steps + 1,
            remaining_budget=next_budget,
        )

    def observation_spec(self) -> ObservationSpec:
        obs_spec = super(PoppyKnapsack, self).observation_spec()
        return ObservationSpec(
            obs_spec.problem_obs,
            obs_spec.first_item_obs,
            obs_spec.last_item_obs,
            obs_spec.invalid_mask,
            specs.DiscreteArray(1, dtype=jnp.int8, name="is_done")
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state to an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """
        return Observation(
            problem=state.problem,
            start_position=state.first_item,
            position=state.last_item,
            action_mask=state.used_mask | (state.remaining_budget < state.problem[:, 0]),
            is_done=jnp.min(jnp.where(state.used_mask == 0, state.problem[:, 0], 1)) > state.remaining_budget,
        )

    def _state_to_timestep(self, state: State, is_valid: bool) -> TimeStep:
        """Checks if the state is terminal and converts it to a timestep.

        Args:
            state: State object containing the dynamics of the environment.
            is_valid: Boolean indicating whether the last action was valid.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
                The episode terminates if there is no legal item to take or if
                the last action was invalid.
        """
        def make_termination_timestep(state: State) -> TimeStep:
            return termination(
                reward=compute_value_items(state.problem, state.used_mask),
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0),
                observation=self._state_to_observation(state)
            )

        return jax.lax.cond(
            state.num_steps - 1 >= self.problem_size,
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

    def render(self, state: State) -> Any:
        raise NotImplementedError

    def get_problem_size(self) -> int:
        return self.problem_size

    def get_min_start(self) -> int:
        return 0

    def get_max_start(self) -> int:
        return self.problem_size - 1

    def get_episode_horizon(self) -> int:
        return self.problem_size
