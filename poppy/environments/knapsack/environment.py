from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array
from jumanji import specs
from jumanji.environments.packing.knapsack.env import Knapsack
from poppy.environments.knapsack.utils import (
    compute_value_items,
    generate_problem,
)
from jumanji.types import TimeStep, restart, termination, transition

from poppy.environments.knapsack.types import Observation, State
from poppy.environments.poppy_env import PoppyEnv


class PoppyKnapsack(Knapsack, PoppyEnv):
    def reset_from_state(
        self, problem: Array, first_item: jnp.int32
    ) -> Tuple[State, TimeStep]:
        state = State(
            weights=problem[:, 0],
            values=problem[:, 1],
            packed_items=jnp.zeros(self.num_items, dtype=jnp.int8),
            remaining_budget=jnp.float32(self.total_budget),
            key=jax.random.PRNGKey(0),
            num_steps=jnp.int32(0),
        )
        state = self._update_state(state, first_item)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: chex.Numeric) -> Tuple[State, TimeStep]:
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
        fit_budget = state.remaining_budget >= state.weights[next_item]
        new_item = state.packed_items[next_item] == 0
        is_valid = fit_budget & new_item

        next_mask, next_budget = jax.lax.cond(
            pred=is_valid,
            true_fun=lambda _: (
                state.packed_items.at[next_item].set(1),
                state.remaining_budget - state.weights[next_item],
            ),
            false_fun=lambda _: (
                state.packed_items,
                state.remaining_budget,
            ),
            operand=[],
        )

        return State(
            weights=state.weights,
            values=state.values,
            packed_items=next_mask,
            remaining_budget=next_budget,
            key=state.key,
            num_steps=state.num_steps + 1,
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.
        Returns:
            Spec for each field in the Observation:
            - weights: BoundedArray (float) of shape (num_items,).
            - values: BoundedArray (float) of shape (num_items,).
            - packed_items: BoundedArray (bool) of shape (num_items,).
            - action_mask: BoundedArray (bool) of shape (num_items,).
        """
        problem = specs.BoundedArray(
            shape=(self.num_items, 2),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="problem",
        )
        start_position = specs.DiscreteArray(
            self.num_items, dtype=jnp.int32, name="start_position"
        )
        position = specs.DiscreteArray(
            self.num_items, dtype=jnp.int32, name="position"
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_items,),
            minimum=False,
            maximum=True,
            dtype=bool,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            problem=problem,
            start_position=start_position,
            position=position,
            action_mask=action_mask,
            is_done=specs.DiscreteArray(1, dtype=jnp.int32, name="is_done"),
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state to an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """
        problem = jnp.hstack([state.weights[:, None], state.values[:, None]])
        return Observation(
            problem=problem,
            start_position=state.packed_items[0],
            position=state.packed_items[state.num_steps - 1],
            action_mask=state.packed_items
            | (state.remaining_budget < state.weights),
            is_done=(
                jnp.min(jnp.where(state.packed_items == 0, state.weights, 1))
                > state.remaining_budget
            ).astype(int),
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
                reward=compute_value_items(state.values, state.packed_items),
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0), observation=self._state_to_observation(state)
            )

        return jax.lax.cond(
            state.num_steps - 1 >= self.num_items,
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

    def render(self, state: State) -> Any:
        raise NotImplementedError

    def get_problem_size(self) -> int:
        return self.num_items

    def get_min_start(self) -> int:
        return 0

    def get_max_start(self) -> int:
        return self.num_items - 1

    def get_episode_horizon(self) -> int:
        return self.num_items

    @staticmethod
    def generate_problem(*args, **kwargs) -> chex.Array:
        return generate_problem(*args, **kwargs)

    @staticmethod
    def make_observation(*args, **kwargs) -> Observation:
        return Observation(*args, **kwargs)

    @staticmethod
    def is_reward_negative() -> bool:
        return False

    @staticmethod
    def get_reward_string() -> str:
        return "value_items"
