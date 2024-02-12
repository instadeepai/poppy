from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array
from jumanji import specs
from jumanji.environments.routing.tsp.env import TSP
from jumanji.environments.routing.tsp.types import State

from poppy.environments.tsp.utils import (
    compute_tour_length,
    generate_problem,
    get_coordinates_augmentations
)
from jumanji.types import TimeStep, termination, transition, restart

from poppy.environments.poppy_env import PoppyEnv
from poppy.environments.tsp.types import Observation


class PoppyTSP(TSP, PoppyEnv):
    def step(self, state: State, action: chex.Numeric) -> Tuple[State, TimeStep]:
        """
        Run one timestep of the environment's dynamics. Unlike the Jumanji environment
        it assumes that the action taken is legal, which should be if the action masking
        is properly done and the agent respects it.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next position to visit.

        Returns:
            state: the next state of the environment.
            timestep: the timestep to be observed.
        """
        state = self._update_state(state, action)
        timestep = self._state_to_timestep(state, True)
        return state, timestep

    def reset_from_state(
            self, problem: Array, start_position: jnp.int32
    ) -> Tuple[State, TimeStep]:
        """
        Resets the environment from a given problem instance and start position.
        Args:
            problem: jax array (float32) of shape (problem_size, 2)
                the coordinates of each city
            start_position: int32
                the identifier (index) of the first city
        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding to the first timestep returned by the
                environment.
        """
        state = State(
            coordinates=problem,
            position=jnp.array(-1, jnp.int32),
            visited_mask=jnp.zeros(self.num_cities, dtype=jnp.int8),
            trajectory=-1 * jnp.ones(self.num_cities, jnp.int32),
            num_visited=jnp.int32(0),
            key=jax.random.PRNGKey(0),
        )
        state = self._update_state(state, start_position)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.
        Returns:
            Spec for the `Observation` whose fields are:
            - coordinates: BoundedArray (float) of shape (num_cities,).
            - position: DiscreteArray (num_values = num_cities) of shape ().
            - trajectory: BoundedArray (int32) of shape (num_cities,).
            - action_mask: BoundedArray (bool) of shape (num_cities,).
        """
        problem = specs.BoundedArray(
            shape=(self.num_cities, 2),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="problem",
        )
        start_position = specs.DiscreteArray(
            self.num_cities, dtype=jnp.int32, name="start_position"
        )
        position = specs.DiscreteArray(
            self.num_cities, dtype=jnp.int32, name="position"
        )
        trajectory = specs.BoundedArray(
            shape=(self.num_cities,),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.num_cities - 1,
            name="trajectory",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_cities,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            problem=problem,
            start_position=start_position,
            position=position,
            trajectory=trajectory,
            action_mask=action_mask,
            is_done=specs.DiscreteArray(1, dtype=jnp.int8, name="is_done"),
        )

    def _state_to_observation(self, state: State) -> Observation:
        """
        Converts a state into an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """
        return Observation(
            problem=state.coordinates,
            start_position=state.trajectory[0],
            position=state.position,
            trajectory=state.trajectory,
            action_mask=state.visited_mask,
            is_done=(state.num_visited == self.num_cities).astype(int),
        )

    def _state_to_timestep(self, state: State, is_valid: bool) -> TimeStep:
        """
        Checks if the state is terminal and converts it into a timestep. The episode terminates if
        there is no legal action to take (i.e., all cities have been visited) or if the last
        action was not valid.

        Args:
            state: State object containing the dynamics of the environment.
            is_valid: Boolean indicating whether the last action was valid.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        def make_termination_timestep(state: State) -> TimeStep:
            reward = jax.lax.cond(
                state.visited_mask.sum() == self.num_cities,
                lambda _: -compute_tour_length(state.coordinates, state.trajectory),
                lambda _: jnp.array(-self.num_cities * jnp.sqrt(2), float),
                None,
            )
            return termination(
                reward=reward,
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0), observation=self._state_to_observation(state)
            )

        return jax.lax.cond(
            state.num_visited - 1 >= self.num_cities,
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

    def render(self, state: State) -> Any:
        raise NotImplementedError

    def get_problem_size(self) -> int:
        return self.num_cities

    def get_min_start(self) -> int:
        return 0

    def get_max_start(self) -> int:
        return self.num_cities - 1

    def get_episode_horizon(self) -> int:
        return self.num_cities

    @staticmethod
    def generate_problem(*args, **kwargs) -> chex.Array:
        return generate_problem(*args, **kwargs)

    @staticmethod
    def get_augmentations(*args, **kwargs) -> chex.Array:
        return get_coordinates_augmentations(*args, **kwargs)

    @staticmethod
    def make_observation(*args, **kwargs) -> Observation:
        return Observation(*args, **kwargs)

    @staticmethod
    def is_reward_negative() -> bool:
        return True

    @staticmethod
    def get_reward_string() -> str:
        return "tour_length"
