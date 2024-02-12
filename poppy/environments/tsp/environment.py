import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.combinatorial.tsp.env import TSP
from jumanji.environments.combinatorial.tsp.types import State
from jumanji.environments.combinatorial.tsp.utils import compute_tour_length
from jumanji.types import Action, TimeStep, termination, transition
from typing import Any, Tuple

from poppy.environments.poppy_env import PoppyEnv
from poppy.environments.tsp.specs import ObservationSpec
from poppy.environments.tsp.types import Observation


class PoppyTSP(TSP, PoppyEnv):
    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
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

    def observation_spec(self) -> ObservationSpec:
        obs_spec = super(PoppyTSP, self).observation_spec()
        return ObservationSpec(
            obs_spec.problem_obs,
            obs_spec.start_position_obs,
            obs_spec.position_obs,
            obs_spec.action_mask,
            specs.DiscreteArray(1, dtype=jnp.int8, name="is_done")
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
            problem=state.problem,
            start_position=state.order[0],
            position=state.position,
            action_mask=state.visited_mask,
            is_done=state.num_visited == self.problem_size,
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
            return termination(
                reward=-compute_tour_length(state.problem, state.order),
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0), observation=self._state_to_observation(state)
            )

        return jax.lax.cond(
            state.num_visited - 1 >= self.problem_size,
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
