from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from jax import random
from jumanji import specs
from jumanji.env import Environment
from poppy.environments.tsp.utils import (
    compute_tour_length,
    generate_start_position,
)
from jumanji.types import TimeStep, restart, termination, transition

from jumanji.environments.routing.cvrp.types import State
from poppy.environments.cvrp.types import Observation
from poppy.environments.cvrp.utils import (
    DEPOT_IDX,
    MIN_NORM_FACTOR,
    generate_problem,
    get_augmentations,
)
from poppy.environments.poppy_env import PoppyEnv


class PoppyCVRP(Environment[State], PoppyEnv):
    """
    Capacitated Vehicle Routing Problem (CVRP) environment as described in [1].
    - observation: Observation
        - problem: jax array (float32) of shape (num_nodes + 1, 3)
            the coordinates of each node and the depot, and the associated cost (0.0 for the depot)
        - position: jax array (float32)
            the index of the last visited node
        - capacity: jax array (float32)
            the current capacity of the vehicle
        - invalid_mask: jax array (int8) of shape (num_nodes + 1,)
            binary mask (0/1 <--> visitable/not visitable)

    - reward: jax array (float32)
        the sum of the distances between consecutive nodes at the end of the episode (the reward is 0 if a previously
        selected non-dept node is selected again, or the depot is selected twice in a row)

    - state: State
        - problem: jax array (float32) of shape (num_nodes + 1, 3)
            the coordinates of each node and the depot, and the associated cost (0.0 for the depot)
        - position: jax array (float32)
            the index of the last visited node
        - capacity: jax array (float32)
            the current capacity of the vehicle
        - visited_mask: jax array (int8) of shape (num_nodes,)
            binary mask (0/1 <--> not visited/visited)
        - order: jax array (int32) of shape (2 * num_nodes,)
            the identifiers of the nodes that have been visited (-1 means that no node has been visited yet at that
            time in the sequence)
        - num_visits: int32
            number of actions that have been taken (i.e., unique visits)

    [1] Kwon Y., Choo J., Kim B., Yoon I., Min S., Gwon Y. (2020). "POMO: Policy Optimization with Multiple Optima for
        Reinforcement Learning".
    """

    def __init__(
        self,
        num_nodes: int = 20,
        norm_factor: int = 30,
    ):
        super(PoppyCVRP, self).__init__()
        assert norm_factor >= MIN_NORM_FACTOR, (
            f"The cost associated to each node must be lower than 1.0, hence the normalization factor must be "
            f">= {MIN_NORM_FACTOR}."
        )
        self.num_nodes = num_nodes
        self.norm_factor = norm_factor
        self.max_capacity = norm_factor

    def __repr__(self):
        return f"CVRP environment with {self.num_nodes} nodes and normalization factor {self.norm_factor}."

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep]:
        """
        Resets the environment.

        Args:
            key: used to randomly generate the problem and the start node.

        Returns:
             state: State object corresponding to the new state of the environment.
             timestep: TimeStep object corresponding to the first timestep returned by the environment.
             extra: Not used.
        """
        problem_key, start_key = random.split(key)
        problem = generate_problem(problem_key, self.num_nodes)
        start_node = generate_start_position(start_key, self.num_nodes)
        return self.reset_from_state(problem, start_node)

    def reset_from_state(
        self, problem: Array, start_node: jnp.int32
    ) -> Tuple[State, TimeStep]:
        """
        Resets the environment from a given problem instance and start node.

        Args:
            problem: jax array (float32) of shape (num_nodes + 1, 3)
                the coordinates of each node and the associated cost
            start_node: jax array (int32)
                the index of the first node visited after the depot
        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding to the first timestep returned by the environment.
            extra: Not used.
        """
        state = State(
            coordinates=problem[:, :2],
            demands=problem[:, -1],
            position=jnp.int32(DEPOT_IDX),
            capacity=self.max_capacity,
            visited_mask=jnp.zeros(self.num_nodes + 1, dtype=jnp.int8).at[DEPOT_IDX].set(1),
            trajectory=jnp.zeros(self.get_episode_horizon(), jnp.int32),
            num_total_visits=jnp.int32(1),
            key=jax.random.PRNGKey(0),
        )
        state = self._update_state(state, start_node)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: chex.Numeric) -> Tuple[State, TimeStep]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next node to visit.

        Returns:
            state, timestep, extra: Tuple[State, TimeStep, Extra] containing the next state of the environment, as well
            as the timestep to be observed.
        """
        state = self._update_state(state, action)
        timestep = self._state_to_timestep(state)
        return state, timestep

    def get_problem_size(self):
        return self.num_nodes

    def get_episode_horizon(self):
        return 2 * self.num_nodes

    def get_min_start(self) -> int:
        return 1

    def get_max_start(self) -> int:
        return self.num_nodes

    def action_spec(self) -> specs.Array:
        """
        Returns the action spec.

        Returns:
            action_spec: a `dm_env.specs.Array` spec.
        """
        return specs.DiscreteArray(self.num_nodes + 1, name="action")

    def _update_state(self, state: State, next_node: jnp.int32) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            next_node: int, index of the next node to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """
        is_not_depot = jnp.int32(next_node != DEPOT_IDX)

        next_node = jax.lax.cond(
            pred=state.visited_mask.sum() == self.num_nodes + 1,
            true_fun=lambda _: DEPOT_IDX,  # stay in the depot if we have visited all nodes
            false_fun=lambda _: next_node,
            operand=[],
        )

        return State(
            coordinates=state.coordinates,
            demands=state.demands,
            position=next_node,
            capacity=is_not_depot
            * (state.capacity - jnp.int32(state.demands[next_node]))
            + (1 - is_not_depot) * self.max_capacity,
            visited_mask=state.visited_mask.at[next_node].set(1),
            trajectory=state.trajectory.at[state.num_total_visits].set(next_node),
            num_total_visits=state.num_total_visits + 1,
            key=state.key,
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.
        Returns:
            Spec for the `Observation` whose fields are:
            - coordinates: BoundedArray (float) of shape (num_nodes + 1, 2).
            - demands: BoundedArray (float) of shape (num_nodes + 1,).
            - unvisited_nodes: BoundedArray (bool) of shape (num_nodes + 1,).
            - position: DiscreteArray (num_values = num_nodes + 1) of shape ().
            - trajectory: BoundedArray (int32) of shape (2 * num_nodes,).
            - capacity: BoundedArray (float) of shape ().
            - action_mask: BoundedArray (bool) of shape (num_nodes + 1,).
        """
        problem = specs.BoundedArray(
            shape=(self.num_nodes + 1, 3),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="problem",
        )
        position = specs.DiscreteArray(
            self.num_nodes + 1, dtype=jnp.int32, name="position"
        )
        capacity = specs.BoundedArray(
            shape=(), minimum=0.0, maximum=1.0, dtype=float, name="capacity"
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_nodes + 1,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            problem=problem,
            position=position,
            capacity=capacity,
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
        # A node is invalid if it has already been visited or the vehicle does not have enough capacity to cover its
        # demand.
        invalid_mask = state.visited_mask | (state.capacity < state.demands)

        # The depot is valid (0) if we are not at it, else it is invalid (1).
        invalid_mask = invalid_mask.at[DEPOT_IDX].set(jnp.int32(state.position == DEPOT_IDX))

        demands = state.demands.at[...].set(jnp.float32(state.demands / self.norm_factor))
        problem = jnp.hstack([state.coordinates, demands[:, None]])

        return Observation(
            problem=problem,
            position=state.position,
            capacity=state.capacity,
            action_mask=invalid_mask,
            is_done=(state.visited_mask.sum() == self.num_nodes + 1).astype(int),
        )

    def _state_to_timestep(self, state: State) -> TimeStep:
        """
        Checks if the state is terminal and converts it into a timestep.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        def make_termination_timestep(state: State) -> TimeStep:
            reward = jax.lax.cond(
                state.visited_mask.sum() >= self.num_nodes,
                lambda _: -compute_tour_length(state.coordinates, state.trajectory),
                lambda _: jnp.array(-self.num_nodes * 2 * jnp.sqrt(2), float),
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

        # We compare state.num_total_visits - 2 against 2 * self.num_nodes because:
        #  (i) the first two visits are part of the initialization (hence -2)
        #  (ii) there will be at most 2 * self.num_nodes visits (worst-case in which the vehicle
        #       goes back to the depot after each visit)
        return jax.lax.cond(
            (state.num_total_visits - 2) >= self.get_episode_horizon(),
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

    def render(self, state: State) -> Any:
        pass

    @staticmethod
    def generate_problem(*args, **kwargs) -> chex.Array:
        return generate_problem(*args, **kwargs)

    @staticmethod
    def get_augmentations(*args, **kwargs) -> chex.Array:
        return get_augmentations(*args, **kwargs)

    @staticmethod
    def make_observation(*args, **kwargs) -> Observation:
        return Observation(*args, **kwargs)

    @staticmethod
    def is_reward_negative() -> bool:
        return True

    @staticmethod
    def get_reward_string() -> str:
        return "tour_length"
