import jax
from jumanji import specs
from jumanji.environments.combinatorial.tsp.specs import ObservationSpec as JumanjiObservationSpec
from poppy.environments.tsp.types import Observation
from typing import Any


class ObservationSpec(JumanjiObservationSpec):
    def __init__(
        self,
        problem_obs: specs.BoundedArray,
        start_position_obs: specs.DiscreteArray,
        position_obs: specs.DiscreteArray,
        action_mask: specs.BoundedArray,
        is_done_obs: specs.DiscreteArray,
    ):
        super(ObservationSpec, self).__init__(
            problem_obs, start_position_obs, position_obs, action_mask
        )
        self.is_done_obs = is_done_obs

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tproblem_obs={repr(self.problem_obs)},\n"
            f"\tstart_position_obs={repr(self.start_position_obs)},\n"
            f"\tposition_obs={repr(self.position_obs)},\n"
            f"\taction_mask={repr(self.action_mask)},\n"
            f"\tis_done_obs={repr(self.is_done_obs)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            problem=self.problem_obs.generate_value(),
            start_position=self.start_position_obs.generate_value(),
            position=self.position_obs.generate_value(),
            action_mask=self.action_mask.generate_value(),
            is_done=self.is_done_obs.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a TSP Observation conforms to the spec.

        Args:
            value: a TSP Observation

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            *jax.tree_map(
                lambda spec, v: spec.validate(v),
                (
                    self.problem_obs,
                    self.start_position_obs,
                    self.position_obs,
                    self.action_mask,
                    self.is_done_obs,
                ),
                (
                    value.problem,
                    value.start_position,
                    value.position,
                    value.action_mask,
                    value.is_done,
                ),
            )
        )
        return observation

    def replace(self, **kwargs: Any) -> "ObservationSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """
        all_kwargs = {
            "problem_obs": self.problem_obs,
            "start_position_obs": self.start_position_obs,
            "position_obs": self.position_obs,
            "action_mask": self.action_mask,
            "is_done_obs": self.is_done_obs,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
