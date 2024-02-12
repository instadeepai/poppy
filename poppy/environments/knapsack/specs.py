import jax
from jumanji import specs
from jumanji.environments.combinatorial.knapsack.specs import ObservationSpec as JumanjiObservationSpec
from typing import Any

from poppy.environments.knapsack.types import Observation


class ObservationSpec(JumanjiObservationSpec):
    def __init__(
        self,
        problem_obs: specs.BoundedArray,
        first_item_obs: specs.DiscreteArray,
        last_item_obs: specs.DiscreteArray,
        invalid_mask: specs.BoundedArray,
        is_done_obs: specs.DiscreteArray,
    ):
        super(ObservationSpec, self).__init__(problem_obs, first_item_obs, last_item_obs, invalid_mask)
        self.is_done_obs = is_done_obs

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tproblem_obs={repr(self.problem_obs)},\n"
            f"\tfirst_item_obs={repr(self.first_item_obs)},\n"
            f"\tlast_item_obs={repr(self.last_item_obs)},\n"
            f"\tinvalid_mask={repr(self.invalid_mask)},\n"
            f"\tis_done_obs={repr(self.is_done_obs)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            problem=self.problem_obs.generate_value(),
            start_position=self.first_item_obs.generate_value(),
            position=self.last_item_obs.generate_value(),
            action_mask=self.invalid_mask.generate_value(),
            is_done=self.is_done_obs.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Knapsack Observation conforms to the spec.

        Args:
            value: a Knapsack Observation

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
                    self.first_item_obs,
                    self.last_item_obs,
                    self.invalid_mask,
                    self.is_done_obs,
                ),
                (
                    value.problem,
                    value.start_position,
                    value.position,
                    value.action_mask,
                    value.is_done
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
            "first_item_obs": self.first_item_obs,
            "last_item_obs": self.last_item_obs,
            "invalid_mask": self.invalid_mask,
            "is_done_obs": self.is_done_obs,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
