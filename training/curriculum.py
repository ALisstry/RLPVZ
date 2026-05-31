from dataclasses import dataclass
from typing import Any, Protocol

from .specs import ScenarioSpec


CURRICULUM_NONE = "none"
CURRICULUM_CHOICES = (CURRICULUM_NONE,)


class CurriculumStrategy(Protocol):
    name: str

    def current_scenario(self) -> ScenarioSpec: ...

    def update(self, metrics: dict[str, Any]) -> ScenarioSpec: ...


@dataclass
class NoCurriculumStrategy:
    scenario: ScenarioSpec
    name: str = "none"

    def current_scenario(self) -> ScenarioSpec:
        return self.scenario

    def update(self, metrics: dict[str, Any]) -> ScenarioSpec:
        return self.scenario


def build_curriculum_strategy(args: Any, scenario: ScenarioSpec) -> CurriculumStrategy:
    curriculum = getattr(args, "curriculum", CURRICULUM_NONE)
    if curriculum == CURRICULUM_NONE:
        return NoCurriculumStrategy(scenario)
    raise NotImplementedError(f"Curriculum strategy is not implemented: {curriculum}")
