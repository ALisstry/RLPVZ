import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from utils.train_utils import load_training_config

from .specs import ScenarioSpec, build_scenario_specs


CURRICULUM_NONE = "none"
CURRICULUM_STAGE_GATE = "stage_gate"
CURRICULUM_CHOICES = (CURRICULUM_NONE, CURRICULUM_STAGE_GATE)


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


@dataclass(frozen=True)
class CurriculumStage:
    scenario: ScenarioSpec
    stage_name: str = ""
    min_episodes: int = 0
    mean_reward_threshold: float = -math.inf
    mean_success_rate_threshold: float = 0.0


class StageGateCurriculumStrategy:
    name = CURRICULUM_STAGE_GATE

    def __init__(self, stages: list[CurriculumStage], metric_window: int = 100):
        if not stages:
            raise ValueError("StageGateCurriculumStrategy 至少需要一个阶段")
        self.stages = stages
        self.metric_window = max(1, int(metric_window))
        self.current_stage_index = 0
        self.completed_episodes_in_stage = 0
        self._rewards = deque(maxlen=self.metric_window)
        self._successes = deque(maxlen=self.metric_window)

    def current_scenario(self) -> ScenarioSpec:
        return self.current_stage.scenario

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_index]

    def current_stage_name(self) -> str:
        return self.current_stage.stage_name or str(self.current_stage_index)

    def update(self, metrics: dict[str, Any]) -> ScenarioSpec:
        # 只根据 episode 结果推进阶段；缺少统一指标时保持当前场景。
        if "episode_reward" not in metrics or "episode_success" not in metrics:
            return self.current_scenario()

        self.completed_episodes_in_stage += 1
        self._rewards.append(float(metrics["episode_reward"]))
        self._successes.append(1.0 if bool(metrics["episode_success"]) else 0.0)

        if self.current_stage_index >= len(self.stages) - 1:
            return self.current_scenario()

        stage = self.current_stage
        min_episodes = max(1, int(stage.min_episodes))
        if self.completed_episodes_in_stage < min_episodes:
            return self.current_scenario()

        mean_reward = sum(self._rewards) / len(self._rewards)
        success_rate = sum(self._successes) / len(self._successes)
        if (
            mean_reward >= stage.mean_reward_threshold
            and success_rate >= stage.mean_success_rate_threshold
        ):
            self.current_stage_index += 1
            self.completed_episodes_in_stage = 0
            self._rewards.clear()
            self._successes.clear()

        return self.current_scenario()


def build_curriculum_strategy(args: Any, scenario: ScenarioSpec) -> CurriculumStrategy:
    # 课程策略构建只解析配置并返回 runtime 策略，不接触算法或环境。
    curriculum = getattr(args, "curriculum", CURRICULUM_NONE)
    if curriculum == CURRICULUM_NONE:
        return NoCurriculumStrategy(scenario)
    if curriculum == CURRICULUM_STAGE_GATE:
        config = load_training_config(getattr(args, "training_config", None))
        return _build_stage_gate_strategy(config, scenario)
    raise ValueError(f"Unsupported curriculum strategy: {curriculum}")


def _build_stage_gate_strategy(
    config: dict[str, Any],
    fallback_scenario: ScenarioSpec,
) -> StageGateCurriculumStrategy:
    stage_gate_cfg = config.get("curriculum", {}).get("stage_gate", {}) or {}
    raw_stages = stage_gate_cfg.get("stages", []) or []
    _, stage_scenarios = build_scenario_specs(config, CURRICULUM_STAGE_GATE)
    metric_window = int(config.get("metric_window", 100))

    if not raw_stages:
        return StageGateCurriculumStrategy(
            stages=[CurriculumStage(scenario=fallback_scenario)],
            metric_window=metric_window,
        )

    stages = []
    for index, stage_cfg in enumerate(raw_stages):
        stages.append(
            CurriculumStage(
                scenario=stage_scenarios[index],
                stage_name=str(stage_cfg.get("stage_name", "")),
                min_episodes=int(stage_cfg.get("min_episodes", 0)),
                mean_reward_threshold=float(
                    stage_cfg.get("mean_reward_threshold", -math.inf)
                ),
                mean_success_rate_threshold=float(
                    stage_cfg.get("mean_success_rate_threshold", 0.0)
                ),
            )
        )
    return StageGateCurriculumStrategy(stages=stages, metric_window=metric_window)
