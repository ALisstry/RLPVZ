from dataclasses import dataclass, field
from typing import Any

from utils.train_utils import get_current_stage_name, to_json_string

from .checkpoint import CheckpointManager, CheckpointPayload
from .curriculum import CurriculumStrategy, build_curriculum_strategy
from .execution import ExecutionConfig, resolve_execution
from .metrics import MetricEvent, MetricsPipeline, build_metrics_pipeline
from .paths import RunPaths
from .specs import EnvSpec, ScenarioSpec, build_specs


@dataclass
class TrainingArtifacts:
    model: Any = None
    env: Any = None
    network: Any = None

    def to_checkpoint_payload(self, tag: str | None = None) -> CheckpointPayload:
        return CheckpointPayload(
            model=self.model,
            env=self.env,
            network=self.network,
            tag=tag,
        )

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()


@dataclass
class TrainContext:
    args: Any
    device: str
    execution: ExecutionConfig
    env_spec: EnvSpec
    scenario_spec: ScenarioSpec
    game_instances: list[dict]
    curriculum: CurriculumStrategy
    metrics: MetricsPipeline
    checkpoint: CheckpointManager
    run_paths: RunPaths
    artifacts: TrainingArtifacts = field(default_factory=TrainingArtifacts)

    def update_curriculum(self, metrics: dict[str, Any]) -> tuple[bool, ScenarioSpec]:
        # Runtime 边界：算法只上报 episode 指标，阶段推进与事件记录在这里完成。
        old_stage_name = get_current_stage_name(self.curriculum)
        old_scenario = self.scenario_spec
        new_scenario = self.curriculum.update(metrics)
        changed = new_scenario != old_scenario
        if changed:
            self.scenario_spec = new_scenario
            new_stage_name = get_current_stage_name(self.curriculum)
            print(
                f"\n[Curriculum] Episode {metrics.get('episode_count')} | "
                f"{old_stage_name} -> {new_stage_name}",
                flush=True,
            )
            self.metrics.emit(
                MetricEvent(
                    source="runtime",
                    name="curriculum_stage_changed",
                    value=to_json_string(new_scenario),
                    step=metrics.get("step"),
                    episode=metrics.get("episode_count"),
                    tags={
                        "curriculum": getattr(self.curriculum, "name", ""),
                        "stage_name": get_current_stage_name(self.curriculum),
                    },
                )
            )
        return changed, new_scenario


def build_train_context(args, algorithm, device, game_instances, checkpoint, run_paths):
    execution = resolve_execution(args, algorithm.spec)
    env_spec, scenario_spec = build_specs(args)
    curriculum = build_curriculum_strategy(args, scenario_spec)
    scenario_spec = curriculum.current_scenario()
    _validate_algorithm_capabilities(args, algorithm.spec, env_spec)
    return TrainContext(
        args=args,
        device=device,
        execution=execution,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
        game_instances=game_instances,
        curriculum=curriculum,
        metrics=build_metrics_pipeline(args, run_paths),
        checkpoint=checkpoint,
        run_paths=run_paths,
    )


def _validate_algorithm_capabilities(args, algorithm_spec, env_spec) -> None:
    if env_spec.use_action_mask and not algorithm_spec.supports_action_mask:
        raise ValueError(
            f"Algorithm '{algorithm_spec.name}' does not support action masks."
        )

    uses_curriculum = getattr(args, "curriculum", "none") != "none"
    if uses_curriculum and not algorithm_spec.supports_curriculum:
        raise ValueError(
            f"Algorithm '{algorithm_spec.name}' does not support curriculum training."
        )
