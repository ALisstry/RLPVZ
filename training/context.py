import json
import os
from dataclasses import dataclass, field
from typing import Any

from utils.train_utils import get_current_stage_name, to_json_string

from .checkpoint import CheckpointManager, CheckpointPayload
from .curriculum import CurriculumStrategy, build_curriculum_strategy
from .execution import ExecutionConfig, resolve_execution
from .evaluation import EvaluationConfig, EvaluationWriter
from .metrics import MetricEvent, MetricsPipeline, build_metrics_pipeline, load_metric_events
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
    eval_game_instances: list[dict]
    eval_config: EvaluationConfig
    evaluation_writer: EvaluationWriter
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


def build_train_context(
    args,
    algorithm,
    device,
    game_instances,
    eval_game_instances,
    eval_config,
    checkpoint,
    run_paths,
):
    execution = resolve_execution(args, algorithm.spec)
    env_spec, scenario_spec = build_specs(args)
    curriculum = build_curriculum_strategy(args, scenario_spec)
    _restore_curriculum_state(args, run_paths, curriculum)
    scenario_spec = curriculum.current_scenario()
    _validate_algorithm_capabilities(args, algorithm.spec, env_spec)
    return TrainContext(
        args=args,
        device=device,
        execution=execution,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
        game_instances=game_instances,
        eval_game_instances=eval_game_instances,
        eval_config=eval_config,
        evaluation_writer=EvaluationWriter(
            run_paths.run_dir,
            save_episode_details=eval_config.save_episode_details,
        ),
        curriculum=curriculum,
        metrics=build_metrics_pipeline(args, run_paths),
        checkpoint=checkpoint,
        run_paths=run_paths,
    )


def _restore_curriculum_state(args, run_paths, curriculum) -> None:
    if not getattr(args, "auto_resume", True):
        return
    if not hasattr(curriculum, "stages") or not hasattr(
        curriculum, "current_stage_index"
    ):
        return

    metadata_path = os.path.join(run_paths.run_dir, "run_metadata.json")
    if not os.path.exists(metadata_path):
        return
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    stage_name = str(metadata.get("curriculum", {}).get("stage_name", ""))
    stage_index = _find_stage_index(curriculum, stage_name)
    if stage_index is None:
        return

    curriculum.current_stage_index = stage_index
    events = load_metric_events(run_paths.metrics_csv_path)
    stage_start_episode = _find_current_stage_start_episode(events, stage_name)
    episode_results = _collect_episode_results_after(events, stage_start_episode)
    curriculum.completed_episodes_in_stage = len(episode_results)
    curriculum._rewards.clear()
    curriculum._successes.clear()
    for reward, success in episode_results[-curriculum.metric_window :]:
        curriculum._rewards.append(reward)
        curriculum._successes.append(1.0 if success else 0.0)


def _find_stage_index(curriculum, stage_name: str) -> int | None:
    for index, stage in enumerate(curriculum.stages):
        if (stage.stage_name or str(index)) == stage_name:
            return index
    return None


def _find_current_stage_start_episode(events, stage_name: str) -> int:
    stage_start_episode = 0
    for event in events:
        if (
            event.source == "runtime"
            and event.name == "curriculum_stage_changed"
            and event.tags.get("stage_name") == stage_name
            and event.episode is not None
        ):
            stage_start_episode = int(event.episode)
    return stage_start_episode


def _collect_episode_results_after(events, stage_start_episode: int):
    by_episode = {}
    for event in events:
        if event.source != "ddqn" or event.episode is None:
            continue
        episode = int(event.episode)
        if episode <= stage_start_episode:
            continue
        item = by_episode.setdefault(episode, {})
        if event.name == "episode_reward":
            item["reward"] = float(event.value)
        elif event.name == "episode_success":
            item["success"] = bool(event.value)

    results = []
    for episode in sorted(by_episode):
        item = by_episode[episode]
        if {"reward", "success"} <= item.keys():
            results.append((item["reward"], item["success"]))
    return results


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
