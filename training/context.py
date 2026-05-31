from dataclasses import dataclass, field
from typing import Any

from .artifacts import TrainingArtifacts
from .checkpoint import CheckpointManager
from .curriculum import CurriculumStrategy
from .execution import ExecutionConfig
from .metrics import MetricsPipeline
from .paths import RunPaths
from .specs import EnvSpec, ScenarioSpec


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
