__all__ = [
    "AlgorithmSpec",
    "AsyncWorkerPool",
    "CheckpointPayload",
    "CurriculumStrategy",
    "EnvSpec",
    "ExecutionConfig",
    "ScenarioSpec",
    "TrainContext",
    "RunPaths",
    "TrainingArtifacts",
    "TrainRunner",
    "create_algorithm",
]


def __getattr__(name):
    if name == "AlgorithmSpec":
        from .registry import AlgorithmSpec

        return AlgorithmSpec
    if name == "AsyncWorkerPool":
        from .worker_pool import AsyncWorkerPool

        return AsyncWorkerPool
    if name == "CurriculumStrategy":
        from .curriculum import CurriculumStrategy

        return CurriculumStrategy
    if name == "CheckpointPayload":
        from .checkpoint import CheckpointPayload

        return CheckpointPayload
    if name == "EnvSpec":
        from .specs import EnvSpec

        return EnvSpec
    if name == "ExecutionConfig":
        from .execution import ExecutionConfig

        return ExecutionConfig
    if name == "ScenarioSpec":
        from .specs import ScenarioSpec

        return ScenarioSpec
    if name == "RunPaths":
        from .paths import RunPaths

        return RunPaths
    if name == "TrainContext":
        from .context import TrainContext

        return TrainContext
    if name == "TrainingArtifacts":
        from .artifacts import TrainingArtifacts

        return TrainingArtifacts
    if name == "TrainRunner":
        from .runner import TrainRunner

        return TrainRunner
    if name == "create_algorithm":
        from .registry import create_algorithm

        return create_algorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
