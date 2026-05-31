from dataclasses import dataclass
from typing import Any

from .registry import AlgorithmSpec


EXECUTION_AUTO = "auto"
EXECUTION_STRATEGIES = (
    "sb3_vec_env",
    "sync_single_env",
    "sync_vector_env",
    "async_worker_pool",
)
EXECUTION_CHOICES = (EXECUTION_AUTO, *EXECUTION_STRATEGIES)


@dataclass(frozen=True)
class ExecutionConfig:
    name: str


def require_execution(execution: ExecutionConfig, expected: str, owner: str) -> None:
    if execution.name != expected:
        raise ValueError(
            f"{owner} received execution '{execution.name}', expected '{expected}'."
        )


def resolve_execution(args: Any, spec: AlgorithmSpec) -> ExecutionConfig:
    requested = getattr(args, "execution", EXECUTION_AUTO)
    if requested == EXECUTION_AUTO:
        return ExecutionConfig(name=spec.supported_execution[0])

    if requested not in spec.supported_execution:
        supported = ", ".join(spec.supported_execution)
        raise ValueError(
            f"Algorithm '{spec.name}' does not support execution '{requested}'. "
            f"Supported: {supported}"
        )
    return ExecutionConfig(name=requested)
