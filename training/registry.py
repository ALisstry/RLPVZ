from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class AlgorithmRegistration:
    entry_module: str
    args_module: str
    checkpoint_module: str
    model_extension: str
    metrics_module: str | None = None


ALGORITHMS = {
    "ppo": AlgorithmRegistration(
        entry_module="models.ppo.train_entry",
        args_module="models.ppo.args",
        checkpoint_module="models.ppo.checkpoint",
        model_extension=".zip",
        metrics_module="models.ppo.metrics",
    ),
    "ddqn": AlgorithmRegistration(
        entry_module="models.ddqn.train_entry",
        args_module="models.ddqn.args",
        checkpoint_module="models.ddqn.checkpoint",
        model_extension=".pt",
        metrics_module="models.ddqn.metrics",
    ),
}


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    policy_type: Literal["on_policy", "off_policy"]
    supported_execution: tuple[str, ...]
    supports_curriculum: bool
    supports_action_mask: bool


class Algorithm(Protocol):
    spec: AlgorithmSpec

    def train(self, context: Any) -> None: ...

    def describe_config(self) -> list[str]: ...


def create_algorithm(name: str, args: Any) -> Algorithm:
    registration = ALGORITHMS.get(name)
    if registration is None:
        raise ValueError(f"Unsupported algorithm: {name}")
    module = import_module(registration.entry_module)
    return module.create_algorithm(args)


def available_algorithms() -> tuple[str, ...]:
    return tuple(ALGORITHMS.keys())


def get_algorithm_registration(name: str) -> AlgorithmRegistration:
    registration = ALGORITHMS.get(name)
    if registration is None:
        raise ValueError(f"Unsupported algorithm: {name}")
    return registration


def add_algorithm_args(parser) -> None:
    for name, registration in ALGORITHMS.items():
        group = parser.add_argument_group(name)
        module = import_module(registration.args_module)
        module.add_args(group)


def get_checkpoint_module(name: str):
    registration = get_algorithm_registration(name)
    return import_module(registration.checkpoint_module)


def get_metrics_module(name: str):
    registration = get_algorithm_registration(name)
    if registration.metrics_module is None:
        return None
    return import_module(registration.metrics_module)
