import os
from dataclasses import dataclass
from typing import Any, Protocol

from .paths import get_model_output_dir
from .registry import get_checkpoint_module


@dataclass
class CheckpointPayload:
    model: Any = None
    env: Any = None
    network: Any = None
    tag: str | None = None

    def has_target(self) -> bool:
        return self.model is not None or self.network is not None


class AlgorithmCheckpointHandler(Protocol):
    def prepare_resume(self) -> None: ...

    def resolve_load_path(self): ...

    def save_payload(self, payload: CheckpointPayload): ...


class ModuleCheckpointHandler:
    def __init__(self, args, run_paths=None):
        self.args = args
        self.run_paths = run_paths
        self.module = get_checkpoint_module(args.algo)

    def prepare_resume(self) -> None:
        self.module.prepare_resume(self.args, run_paths=self.run_paths)

    def resolve_load_path(self):
        return self.module.resolve_load_path(self.args, run_paths=self.run_paths)

    def save_payload(self, payload: CheckpointPayload):
        return self.module.save_checkpoint(
            self.args,
            payload=payload,
            run_paths=self.run_paths,
        )


class CheckpointManager:
    def __init__(self, args, run_paths=None):
        self.args = args
        self.run_paths = run_paths
        self.handler: AlgorithmCheckpointHandler = ModuleCheckpointHandler(
            args,
            run_paths=run_paths,
        )

    def prepare_resume(self):
        self.handler.prepare_resume()

    def resolve_load_path(self):
        return self.handler.resolve_load_path()

    def save_payload(self, payload: CheckpointPayload):
        if not payload.has_target():
            return None
        output_dir = (
            self.run_paths.output_dir
            if self.run_paths is not None
            else get_model_output_dir(self.args.algo)
        )
        os.makedirs(output_dir, exist_ok=True)
        return self.handler.save_payload(payload)

    def save(self, model=None, env=None, network=None, tag=None):
        return self.save_payload(
            CheckpointPayload(
                model=model,
                env=env,
                network=network,
                tag=tag,
            )
        )
