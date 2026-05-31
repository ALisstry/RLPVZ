from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointPayload


@dataclass
class TrainingArtifacts:
    model: Any = None
    env: Any = None
    network: Any = None

    def has_checkpoint_target(self) -> bool:
        return self.to_checkpoint_payload().has_target()

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
