from __future__ import annotations

import json
import os
import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .registry import get_metrics_module
from utils.training_plotter import TrainingCurvePlotter


@dataclass(frozen=True)
class MetricEvent:
    source: str
    name: str
    value: float | int | str
    step: int | None = None
    episode: int | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSnapshot:
    algo: str
    step_count: int
    episode_count: int
    episode_rewards: list[float] = field(default_factory=list)
    mean_rewards: list[float] = field(default_factory=list)
    mean_iterations: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    force: bool = False


class MetricsWriter(Protocol):
    def on_event(self, event: MetricEvent) -> None: ...

    def on_snapshot(self, snapshot: TrainingSnapshot) -> None: ...


class MetricsPipeline:
    def __init__(self, writers: list[MetricsWriter]):
        self._writers = writers
        self._disabled_writer_ids = set()

    def emit(self, event: MetricEvent) -> None:
        self._dispatch("on_event", event)

    def emit_many(self, events: Iterable[MetricEvent]) -> None:
        for event in events:
            self.emit(event)

    def emit_snapshot(self, snapshot: TrainingSnapshot) -> None:
        self._dispatch("on_snapshot", snapshot)

    def _dispatch(self, method_name: str, payload) -> None:
        for writer in self._writers:
            writer_id = id(writer)
            if writer_id in self._disabled_writer_ids:
                continue
            try:
                getattr(writer, method_name)(payload)
            except Exception as exc:
                self._disabled_writer_ids.add(writer_id)
                print(
                    f"\n[Metrics] {writer.__class__.__name__} 已禁用: {exc}",
                    flush=True,
                )


class JsonlMetricsWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def on_event(self, event: MetricEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def on_snapshot(self, snapshot: TrainingSnapshot) -> None:
        return None


class CsvMetricsWriter:
    fieldnames = ("source", "name", "value", "step", "episode", "tags")

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._has_header = os.path.exists(path) and os.path.getsize(path) > 0

    def on_event(self, event: MetricEvent) -> None:
        with open(self.path, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if not self._has_header:
                writer.writeheader()
                self._has_header = True
            writer.writerow(
                {
                    "source": event.source,
                    "name": event.name,
                    "value": event.value,
                    "step": event.step,
                    "episode": event.episode,
                    "tags": json.dumps(event.tags, ensure_ascii=False),
                }
            )

    def on_snapshot(self, snapshot: TrainingSnapshot) -> None:
        return None


class JsonSnapshotWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def on_event(self, event: MetricEvent) -> None:
        return None

    def on_snapshot(self, snapshot: TrainingSnapshot) -> None:
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(asdict(snapshot), file, ensure_ascii=False, indent=2)


class TrainingCurveWriter:
    def __init__(self, output_path: str, refresh_freq: int):
        self.plotter = TrainingCurvePlotter(
            output_path=output_path,
            refresh_freq=refresh_freq,
        )

    def on_event(self, event: MetricEvent) -> None:
        return None

    def on_snapshot(self, snapshot: TrainingSnapshot) -> None:
        self.plotter.maybe_update(
            step_count=snapshot.step_count,
            episode_rewards=snapshot.episode_rewards,
            mean_rewards=snapshot.mean_rewards,
            mean_iterations=snapshot.mean_iterations,
            eval_steps=snapshot.eval_steps,
            eval_rewards=snapshot.eval_rewards,
            losses=snapshot.losses,
            force=snapshot.force,
        )


def build_metrics_pipeline(args, run_paths) -> MetricsPipeline:
    writers: list[MetricsWriter] = []

    writers.append(JsonlMetricsWriter(run_paths.metrics_path))
    writers.append(CsvMetricsWriter(run_paths.metrics_csv_path))
    writers.append(JsonSnapshotWriter(run_paths.metrics_snapshot_path))

    metrics_module = get_metrics_module(args.algo)
    if metrics_module is not None:
        writers.extend(metrics_module.build_metrics_writers(args, run_paths))

    return MetricsPipeline(writers)
