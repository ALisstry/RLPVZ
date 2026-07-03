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
    td_error_means: list[float] = field(default_factory=list)
    mean_q_values: list[float] = field(default_factory=list)
    max_q_values: list[float] = field(default_factory=list)
    entropy_values: list[float] = field(default_factory=list)
    advantage_values: list[float] = field(default_factory=list)
    grad_norm_values: list[float] = field(default_factory=list)
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
            td_error_means=snapshot.td_error_means,
            mean_q_values=snapshot.mean_q_values,
            max_q_values=snapshot.max_q_values,
            entropy_values=snapshot.entropy_values,
            advantage_values=snapshot.advantage_values,
            grad_norm_values=snapshot.grad_norm_values,
            force=snapshot.force,
        )


def load_training_snapshot(path: str) -> TrainingSnapshot | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError:
        return None
    return TrainingSnapshot(
        algo=str(raw.get("algo", "")),
        step_count=int(raw.get("step_count", 0)),
        episode_count=int(raw.get("episode_count", 0)),
        episode_rewards=[float(value) for value in raw.get("episode_rewards", [])],
        mean_rewards=[float(value) for value in raw.get("mean_rewards", [])],
        mean_iterations=[
            float(value) for value in raw.get("mean_iterations", [])
        ],
        eval_steps=[int(value) for value in raw.get("eval_steps", [])],
        eval_rewards=[float(value) for value in raw.get("eval_rewards", [])],
        losses=[float(value) for value in raw.get("losses", [])],
        td_error_means=[float(value) for value in raw.get("td_error_means", [])],
        mean_q_values=[float(value) for value in raw.get("mean_q_values", [])],
        max_q_values=[float(value) for value in raw.get("max_q_values", [])],
        entropy_values=[float(value) for value in raw.get("entropy_values", [])],
        advantage_values=[float(value) for value in raw.get("advantage_values", [])],
        grad_norm_values=[float(value) for value in raw.get("grad_norm_values", [])],
        force=bool(raw.get("force", False)),
    )


def load_metric_events(path: str) -> list[MetricEvent]:
    if not os.path.exists(path):
        return []
    events = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                events.append(
                    MetricEvent(
                        source=row.get("source", ""),
                        name=row.get("name", ""),
                        value=_parse_metric_value(row.get("value", "")),
                        step=_parse_optional_int(row.get("step")),
                        episode=_parse_optional_int(row.get("episode")),
                        tags=json.loads(row.get("tags") or "{}"),
                    )
                )
            except (ValueError, json.JSONDecodeError):
                continue
    return events


def _parse_metric_value(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def build_metrics_pipeline(args, run_paths) -> MetricsPipeline:
    writers: list[MetricsWriter] = []

    writers.append(JsonlMetricsWriter(run_paths.metrics_path))
    writers.append(CsvMetricsWriter(run_paths.metrics_csv_path))
    writers.append(JsonSnapshotWriter(run_paths.metrics_snapshot_path))

    metrics_module = get_metrics_module(args.algo)
    if metrics_module is not None:
        writers.extend(metrics_module.build_metrics_writers(args, run_paths))

    return MetricsPipeline(writers)
