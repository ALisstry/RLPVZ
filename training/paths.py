import os
from dataclasses import dataclass
from datetime import datetime

from .registry import get_algorithm_registration


@dataclass(frozen=True)
class RunPaths:
    algo: str
    run_id: str
    output_dir: str
    run_dir: str
    cached_model_path: str
    metrics_path: str
    metrics_csv_path: str
    metrics_snapshot_path: str
    training_curve_path: str
    heatmap_path: str
    log_dir: str
    log_file_path: str


@dataclass(frozen=True)
class CheckpointPaths:
    explicit_path: str | None
    cached_path: str
    tagged_path: str | None


def build_run_paths(args) -> RunPaths:
    algo = args.algo
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_model_output_dir(algo)
    run_dir = os.path.join(output_dir, "runs", run_id)
    cached_model_path = get_cached_model_path(algo)
    log_dir = "logs"
    return RunPaths(
        algo=algo,
        run_id=run_id,
        output_dir=output_dir,
        run_dir=run_dir,
        cached_model_path=cached_model_path,
        metrics_path=os.path.join(run_dir, "metrics.jsonl"),
        metrics_csv_path=os.path.join(run_dir, "metrics.csv"),
        metrics_snapshot_path=os.path.join(run_dir, "metrics_snapshot.json"),
        training_curve_path=os.path.join(run_dir, "training_curve.png"),
        heatmap_path=os.path.join(run_dir, "heatmap.html"),
        log_dir=log_dir,
        log_file_path=os.path.join(log_dir, f"training_{run_id}.log"),
    )


def get_model_output_dir(algo: str) -> str:
    return os.path.join("models_output", algo)


def get_cached_model_path(algo: str) -> str:
    extension = get_algorithm_registration(algo).model_extension
    return os.path.join(get_model_output_dir(algo), f"latest_model{extension}")


def build_checkpoint_paths(
    algo: str,
    run_paths: RunPaths | None = None,
    explicit_path: str | None = None,
    tag: str | None = None,
) -> CheckpointPaths:
    registration = get_algorithm_registration(algo)
    cached_path = (
        run_paths.cached_model_path
        if run_paths is not None
        else get_cached_model_path(algo)
    )
    tagged_path = None
    if tag:
        output_dir = (
            run_paths.output_dir if run_paths is not None else get_model_output_dir(algo)
        )
        tagged_path = os.path.join(output_dir, f"{tag}{registration.model_extension}")
    return CheckpointPaths(
        explicit_path=explicit_path,
        cached_path=cached_path,
        tagged_path=tagged_path,
    )
