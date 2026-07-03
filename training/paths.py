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


def _resolve_config_name(args) -> str:
    """Extract a short config name from the training config path.

    training_config_baseline.yaml       -> baseline
    training_config_baseline_4layer.yaml -> baseline_4layer
    training_config.yaml                -> default
    """
    config_path = getattr(args, "training_config", "training_config.yaml")
    name = os.path.splitext(os.path.basename(config_path))[0]
    # Strip common prefix for brevity
    for prefix in ("training_config_", "training_config"):
        if name.startswith(prefix) and name != prefix:
            name = name[len(prefix):]
            break
    return name or "default"


# 构建输出目录和文件地址、格式
def build_run_paths(args) -> RunPaths:
    algo = args.algo
    config_name = _resolve_config_name(args)
    output_dir = get_model_output_dir(algo, config_name)
    run_id = _resolve_run_id(args, output_dir)
    run_dir = os.path.join(output_dir, "runs", run_id)
    registration = get_algorithm_registration(algo)
    cached_model_path = os.path.join(
        run_dir, f"latest_model{registration.model_extension}"
    )
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


def _resolve_run_id(args, output_dir: str) -> str:
    if (
        getattr(args, "auto_resume", True)
        and os.path.exists(get_cached_model_path(args.algo, _resolve_config_name(args)))
    ):
        latest_run_id = _find_latest_run_id(output_dir)
        if latest_run_id:
            return latest_run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_latest_run_id(output_dir: str) -> str | None:
    runs_dir = os.path.join(output_dir, "runs")
    if not os.path.isdir(runs_dir):
        return None

    candidates = []
    for run_id in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
        metadata_path = os.path.join(run_dir, "run_metadata.json")
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(metadata_path) and not os.path.exists(metrics_path):
            continue
        marker_path = metadata_path if os.path.exists(metadata_path) else metrics_path
        candidates.append((os.path.getmtime(marker_path), run_id))
    if not candidates:
        return None
    return max(candidates)[1]


def get_model_output_dir(algo: str, config_name: str = "") -> str:
    if config_name:
        return os.path.join("models_output", algo, config_name)
    return os.path.join("models_output", algo)


def get_cached_model_path(algo: str, config_name: str = "") -> str:
    extension = get_algorithm_registration(algo).model_extension
    return os.path.join(get_model_output_dir(algo, config_name), f"latest_model{extension}")


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
        save_dir = (
            run_paths.run_dir
            if run_paths is not None
            else get_model_output_dir(algo)
        )
        tagged_path = os.path.join(save_dir, f"{tag}{registration.model_extension}")
    return CheckpointPaths(
        explicit_path=explicit_path,
        cached_path=cached_path,
        tagged_path=tagged_path,
    )
