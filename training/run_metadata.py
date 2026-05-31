import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime


def write_run_metadata(context, algorithm, status: str = "initialized", error=None) -> str:
    os.makedirs(context.run_paths.run_dir, exist_ok=True)
    config_snapshot_path = _write_config_snapshot(context)
    metadata_path = os.path.join(context.run_paths.run_dir, "run_metadata.json")
    metadata = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "error": repr(error) if error is not None else None,
        "training_config_snapshot": config_snapshot_path,
        "args": _to_jsonable(vars(context.args)),
        "algorithm": _to_jsonable(algorithm.spec),
        "execution": _to_jsonable(context.execution),
        "env_spec": _to_jsonable(context.env_spec),
        "scenario_spec": _to_jsonable(context.scenario_spec),
        "curriculum": {
            "name": getattr(context.curriculum, "name", None),
            "scenario": _to_jsonable(context.curriculum.current_scenario()),
        },
        "run_paths": _to_jsonable(context.run_paths),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata_path


def _write_config_snapshot(context) -> str | None:
    config_path = getattr(context.args, "training_config", None)
    if not config_path or not os.path.exists(config_path):
        return None
    snapshot_path = os.path.join(context.run_paths.run_dir, "training_config.yaml")
    if not os.path.exists(snapshot_path):
        shutil.copy2(config_path, snapshot_path)
    return snapshot_path


def _to_jsonable(value):
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
