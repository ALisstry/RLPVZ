import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime

import yaml

from training.constants import CONFIG_PATH


def load_training_config(path):
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def set_torch_num_threads(num_threads: int) -> None:
    import torch

    torch.set_num_threads(num_threads)


def setup_device():
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        print(f"[设备] {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"
        print("[设备] CPU")
    return device


def print_gpu_memory():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存: {allocated:.2f} GB")


def print_metadata(args, algorithm=None, run_paths=None):
    print("\r\n" + "=" * 60)
    print("PVZ 训练")
    print("=" * 60)

    actual_game_speed = min(args.speed, 10.0)
    _ = actual_game_speed * args.frameskip

    print(f"\r\n配置:")
    print(f"  算法: {args.algo}")
    print(f"  速度: {actual_game_speed}x | 帧跳过: {args.frameskip}")
    if run_paths is not None:
        print(f"  运行目录: {run_paths.run_dir}")
        print(f"  缓存模型: {run_paths.cached_model_path}")
    if getattr(args, "num_envs", 1) > 1:
        print(f"  并行环境: {args.num_envs} | base_port: {args.base_port}")
    print(f"  执行策略: {getattr(args, 'execution', 'auto')}")
    print(f"  课程学习: {getattr(args, 'curriculum', 'none')}")
    if algorithm is not None and hasattr(algorithm, "describe_config"):
        for line in algorithm.describe_config():
            print(f"  {line}")


def write_run_metadata(context, algorithm, status: str = "initialized", error=None) -> str:
    os.makedirs(context.run_paths.run_dir, exist_ok=True)
    config_snapshot_path = _write_config_snapshot(context)
    metadata_path = os.path.join(context.run_paths.run_dir, "run_metadata.json")
    metadata = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "error": repr(error) if error is not None else None,
        "training_config_snapshot": config_snapshot_path,
        "args": to_jsonable(vars(context.args)),
        "algorithm": to_jsonable(algorithm.spec),
        "execution": to_jsonable(context.execution),
        "env_spec": to_jsonable(context.env_spec),
        "scenario_spec": to_jsonable(context.scenario_spec),
        "curriculum": {
            "name": getattr(context.curriculum, "name", None),
            "stage_name": get_current_stage_name(context.curriculum),
            "scenario": to_jsonable(context.curriculum.current_scenario()),
        },
        "run_paths": to_jsonable(context.run_paths),
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


def to_jsonable(value):
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def to_json_string(value) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)


def get_current_stage_name(curriculum) -> str:
    getter = getattr(curriculum, "current_stage_name", None)
    if getter is None:
        return ""
    return str(getter())
