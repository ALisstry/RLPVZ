import os

import torch

from training.paths import build_checkpoint_paths, get_cached_model_path


def prepare_resume(args, run_paths=None):
    if args.ddqn_load_path:
        print(f"使用参数指定 DDQN 模型路径：{args.ddqn_load_path}")
        return
    if args.no_auto_resume:
        print("自动恢复已禁用，DDQN 从零开始训练")
        return

    cached_path = (
        run_paths.cached_model_path if run_paths else get_cached_model_path("ddqn")
    )
    if os.path.exists(cached_path):
        args.ddqn_load_path = cached_path
        print(f"自动恢复 DDQN: {cached_path}")
    else:
        print("未找到 DDQN 缓存模型，从零开始训练")


def resolve_load_path(args, run_paths=None):
    return getattr(args, "ddqn_load_path", None)


def save_checkpoint(args, payload=None, run_paths=None, **_kwargs):
    from .ddqn import copy_state_dict_to_cpu

    network = payload.network if payload is not None else None
    tag = payload.tag if payload is not None else None
    if network is None:
        return None

    cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    paths = build_checkpoint_paths(
        "ddqn",
        run_paths=run_paths,
        explicit_path=getattr(args, "ddqn_save_path", None),
        tag=tag,
    )

    if paths.explicit_path:
        os.makedirs(os.path.dirname(paths.explicit_path) or ".", exist_ok=True)
        torch.save(cpu_state_dict, paths.explicit_path)
    torch.save(cpu_state_dict, paths.cached_path)
    if paths.tagged_path:
        torch.save(cpu_state_dict, paths.tagged_path)

    print(f"\n模型已保存: {paths.cached_path}")
    return paths.cached_path
