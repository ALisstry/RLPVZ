import os

import numpy as np
import torch

from training.paths import build_checkpoint_paths, get_cached_model_path


def prepare_resume(args, run_paths=None):
    if args.ddqn_load_path:
        print(f"使用参数指定 DDQN 模型路径：{args.ddqn_load_path}")
        return
    if not args.auto_resume:
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


def _serialize_buffer(buffer):
    """Convert replay buffer → dict of numpy arrays + optional PER tree."""
    if buffer is None:
        return None

    n = len(buffer)
    if n <= 0:
        return None

    replay_memory = getattr(buffer, "replay_memory", None)
    if replay_memory is None:
        return None

    memory_size = int(getattr(buffer, "memory_size", 0) or 0)
    if memory_size <= 0:
        memory_size = max(1, len(replay_memory))

    if n < memory_size:
        entries = [replay_memory[i] for i in range(n)]
    else:
        start = int(getattr(buffer, "_write_ptr", 0))
        entries = [replay_memory[(start + i) % memory_size] for i in range(n)]

    if not entries:
        return None

    data = {
        "states": np.stack([entry.state for entry in entries]),
        "actions": np.array([entry.action for entry in entries], dtype=np.int64),
        "rewards": np.array([entry.reward for entry in entries], dtype=np.float32),
        "dones": np.array([entry.done for entry in entries], dtype=bool),
        "next_states": np.stack([entry.next_state for entry in entries]),
        "masks": np.stack([entry.mask for entry in entries]),
        "next_masks": np.stack([entry.next_mask for entry in entries]),
    }

    data["_memory_size"] = int(getattr(buffer, "memory_size", 0) or 0)
    data["_burn_in"] = int(getattr(buffer, "burn_in", 1) or 1)

    if hasattr(buffer, "sum_tree"):
        data["_per_tree"] = buffer.sum_tree.tree.copy()
        data["_per_ptr"] = int(getattr(buffer, "_write_ptr", 0))
        data["_per_alpha"] = getattr(buffer, "alpha", 0.6)
        data["_per_epsilon"] = getattr(buffer, "epsilon", 1e-6)

    return data


def _deserialize_buffer(buffer_data):
    """Restore replay buffer from serialized dict → PrioritizedReplayBuffer."""
    if buffer_data is None:
        return None

    from .ddqn import PrioritizedReplayBuffer

    n = len(buffer_data["actions"])
    if n <= 0:
        return None

    memory_size = int(buffer_data.get("_memory_size", n))
    memory_size = max(memory_size, n, 1)
    burn_in = int(buffer_data.get("_burn_in", 1))
    alpha = float(buffer_data.get("_per_alpha", 0.6))
    epsilon = float(buffer_data.get("_per_epsilon", 1e-6))

    buf = PrioritizedReplayBuffer(
        memory_size=memory_size,
        burn_in=burn_in,
        alpha=alpha,
        epsilon=epsilon,
    )

    start_index = int(buffer_data.get("_per_ptr", n % memory_size))
    if n < memory_size:
        for i in range(n):
            buf.replay_memory[i] = buf.Buffer(
                state=buffer_data["states"][i],
                action=int(buffer_data["actions"][i]),
                reward=float(buffer_data["rewards"][i]),
                done=bool(buffer_data["dones"][i]),
                next_state=buffer_data["next_states"][i],
                mask=buffer_data["masks"][i],
                next_mask=buffer_data["next_masks"][i],
            )
    else:
        for i in range(n):
            slot_idx = (start_index + i) % memory_size
            buf.replay_memory[slot_idx] = buf.Buffer(
                state=buffer_data["states"][i],
                action=int(buffer_data["actions"][i]),
                reward=float(buffer_data["rewards"][i]),
                done=bool(buffer_data["dones"][i]),
                next_state=buffer_data["next_states"][i],
                mask=buffer_data["masks"][i],
                next_mask=buffer_data["next_masks"][i],
            )

    if "_per_tree" in buffer_data:
        saved_tree = buffer_data["_per_tree"]
        if len(saved_tree) == len(buf.sum_tree.tree):
            buf.sum_tree.tree = saved_tree.copy()
        buf._write_ptr = start_index
        buf.sum_tree._ptr = buf._write_ptr
        buf.sum_tree.n_entries = n

    return buf


def save_checkpoint(args, payload=None, run_paths=None, **_kwargs):
    from .ddqn import copy_state_dict_to_cpu

    network = payload.network if payload is not None else None
    tag = payload.tag if payload is not None else None
    extra = payload.extra if payload is not None else None

    if network is None:
        return None

    paths = build_checkpoint_paths(
        "ddqn",
        run_paths=run_paths,
        explicit_path=getattr(args, "ddqn_save_path", None),
        tag=tag,
    )

    cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())

    if extra is not None:
        full_state = {
            "model_state_dict": cpu_state_dict,
            "optimizer_state_dict": extra.get("optimizer_state_dict"),
            "buffer_data": _serialize_buffer(extra.get("buffer")),
            "episode_count": extra.get("episode_count", 0),
            "transition_count": extra.get("transition_count", 0),
            "per_alpha": extra.get("per_alpha", 0.6),
            "per_beta_start": extra.get("per_beta_start", 0.4),
        }
        os.makedirs(os.path.dirname(paths.cached_path) or ".", exist_ok=True)
        torch.save(full_state, paths.cached_path)
        print(f"\n[DDQN] 完整状态已保存: {paths.cached_path}")
        if extra.get("optimizer_state_dict"):
            print("  optimizer  : ✓")
        if extra.get("buffer") and len(extra["buffer"]) > 0:
            print(f"  buffer     : {len(extra['buffer'])} entries")
        print(f"  episode    : {extra.get('episode_count', 0)}")
    else:
        torch.save(cpu_state_dict, paths.cached_path)

    if paths.tagged_path:
        torch.save(cpu_state_dict, paths.tagged_path)

    if paths.explicit_path:
        os.makedirs(os.path.dirname(paths.explicit_path) or ".", exist_ok=True)
        torch.save(cpu_state_dict, paths.explicit_path)

    print(f"模型已保存: {paths.cached_path}")
    return paths.cached_path


def load_full_state(load_path, device="cpu"):
    """Load a full-state checkpoint and return (state_dict, extra) tuple.

    Supports both new full-state format and legacy weights-only format.
    - New format: returns (model_state_dict, {optimizer, buffer, stats})
    - Legacy format: returns (state_dict, None) — only weights were saved
    """
    if not load_path or not os.path.exists(load_path):
        return None, None

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        extra = {}
        if checkpoint.get("optimizer_state_dict"):
            extra["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        if checkpoint.get("buffer_data"):
            extra["buffer_data"] = checkpoint["buffer_data"]
        extra["episode_count"] = checkpoint.get("episode_count", 0)
        extra["transition_count"] = checkpoint.get("transition_count", 0)
        extra["per_alpha"] = checkpoint.get("per_alpha", 0.6)
        extra["per_beta_start"] = checkpoint.get("per_beta_start", 0.4)
        return checkpoint["model_state_dict"], extra

    return checkpoint, None
