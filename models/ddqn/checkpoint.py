import os
from collections import deque

import numpy as np
import numpy as np
import torch

from training.paths import build_checkpoint_paths, get_cached_model_path


# ─────────────────────────────────────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
# Full state save / load  (model + optimizer + replay buffer + episode count)
# ─────────────────────────────────────────────────────────────────────────

def _serialize_buffer(buffer):
    """Convert replay buffer → dict of numpy arrays + optional PER tree."""
    if buffer is None or len(buffer) == 0:
        return None

    # Gather all valid entries (skip None slots in pre-allocated ring buffer)
    n = len(buffer)
    entries = []
    for i in range(buffer.memory_size):
        e = buffer.replay_memory[i]
        if e is not None:
            entries.append(e)
    # Truncate to actual stored count (ring buffer may wrap)
    entries = entries[:n]

    data = {
        "states": np.stack([e.state for e in entries]),
        "actions": np.array([e.action for e in entries], dtype=np.int64),
        "rewards": np.array([e.reward for e in entries], dtype=np.float32),
        "dones": np.array([e.done for e in entries], dtype=bool),
        "next_states": np.stack([e.next_state for e in entries]),
        "masks": np.stack([e.mask for e in entries]),
        "next_masks": np.stack([e.next_mask for e in entries]),
    }

    # PER SumTree state (for exact priority restoration)
    if hasattr(buffer, 'sum_tree'):
        data["_per_tree"] = buffer.sum_tree.tree.copy()
        data["_per_ptr"] = buffer._write_ptr
        data["_per_alpha"] = getattr(buffer, 'alpha', 0.6)
        data["_per_epsilon"] = getattr(buffer, 'epsilon', 1e-6)

    return data


def _deserialize_buffer(buffer_data):
    """Restore replay buffer from serialized dict → PrioritizedReplayBuffer."""
    if buffer_data is None:
        return None
    from .ddqn import PrioritizedReplayBuffer

    n = len(buffer_data["actions"])
    memory_size = max(n, 100000)
    alpha = float(buffer_data.get("_per_alpha", 0.6))
    epsilon = float(buffer_data.get("_per_epsilon", 1e-6))

    buf = PrioritizedReplayBuffer(
        memory_size=memory_size, burn_in=1,
        alpha=alpha, epsilon=epsilon,
    )

    # Replay transitions in order
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

    # Restore SumTree if present
    if "_per_tree" in buffer_data:
        saved_tree = buffer_data["_per_tree"]
        if len(saved_tree) == len(buf.sum_tree.tree):
            buf.sum_tree.tree = saved_tree.copy()
        buf._write_ptr = int(buffer_data.get("_per_ptr", n % memory_size))
        buf.sum_tree._ptr = buf._write_ptr
        buf.sum_tree.n_entries = n

    return buf


# ─────────────────────────────────────────────────────────────────────────
# Full state save / load  (model + optimizer + replay buffer + episode count)
# ─────────────────────────────────────────────────────────────────────────

def _serialize_buffer(buffer):
    """Convert replay buffer deque → dict of numpy arrays for serialization."""
    if buffer is None or len(buffer.replay_memory) == 0:
        return None
    entries = list(buffer.replay_memory)
    return {
        "states": np.stack([e.state for e in entries]),
        "actions": np.array([e.action for e in entries], dtype=np.int64),
        "rewards": np.array([e.reward for e in entries], dtype=np.float32),
        "dones": np.array([e.done for e in entries], dtype=bool),
        "next_states": np.stack([e.next_state for e in entries]),
        "masks": np.stack([e.mask for e in entries]),
        "next_masks": np.stack([e.next_mask for e in entries]),
    }


def _deserialize_buffer(buffer_data):
    """Restore replay buffer from serialized dict → deque of namedtuples."""
    if buffer_data is None:
        return None
    from .ddqn import experienceReplayBuffer

    buf = experienceReplayBuffer(memory_size=1, burn_in=1)  # dummy, will be rebuilt
    n = len(buffer_data["actions"])
    entries = []
    for i in range(n):
        entries.append(
            buf.Buffer(
                state=buffer_data["states"][i],
                action=int(buffer_data["actions"][i]),
                reward=float(buffer_data["rewards"][i]),
                done=bool(buffer_data["dones"][i]),
                next_state=buffer_data["next_states"][i],
                mask=buffer_data["masks"][i],
                next_mask=buffer_data["next_masks"][i],
            )
        )
    # Create a proper buffer with the right size
    memory_size = max(n, 100000)
    from collections import deque
    buf.memory_size = memory_size
    buf.replay_memory = deque(entries, maxlen=memory_size)
    return buf


def save_checkpoint(args, payload=None, run_paths=None, **_kwargs):
    from .ddqn import copy_state_dict_to_cpu

    network = payload.network if payload is not None else None
    tag = payload.tag if payload is not None else None
    extra = payload.extra if payload is not None else None

    extra = payload.extra if payload is not None else None

    if network is None:
        return None

    paths = build_checkpoint_paths(
        "ddqn",
        run_paths=run_paths,
        explicit_path=getattr(args, "ddqn_save_path", None),
        tag=tag,
    )

    # ── full state (for resume) ──
    if extra is not None:
        full_state = {
            "model_state_dict": copy_state_dict_to_cpu(network.state_dict()),
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
            print(f"  optimizer  : ✓")
        if extra.get("buffer") and len(extra["buffer"]) > 0:
            print(f"  buffer     : {len(extra['buffer'])} entries")
        print(f"  episode    : {extra.get('episode_count', 0)}")
    else:
        # ── weights-only fallback (legacy / tagged) ──
        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
        torch.save(cpu_state_dict, paths.cached_path)

    # ── tagged checkpoint: weights-only (smaller file for milestones) ──
    if paths.tagged_path:
        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
        torch.save(cpu_state_dict, paths.tagged_path)

    # ── explicit path (if specified) ──
    # ── full state (for resume) ──
    if extra is not None:
        full_state = {
            "model_state_dict": copy_state_dict_to_cpu(network.state_dict()),
            "optimizer_state_dict": extra.get("optimizer_state_dict"),
            "buffer_data": _serialize_buffer(extra.get("buffer")),
            "episode_count": extra.get("episode_count", 0),
            "transition_count": extra.get("transition_count", 0),
        }
        os.makedirs(os.path.dirname(paths.cached_path) or ".", exist_ok=True)
        torch.save(full_state, paths.cached_path)
        print(f"\n[DDQN] 完整状态已保存: {paths.cached_path}")
        if extra.get("optimizer_state_dict"):
            print(f"  optimizer  : ✓")
        if extra.get("buffer") and len(extra["buffer"].replay_memory) > 0:
            print(f"  buffer     : {len(extra['buffer'].replay_memory)} entries")
        print(f"  episode    : {extra.get('episode_count', 0)}")
    else:
        # ── weights-only fallback (legacy / tagged) ──
        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
        torch.save(cpu_state_dict, paths.cached_path)

    # ── tagged checkpoint: weights-only (smaller file for milestones) ──
    if paths.tagged_path:
        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
        torch.save(cpu_state_dict, paths.tagged_path)

    # ── explicit path (if specified) ──
    if paths.explicit_path:
        os.makedirs(os.path.dirname(paths.explicit_path) or ".", exist_ok=True)
        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
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

    # Detect format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New full-state format
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
    else:
        # Legacy weights-only format
        return checkpoint, None


def load_full_state(load_path, device="cpu"):
    """Load a full-state checkpoint and return (state_dict, extra) tuple.

    Supports both new full-state format and legacy weights-only format.
    - New format: returns (model_state_dict, {optimizer, buffer, stats})
    - Legacy format: returns (state_dict, None) — only weights were saved
    """
    if not load_path or not os.path.exists(load_path):
        return None, None

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    # Detect format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New full-state format
        extra = {}
        if checkpoint.get("optimizer_state_dict"):
            extra["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        if checkpoint.get("buffer_data"):
            extra["buffer_data"] = checkpoint["buffer_data"]
        extra["episode_count"] = checkpoint.get("episode_count", 0)
        extra["transition_count"] = checkpoint.get("transition_count", 0)
        return checkpoint["model_state_dict"], extra
    else:
        # Legacy weights-only format
        return checkpoint, None
