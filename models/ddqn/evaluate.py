import multiprocessing as mp
import os
import queue

import numpy as np
import torch

from training.evaluation import (
    EpisodeEvalResult,
    elapsed_since,
    new_eval_id,
    summarize_diagnostics,
    summarize_plant_stats,
    summarize_eval_results,
    time_eval_run,
)

from .adapter import typed_onehot_state_dim
from .ddqn import QNetwork, DuelingQNetwork, DifferentialQNetwork
from .factored_network import FactoredQNetwork
from .train_entry import _parse_hidden_sizes
from .worker_pool import build_ddqn_env


def _split_episodes(total, num_workers):
    """Split *total* episodes evenly across *num_workers* workers."""
    base = total // num_workers
    remainder = total % num_workers
    splits = []
    start = 0
    for i in range(num_workers):
        count = base + (1 if i < remainder else 0)
        if count > 0:
            splits.append((start, count))
            start += count
    return splits


def _ddqn_worker_run(env, network, num_episodes, start_index, total_episodes, worker_id):
    """Run DDQN eval episodes on a single env (used by parallel workers)."""
    details = []
    _wait_idx = env.action_space.n - 1
    for i in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        actions = 0
        info = {}
        _ep_step_times = []
        _ep_gaps = []
        _wait_choice = 0   # 能种却选了 wait
        _wait_forced = 0   # 只能 wait（冷却/阳光/满格）
        _prev_done_ts = __import__("time").time()
        while not done:
            mask = env.mask_available_actions()
            action = network.get_greedy_action(state, mask)
            if action == _wait_idx:
                if np.any(mask[:-1]):
                    _wait_choice += 1
                else:
                    _wait_forced += 1
            _t0 = __import__("time").time()
            state, reward, done, info = env.step(action)
            _dt = __import__("time").time() - _t0
            _ep_step_times.append(_dt)
            _now = __import__("time").time()
            _gap = _now - _prev_done_ts
            _ep_gaps.append(_gap)
            _prev_done_ts = _now
            if _dt > 1.0:
                print(
                    f"[Eval][DDQN][W{worker_id}] SLOW STEP "
                    f"ep={start_index + i + 1}/{total_episodes} "
                    f"step={actions} dt={_dt:.2f}s gap={_gap:.2f}s "
                    f"action={action}",
                    flush=True,
                )
            total_reward += float(reward)
            actions += 1

        episode_index = start_index + i + 1
        _diag = dict(info.get("diagnostics", {}))
        _diag["step_timing"] = {
            "total_sec": float(sum(_ep_step_times)),
            "max_sec": float(max(_ep_step_times)) if _ep_step_times else 0.0,
            "max_gap_sec": float(max(_ep_gaps)) if _ep_gaps else 0.0,
            "slow_steps": sum(1 for t in _ep_step_times if t > 1.0),
            "mean_sec": float(sum(_ep_step_times) / len(_ep_step_times))
            if _ep_step_times else 0.0,
            "wait_choice": _wait_choice,
            "wait_forced": _wait_forced,
        }
        details.append(
            EpisodeEvalResult(
                eval_id="",
                episode_index=episode_index,
                reward=float(total_reward),
                survival=float(info.get("steps", getattr(env, "steps", actions))),
                win=bool(info.get("win") is True),
                game_ended=bool(info.get("game_ended", done)),
                completed_sublevels=_optional_int(info.get("completed_sublevels")),
                zombies_killed=_optional_int(info.get("zombies_killed")),
                plants_lost=_optional_int(info.get("plants_lost")),
                actions=actions,
                extra={
                    "current_sublevel_index": info.get("current_sublevel_index"),
                    "sublevel_cleared_this_step": info.get(
                        "sublevel_cleared_this_step"
                    ),
                    "plant_stats": info.get("plant_stats", {}),
                    "diagnostics": _diag,
                },
            )
        )
        _st = _diag["step_timing"]
        print(
            f"[Eval][DDQN][W{worker_id}] episode {episode_index}/{total_episodes} | "
            f"reward={total_reward:.2f} | "
            f"survival={details[-1].survival:.0f} | "
            f"win={details[-1].win} | "
            f"actions={actions} | "
            f"wait(choice={_wait_choice}, forced={_wait_forced}) | "
            f"max_gap={_st.get('max_gap_sec', 0):.2f}s | "
            f"step_time(total={_st['total_sec']:.1f}s, max={_st['max_sec']:.2f}s, "
            f"slow={_st['slow_steps']}, mean={_st['mean_sec']*1000:.0f}ms)",
            flush=True,
        )
    return details


def _ddqn_parallel_worker(
    args, instance, env_spec, scenario_spec, state_dict,
    num_episodes, start_index, total_episodes, worker_id,
):
    """Standalone worker entry-point: builds env + network, runs episodes."""
    env = None
    envs = None
    try:
        envs = _build_eval_envs(args, [instance], env_spec, scenario_spec)
        env = envs[0]
        network = _build_network(args, env)
        network.load_state_dict(state_dict)
        network.eval()
        return _ddqn_worker_run(
            env=env,
            network=network,
            num_episodes=num_episodes,
            start_index=start_index,
            total_episodes=total_episodes,
            worker_id=worker_id,
        )
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        elif envs is not None:
            _close_envs(envs)


def _ddqn_parallel_worker_proc(
    result_queue,
    stop_event,
    args,
    instance,
    env_spec,
    scenario_spec,
    state_dict,
    num_episodes,
    start_index,
    total_episodes,
    worker_id,
):
    """Multiprocessing worker target — wraps _ddqn_parallel_worker."""
    try:
        details = _ddqn_parallel_worker(
            args=args,
            instance=instance,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            state_dict=state_dict,
            num_episodes=num_episodes,
            start_index=start_index,
            total_episodes=total_episodes,
            worker_id=worker_id,
        )
        result_queue.put(("ok", details))
    except KeyboardInterrupt:
        pass  # 被主进程终止，静默退出
    except Exception as exc:
        result_queue.put(("error", (worker_id, repr(exc))))


def _detect_architecture_from_state_dict(state_dict: dict) -> str:
    """Detect network architecture from checkpoint keys.

    Returns one of ``'cnn'``, ``'differential'``, ``'dueling'``,
    ``'factored'``, or ``'standard'``.
    """
    keys = list(state_dict.keys())
    if any(k.startswith('branch_3x3.') for k in keys):
        return 'cnn'
    if any(k.startswith('wait_head.') for k in keys):
        return 'differential'
    if any(k.startswith('head_row.') for k in keys) and any(k.startswith('head_col.') for k in keys):
        return 'factored'
    if any(k.startswith('value_head.') for k in keys):
        return 'dueling'
    return 'standard'


def _apply_detected_architecture(args, arch: str):
    """Set architecture flags on *args* to force the correct network class."""
    args.use_cnn = (arch == 'cnn')
    args.use_factored = (arch == 'factored')
    args.use_differential = (arch == 'differential')
    args.use_dueling = (arch == 'dueling')


def evaluate_ddqn(
    args,
    model_path,
    instances,
    env_spec,
    scenario_spec,
    episodes,
    num_workers=1,
):
    if not instances:
        raise ValueError("DDQN eval requires at least one game instance")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DDQN model not found: {model_path}")

    # Auto-detect architecture from checkpoint so the correct network is built
    # even if the user omits --training_config or uses a mismatched one.
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    arch = _detect_architecture_from_state_dict(state_dict)
    _apply_detected_architecture(args, arch)
    print(f"[Eval][DDQN] Detected architecture: {arch}", flush=True)

    num_workers = max(1, min(num_workers, len(instances)))
    eval_id = new_eval_id("real_ddqn")
    start_time = time_eval_run()

    episode_splits = _split_episodes(episodes, num_workers)
    actual_workers = len(episode_splits)

    all_details = []
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    print(
        f"[Eval][DDQN] Dispatching {episodes} episodes "
        f"across {actual_workers} workers (multiprocessing)"
    )

    processes = []
    for worker_id, (start_idx, count) in enumerate(episode_splits):
        p = ctx.Process(
            target=_ddqn_parallel_worker_proc,
            args=(
                result_queue, stop_event, args, instances[worker_id],
                env_spec, scenario_spec, state_dict, count, start_idx,
                episodes, worker_id,
            ),
        )
        p.start()
        processes.append(p)
        print(
            f"[Eval][DDQN] Worker {worker_id} started: "
            f"pid={instances[worker_id]['pid']} port={instances[worker_id]['port']}"
        )

    completed = 0
    try:
        while completed < len(processes):
            try:
                status, data = result_queue.get(timeout=1.0)
            except queue.Empty:
                for p in processes:
                    if p.exitcode is not None and p.exitcode != 0:
                        print(
                            f"[Eval][DDQN] Worker exited with "
                            f"code {p.exitcode}",
                            flush=True,
                        )
                continue
            completed += 1
            if status == "ok":
                all_details.extend(data)
            else:
                wid, error = data
                print(f"[Eval][DDQN] Worker {wid} error: {error}", flush=True)
    except KeyboardInterrupt:
        print(
            "\n[Eval][DDQN] Interrupted, stopping workers...",
            flush=True,
        )
        stop_event.set()
    finally:
        for p in processes:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    all_details.sort(key=lambda d: d.episode_index)

    return summarize_eval_results(
        eval_id=eval_id,
        algo="ddqn",
        env_kind="real",
        episode=None,
        step=None,
        stage_name="base",
        win_condition=scenario_spec.win_condition,
        target_sublevels=scenario_spec.target_sublevels,
        details=all_details,
        duration_sec=elapsed_since(start_time),
        model_path=model_path,
        extra={
            "game_mode_id": scenario_spec.game_mode_id,
            "rows": scenario_spec.rows,
            "cols": scenario_spec.cols,
            "initial_sun": scenario_spec.initial_sun,
            "cards": list(scenario_spec.cards),
            "plant_stats": summarize_plant_stats(all_details),
            "diagnostics": summarize_diagnostics(all_details),
        },
    )


def evaluate_ddqn_state_dict(
    args,
    state_dict,
    instances,
    env_spec,
    scenario_spec,
    episodes,
    episode=None,
    step=None,
    stage_name="base",
):
    if not instances:
        raise ValueError("DDQN eval requires at least one game instance")

    # Auto-detect architecture from state dict keys
    arch = _detect_architecture_from_state_dict(state_dict)
    _apply_detected_architecture(args, arch)

    envs = []
    try:
        envs = _build_eval_envs(
            args,
            instances,
            env_spec,
            scenario_spec,
        )
        network = _build_network(args, envs[0])
        network.load_state_dict(state_dict)
        network.eval()
        return _evaluate_with_envs(
            network,
            envs,
            model_path=None,
            scenario_spec=scenario_spec,
            episodes=episodes,
            episode=episode,
            step=step,
            stage_name=stage_name,
        )
    finally:
        _close_envs(envs)


def _build_eval_envs(args, instances, env_spec, scenario_spec):
    return [
        build_ddqn_env(
            args,
            instance,
            worker_id=f"eval-{index}",
            env_spec=env_spec,
            scenario_spec=scenario_spec,
        )
        for index, instance in enumerate(instances)
    ]


def _build_network(args, env):
    hidden_sizes = _parse_hidden_sizes(getattr(args, "ddqn_hidden_sizes", None))
    use_cnn = getattr(args, "use_cnn", False)
    if use_cnn:
        from .cnn_network import CNNQNetwork
        network = CNNQNetwork(
            env,
            learning_rate=args.ddqn_lr,
            device="cpu",
            create_optimizer=False,
        )
    else:
        use_factored = getattr(args, "use_factored", False)
        use_differential = getattr(args, "use_differential", False)
        use_dueling = getattr(args, "use_dueling", False)
        if use_factored:
            EvalNet = FactoredQNetwork
        elif use_differential:
            EvalNet = DifferentialQNetwork
        elif use_dueling:
            EvalNet = DuelingQNetwork
        else:
            EvalNet = QNetwork
        network = EvalNet(
            env,
            learning_rate=args.ddqn_lr,
            device="cpu",
            hidden_sizes=hidden_sizes,
            n_inputs_override=typed_onehot_state_dim(
                env.rows, env.cols, env.num_cards
            ),
            create_optimizer=False,
        )
    return network


def _close_envs(envs):
    for env in envs:
        if hasattr(env, "close"):
            env.close()


def _evaluate_with_envs(
    network,
    envs,
    model_path,
    scenario_spec,
    episodes,
    episode=None,
    step=None,
    stage_name="base",
):
    eval_id = new_eval_id("real_ddqn")
    start_time = time_eval_run()
    details = []

    _wait_idx = envs[0].action_space.n - 1
    for index in range(episodes):
        env = envs[index % len(envs)]
        state = env.reset()
        done = False
        total_reward = 0.0
        actions = 0
        info = {}
        _ep_step_times = []
        _ep_gaps = []
        _wait_choice = 0
        _wait_forced = 0
        _prev_done_ts = __import__("time").time()
        while not done:
            mask = env.mask_available_actions()
            action = network.get_greedy_action(state, mask)
            if action == _wait_idx:
                if np.any(mask[:-1]):
                    _wait_choice += 1
                else:
                    _wait_forced += 1
            _t0 = __import__("time").time()
            state, reward, done, info = env.step(action)
            _dt = __import__("time").time() - _t0
            _ep_step_times.append(_dt)
            _now = __import__("time").time()
            _gap = _now - _prev_done_ts
            _ep_gaps.append(_gap)
            _prev_done_ts = _now
            if _dt > 1.0:
                print(
                    f"[Eval][DDQN] SLOW STEP "
                    f"ep={index + 1}/{episodes} "
                    f"step={actions} dt={_dt:.2f}s gap={_gap:.2f}s "
                    f"action={action}",
                    flush=True,
                )
            total_reward += float(reward)
            actions += 1

        _diag = dict(info.get("diagnostics", {}))
        _diag["step_timing"] = {
            "total_sec": float(sum(_ep_step_times)),
            "max_sec": float(max(_ep_step_times)) if _ep_step_times else 0.0,
            "max_gap_sec": float(max(_ep_gaps)) if _ep_gaps else 0.0,
            "slow_steps": sum(1 for t in _ep_step_times if t > 1.0),
            "mean_sec": float(sum(_ep_step_times) / len(_ep_step_times))
            if _ep_step_times else 0.0,
            "wait_choice": _wait_choice,
            "wait_forced": _wait_forced,
        }
        details.append(
            EpisodeEvalResult(
                eval_id=eval_id,
                episode_index=index + 1,
                reward=float(total_reward),
                survival=float(info.get("steps", getattr(env, "steps", actions))),
                win=bool(info.get("win") is True),
                game_ended=bool(info.get("game_ended", done)),
                completed_sublevels=_optional_int(info.get("completed_sublevels")),
                zombies_killed=_optional_int(info.get("zombies_killed")),
                plants_lost=_optional_int(info.get("plants_lost")),
                actions=actions,
                extra={
                    "current_sublevel_index": info.get("current_sublevel_index"),
                    "sublevel_cleared_this_step": info.get(
                        "sublevel_cleared_this_step"
                    ),
                    "plant_stats": info.get("plant_stats", {}),
                    "diagnostics": _diag,
                },
            )
        )
        _st = _diag["step_timing"]
        print(
            f"[Eval][DDQN] episode {index + 1}/{episodes} | "
            f"reward={total_reward:.2f} | "
            f"survival={details[-1].survival:.0f} | "
            f"win={details[-1].win} | "
            f"actions={actions} | "
            f"wait(choice={_wait_choice}, forced={_wait_forced}) | "
            f"max_gap={_st.get('max_gap_sec', 0):.2f}s | "
            f"step_time(total={_st['total_sec']:.1f}s, max={_st['max_sec']:.2f}s, "
            f"slow={_st['slow_steps']}, mean={_st['mean_sec']*1000:.0f}ms)",
            flush=True,
        )

    return summarize_eval_results(
        eval_id=eval_id,
        algo="ddqn",
        env_kind="real",
        episode=episode,
        step=step,
        stage_name=stage_name,
        win_condition=scenario_spec.win_condition,
        target_sublevels=scenario_spec.target_sublevels,
        details=details,
        duration_sec=elapsed_since(start_time),
        model_path=model_path,
        extra={
            "game_mode_id": scenario_spec.game_mode_id,
            "rows": scenario_spec.rows,
            "cols": scenario_spec.cols,
            "initial_sun": scenario_spec.initial_sun,
            "cards": list(scenario_spec.cards),
            "plant_stats": summarize_plant_stats(details),
            "diagnostics": summarize_diagnostics(details),
        },
    )


def _optional_int(value):
    if value is None:
        return None
    return int(value)
