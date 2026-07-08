import multiprocessing as mp
import queue

from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecNormalize

from training.evaluation import (
    EpisodeEvalResult,
    elapsed_since,
    new_eval_id,
    summarize_diagnostics,
    summarize_plant_stats,
    summarize_eval_results,
    time_eval_run,
)

from .env import get_env
from .model import get_model


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


def _ppo_worker_run(model, env, num_episodes, start_index, total_episodes, worker_id):
    """Run PPO eval episodes on a single env (used by parallel workers)."""
    _set_vecnormalize_eval_mode(env)
    details = []
    obs = env.reset()

    episode_reward = 0.0
    episode_actions = 0

    while len(details) < num_episodes:
        action_masks = get_action_masks(env)
        actions, _states = model.predict(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )
        obs, rewards, dones, infos = env.step(actions)

        # Single-env mode: unpack scalar values
        reward = float(rewards[0]) if hasattr(rewards, "__len__") else float(rewards)
        done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
        info = infos[0] if hasattr(infos, "__len__") else infos

        episode_reward += reward
        episode_actions += 1

        if not done:
            continue

        episode_index = start_index + len(details) + 1
        details.append(
            EpisodeEvalResult(
                eval_id="",
                episode_index=episode_index,
                reward=float(episode_reward),
                survival=float(
                    info.get(
                        "steps",
                        (info.get("episode") or {}).get("l", episode_actions),
                    )
                ),
                win=bool(info.get("win") is True),
                game_ended=bool(info.get("game_ended", done)),
                completed_sublevels=_optional_int(
                    info.get("completed_sublevels")
                ),
                zombies_killed=_optional_int(info.get("zombies_killed")),
                plants_lost=_optional_int(info.get("plants_lost")),
                actions=episode_actions,
                extra={
                    "current_sublevel_index": info.get(
                        "current_sublevel_index"
                    ),
                    "sublevel_cleared_this_step": info.get(
                        "sublevel_cleared_this_step"
                    ),
                    "plant_stats": info.get("plant_stats", {}),
                    "diagnostics": info.get("diagnostics", {}),
                },
            )
        )
        print(
            f"[Eval][PPO][W{worker_id}] episode {episode_index}/{total_episodes} | "
            f"reward={details[-1].reward:.2f} | "
            f"survival={details[-1].survival:.0f} | "
            f"win={details[-1].win} | "
            f"actions={details[-1].actions}",
            flush=True,
        )
        episode_reward = 0.0
        episode_actions = 0
        if len(details) >= num_episodes:
            break

    return details


def _ppo_parallel_worker(
    args, instance, env_spec, scenario_spec, model_path, device,
    num_episodes, start_index, total_episodes, worker_id,
):
    """Standalone worker: builds env + model, runs episodes."""
    env = None
    try:
        env = get_env(
            args,
            [instance],
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            load_path=model_path,
        )
        model = get_model(args, env, device=device, load_path=model_path)
        return _ppo_worker_run(
            model=model,
            env=env,
            num_episodes=num_episodes,
            start_index=start_index,
            total_episodes=total_episodes,
            worker_id=worker_id,
        )
    finally:
        if env is not None:
            env.close()


def _ppo_parallel_worker_proc(
    result_queue,
    stop_event,
    args,
    instance,
    env_spec,
    scenario_spec,
    model_path,
    device,
    num_episodes,
    start_index,
    total_episodes,
    worker_id,
):
    """Multiprocessing worker target — wraps _ppo_parallel_worker."""
    try:
        details = _ppo_parallel_worker(
            args=args,
            instance=instance,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            model_path=model_path,
            device=device,
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


def evaluate_ppo(
    args,
    model_path,
    instances,
    env_spec,
    scenario_spec,
    episodes,
    device="cpu",
    num_workers=1,
):
    num_workers = max(1, min(num_workers, len(instances)))
    eval_id = new_eval_id("real_ppo")
    start_time = time_eval_run()

    episode_splits = _split_episodes(episodes, num_workers)
    actual_workers = len(episode_splits)

    all_details = []
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    print(
        f"[Eval][PPO] Dispatching {episodes} episodes "
        f"across {actual_workers} workers (multiprocessing)"
    )

    processes = []
    for worker_id, (start_idx, count) in enumerate(episode_splits):
        p = ctx.Process(
            target=_ppo_parallel_worker_proc,
            args=(
                result_queue, stop_event, args, instances[worker_id],
                env_spec, scenario_spec, model_path, device, count,
                start_idx, episodes, worker_id,
            ),
        )
        p.start()
        processes.append(p)
        print(
            f"[Eval][PPO] Worker {worker_id} started: "
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
                            f"[Eval][PPO] Worker exited with "
                            f"code {p.exitcode}",
                            flush=True,
                        )
                continue
            completed += 1
            if status == "ok":
                all_details.extend(data)
            else:
                wid, error = data
                print(f"[Eval][PPO] Worker {wid} error: {error}", flush=True)
    except KeyboardInterrupt:
        print(
            "\n[Eval][PPO] Interrupted, stopping workers...",
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
        algo="ppo",
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


def evaluate_ppo_model(
    model,
    env,
    scenario_spec,
    episodes,
    model_path=None,
    episode=None,
    step=None,
):
    _set_vecnormalize_eval_mode(env)
    eval_id = new_eval_id("real_ppo")
    start_time = time_eval_run()
    details = []
    obs = env.reset()
    episode_rewards = [0.0 for _ in range(env.num_envs)]
    episode_actions = [0 for _ in range(env.num_envs)]

    while len(details) < episodes:
        action_masks = get_action_masks(env)
        actions, _states = model.predict(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )
        obs, rewards, dones, infos = env.step(actions)
        for env_index, done in enumerate(dones):
            episode_rewards[env_index] += float(rewards[env_index])
            episode_actions[env_index] += 1
            if not done:
                continue

            info = infos[env_index]
            details.append(
                EpisodeEvalResult(
                    eval_id=eval_id,
                    episode_index=len(details) + 1,
                    reward=float(episode_rewards[env_index]),
                    survival=float(
                        info.get(
                            "steps",
                            (info.get("episode") or {}).get(
                                "l", episode_actions[env_index]
                            ),
                        )
                    ),
                    win=bool(info.get("win") is True),
                    game_ended=bool(info.get("game_ended", done)),
                    completed_sublevels=_optional_int(
                        info.get("completed_sublevels")
                    ),
                    zombies_killed=_optional_int(info.get("zombies_killed")),
                    plants_lost=_optional_int(info.get("plants_lost")),
                    actions=episode_actions[env_index],
                    extra={
                        "current_sublevel_index": info.get(
                            "current_sublevel_index"
                        ),
                        "sublevel_cleared_this_step": info.get(
                            "sublevel_cleared_this_step"
                        ),
                        "plant_stats": info.get("plant_stats", {}),
                        "diagnostics": info.get("diagnostics", {}),
                    },
                )
            )
            print(
                f"[Eval][PPO] episode {len(details)}/{episodes} | "
                f"reward={details[-1].reward:.2f} | "
                f"survival={details[-1].survival:.0f} | "
                f"win={details[-1].win} | "
                f"actions={details[-1].actions}",
                flush=True,
            )
            episode_rewards[env_index] = 0.0
            episode_actions[env_index] = 0
            if len(details) >= episodes:
                break

    return summarize_eval_results(
        eval_id=eval_id,
        algo="ppo",
        env_kind="real",
        episode=episode,
        step=step,
        stage_name="base",
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


def _set_vecnormalize_eval_mode(env):
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            current.training = False
            current.norm_reward = False
            return
        current = getattr(current, "venv", None)


def _optional_int(value):
    if value is None:
        return None
    return int(value)
