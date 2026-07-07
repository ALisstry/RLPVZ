import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    for i in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        actions = 0
        info = {}
        while not done:
            mask = env.mask_available_actions()
            action = network.get_greedy_action(state, mask)
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            actions += 1

        episode_index = start_index + i + 1
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
                    "diagnostics": info.get("diagnostics", {}),
                },
            )
        )
        print(
            f"[Eval][DDQN][W{worker_id}] episode {episode_index}/{total_episodes} | "
            f"reward={total_reward:.2f} | "
            f"survival={details[-1].survival:.0f} | "
            f"win={details[-1].win} | "
            f"actions={actions}",
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

    num_workers = max(1, min(num_workers, len(instances)))
    eval_id = new_eval_id("real_ddqn")
    start_time = time_eval_run()

    episode_splits = _split_episodes(episodes, num_workers)
    actual_workers = len(episode_splits)

    all_details = []
    if actual_workers == 1:
        envs = _build_eval_envs(args, instances[:1], env_spec, scenario_spec)
        try:
            network = _build_network(args, envs[0])
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            network.load_state_dict(state_dict)
            network.eval()
            start_idx, count = episode_splits[0]
            all_details = _ddqn_worker_run(
                env=envs[0],
                network=network,
                num_episodes=count,
                start_index=start_idx,
                total_episodes=episodes,
                worker_id=0,
            )
        finally:
            _close_envs(envs)
    else:
        # Pre-load state_dict once — read-only sharing across threads is safe
        temp_envs = _build_eval_envs(args, instances[:1], env_spec, scenario_spec)
        try:
            temp_network = _build_network(args, temp_envs[0])
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        finally:
            _close_envs(temp_envs)

        print(
            f"[Eval][DDQN] Dispatching {episodes} episodes "
            f"across {actual_workers} parallel workers"
        )
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {}
            for worker_id, (start_idx, count) in enumerate(episode_splits):
                future = executor.submit(
                    _ddqn_parallel_worker,
                    args=args,
                    instance=instances[worker_id],
                    env_spec=env_spec,
                    scenario_spec=scenario_spec,
                    state_dict=state_dict,
                    num_episodes=count,
                    start_index=start_idx,
                    total_episodes=episodes,
                    worker_id=worker_id,
                )
                futures[future] = worker_id

            for future in as_completed(futures):
                all_details.extend(future.result())

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
        use_factored = getattr(args, "use_factored", False)
        network = CNNQNetwork(
            env,
            learning_rate=args.ddqn_lr,
            device="cpu",
            create_optimizer=False,
            use_factored=use_factored,
        )
    else:
        use_differential = getattr(args, "use_differential", False)
        use_dueling = getattr(args, "use_dueling", False)
        if use_differential:
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

    for index in range(episodes):
        env = envs[index % len(envs)]
        state = env.reset()
        done = False
        total_reward = 0.0
        actions = 0
        info = {}
        while not done:
            mask = env.mask_available_actions()
            action = network.get_greedy_action(state, mask)
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            actions += 1

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
                    "diagnostics": info.get("diagnostics", {}),
                },
            )
        )
        print(
            f"[Eval][DDQN] episode {index + 1}/{episodes} | "
            f"reward={total_reward:.2f} | "
            f"survival={details[-1].survival:.0f} | "
            f"win={details[-1].win} | "
            f"actions={actions}",
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
