import os

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
from .ddqn import QNetwork
from .train_entry import _parse_hidden_sizes
from .worker_pool import build_ddqn_env


def evaluate_ddqn(
    args,
    model_path,
    instances,
    env_spec,
    scenario_spec,
    episodes,
):
    if not instances:
        raise ValueError("DDQN eval requires at least one game instance")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DDQN model not found: {model_path}")

    envs = []
    try:
        envs = _build_eval_envs(
            args,
            instances,
            env_spec,
            scenario_spec,
        )
        network = _build_network(args, envs[0])
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        network.load_state_dict(state_dict)
        network.eval()
        return _evaluate_with_envs(
            network,
            envs,
            model_path=model_path,
            scenario_spec=scenario_spec,
            episodes=episodes,
        )
    finally:
        _close_envs(envs)


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
    network = QNetwork(
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
