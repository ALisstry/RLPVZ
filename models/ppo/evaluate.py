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


def evaluate_ppo(
    args,
    model_path,
    instances,
    env_spec,
    scenario_spec,
    episodes,
    device="cpu",
):
    env = get_env(
        args,
        instances,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
        load_path=model_path,
    )
    _set_vecnormalize_eval_mode(env)
    try:
        model = get_model(args, env, device=device, load_path=model_path)
        return evaluate_ppo_model(
            model=model,
            env=env,
            scenario_spec=scenario_spec,
            episodes=episodes,
            model_path=model_path,
        )
    finally:
        env.close()


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
