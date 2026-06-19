"""
Evaluate baseline agents on the SimPVZ environment.

Usage:
    python -m simenv.evaluate_baselines [--episodes 500] [--render]
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from simenv.baselines import RandomAgent, SimpleHeuristicAgent, SmartHeuristicAgent


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_agent(agent, env: SimPVZEnv, n_episodes: int = 500
                   ) -> dict:
    """Run *n_episodes* with a greedy/noisy-free policy and return statistics."""
    rewards = []
    survivals_frames = []
    actions_taken = []
    action_counts: Counter = Counter()
    max_frames = config.MAX_FRAMES

    for ep in range(n_episodes):
        env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(env)
            action_counts[int(action)] += 1
            _state, reward, done, _info = env.step(action)
            total_reward += float(reward)
            steps += 1

        rewards.append(total_reward)
        survivals_frames.append(min(max_frames, env._scene._chrono))
        actions_taken.append(steps)

        if (ep + 1) % 100 == 0:
            print(f"  [{agent.name}] {ep + 1:4d}/{n_episodes}  "
                  f"mean_R={np.mean(rewards[-100:]):8.2f}  "
                  f"mean_T={np.mean(survivals_frames[-100:]) / config.FPS:6.1f}s")

    rewards_arr = np.array(rewards)
    survivals_arr = np.array(survivals_frames)
    actions_arr = np.array(actions_taken)
    fps = config.FPS

    survived_full = int((survivals_arr >= max_frames).sum())

    return {
        "agent": agent.name,
        "episodes": n_episodes,
        "reward_mean": float(rewards_arr.mean()),
        "reward_std": float(rewards_arr.std()),
        "reward_min": float(rewards_arr.min()),
        "reward_max": float(rewards_arr.max()),
        "survival_frames_mean": float(survivals_arr.mean()),
        "survival_frames_std": float(survivals_arr.std()),
        "survival_frames_min": int(survivals_arr.min()),
        "survival_frames_max": int(survivals_arr.max()),
        "survival_seconds_mean": float(survivals_arr.mean() / fps),
        "survival_seconds_std": float(survivals_arr.std() / fps),
        "actions_mean": float(actions_arr.mean()),
        "actions_std": float(actions_arr.std()),
        "full_survival_rate": float(survived_full / n_episodes),
        "full_survival_count": survived_full,
        "top_actions": action_counts.most_common(10),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty-print
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results: dict):
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {results['agent']}  ({results['episodes']} episodes)")
    print(f"{sep}")
    fps = config.FPS
    print(f"  {'Reward:':24s} mean={results['reward_mean']:8.2f}  "
          f"std={results['reward_std']:8.2f}  "
          f"min={results['reward_min']:8.0f}  max={results['reward_max']:8.0f}")
    print(f"  {'Survival (frames):':24s} mean={results['survival_frames_mean']:8.1f}  "
          f"std={results['survival_frames_std']:8.1f}  "
          f"min={results['survival_frames_min']:5d}  max={results['survival_frames_max']:5d}")
    print(f"  {'Survival (seconds):':24s} mean={results['survival_seconds_mean']:8.1f}  "
          f"std={results['survival_seconds_std']:8.1f}")
    print(f"  {'Actions taken:':24s} mean={results['actions_mean']:8.1f}  "
          f"std={results['actions_std']:8.1f}")
    print(f"  {'Full survival (400f):':24s} "
          f"{results['full_survival_count']}/{results['episodes']} "
          f"({100 * results['full_survival_rate']:.1f}%)")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Single-episode render helper
# ═══════════════════════════════════════════════════════════════════════════════

def render_episode(agent, env: SimPVZEnv, save_dir: str = "saved"):
    """Run one episode with render data collection and save a replay GIF."""
    from simenv.render import replay_episode

    env.enable_render_collection()
    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(env)
        _state, reward, done, _info = env.step(action)
        total_reward += float(reward)
    env.disable_render_collection()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(
        save_dir, f"baseline_{agent.name.lower()}_replay.gif")
    print(f"  Saving replay ({len(env.render_data)} frames, "
          f"reward={total_reward:.0f}) → {path}")
    replay_episode(
        env.render_data, fps=15, save_path=path,
        title=f"{agent.name} — Reward: {total_reward:.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline agents on SimPVZ.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of evaluation episodes per agent.")
    parser.add_argument("--render", action="store_true",
                        help="Render one episode per agent as a GIF replay.")
    parser.add_argument("--save-dir", type=str, default="saved",
                        help="Directory for results JSON and replays.")
    args = parser.parse_args()

    agents = [
        RandomAgent(),
        SimpleHeuristicAgent(),
        SmartHeuristicAgent(),
    ]

    os.makedirs(args.save_dir, exist_ok=True)
    print("=" * 62)
    print("  SimPVZ Baseline Evaluation")
    print(f"  Episodes per agent: {args.episodes}")
    print(f"  Grid: {config.N_LANES}×{config.LANE_LENGTH}  "
          f"Max frames: {config.MAX_FRAMES} ({config.MAX_FRAMES / config.FPS}s)")
    print(f"  Plants: sunflower, peashooter, wall-nut, potatomine")
    print("=" * 62)

    all_results = []
    env = SimPVZEnv()

    for agent in agents:
        t0 = time.perf_counter()
        print(f"\n>>> Evaluating {agent.name} ...")
        results = evaluate_agent(agent, env, n_episodes=args.episodes)
        elapsed = time.perf_counter() - t0
        results["wall_time_s"] = round(elapsed, 1)
        print_results(results)
        all_results.append(results)

        if args.render:
            render_episode(agent, env, save_dir=args.save_dir)

    env.close()

    # ── Save aggregate JSON ──────────────────────────────────────────────
    result_path = os.path.join(args.save_dir, "baselines_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {result_path}")

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Summary")
    print("=" * 62)
    print(f"  {'Agent':20s} {'Reward':>8s} {'Surv(s)':>8s} "
          f"{'FullSurv':>8s} {'Time(s)':>8s}")
    print("  " + "-" * 56)
    for r in all_results:
        print(f"  {r['agent']:20s} {r['reward_mean']:8.1f} "
              f"{r['survival_seconds_mean']:8.1f} "
              f"{100 * r['full_survival_rate']:7.1f}% "
              f"{r['wall_time_s']:8.1f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
