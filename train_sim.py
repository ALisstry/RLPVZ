"""
Simulation environment training entry point.

Usage:
    python train_sim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def _rolling_mean(data, window):
    """Return (x, y) where y is the rolling mean and x is aligned to the
    first index of each window (so the curve lines up with the raw data)."""
    data = np.asarray(data, dtype=np.float64)
    if len(data) < window:
        return np.arange(len(data)), data
    smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
    x = np.arange(window - 1, len(data))
    return x, smoothed


def plot_training(save_path, rewards, iterations, loss,
                  advantage=None, entropy=None, mean_q=None,
                  max_q=None, td_error=None, grad_norm=None):
    """3×4 training dashboard matching the DDQN LivePlotter style.

    Parameters are the per-episode metric arrays.  Eval data is loaded
    from ``eval.csv`` and curriculum events from
    ``sim_curriculum_events.csv`` (both relative to *save_path*).
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = save_path.replace(".pt", "_training.png")
    output_dir = os.path.dirname(save_path)
    eval_path = os.path.join(output_dir, "eval.csv")
    curriculum_events_path = os.path.join(output_dir, "sim_curriculum_events.csv")
    if len(rewards) == 0:
        return

    window = min(100, max(1, len(rewards)))
    max_episodes = len(rewards)

    # ── Load eval data from CSV ────────────────────────────────────────
    eval_episodes = []
    eval_rewards = []
    eval_survivals = []
    if os.path.isfile(eval_path):
        with open(eval_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = row.get("episode")
                rew = row.get("reward_mean")
                surv = row.get("survival_mean")
                if ep is None or rew is None or surv is None:
                    continue
                eval_episodes.append(int(float(ep)))
                eval_rewards.append(float(rew))
                eval_survivals.append(float(surv))

    # ── Load curriculum stage-change events ────────────────────────────
    stage_events = []
    if os.path.isfile(curriculum_events_path):
        with open(curriculum_events_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = row.get("episode")
                to_stage = row.get("to_stage")
                if ep is None or to_stage is None:
                    continue
                stage_events.append((int(float(ep)), to_stage))

    # ── Figure setup: 3×4 grid ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(
        f"Sim DDQN Training Dashboard  —  Episodes {max_episodes}  |  "
        f"Mean Reward: {np.mean(rewards[-window:]):.1f}",
        fontsize=17, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Helper: standard metric subplot ────────────────────────────────
    def _plot_metric(ax, data, title, xlabel, ylabel, color):
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(title, fontsize=11, fontweight="bold")
            return
        episodes = np.arange(len(data))
        ax.plot(episodes, data, alpha=0.25, color=color, linewidth=0.6)
        sx, sy = _rolling_mean(data, window)
        ax.plot(sx, sy, color=color, linewidth=1.8, label=f"MA{window}")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(len(data), max_episodes))
        ax.grid(True, alpha=0.3)

    # ── Row 0 ──────────────────────────────────────────────────────────
    _plot_metric(axes[0, 0], rewards, "Episode Reward",
                 "Episode", "Reward", color="tab:blue")
    if eval_episodes:
        axes[0, 0].scatter(eval_episodes, eval_rewards,
                           color="red", marker="o", s=30, zorder=5,
                           label="Eval", edgecolors="black", linewidths=0.3)
        axes[0, 0].legend(fontsize=7, loc="best")

    _plot_metric(axes[0, 1], loss, "Training Loss (MSE)",
                 "Episode", "Loss", color="tab:red")

    # Q-Value Statistics (mean + max together)
    ax_q = axes[0, 2]
    has_q = False
    if mean_q is not None and len(mean_q) > 0:
        sx, sy = _rolling_mean(mean_q, window)
        ax_q.plot(sx, sy, color="tab:blue", linewidth=1.8,
                  label=f"Mean Q (MA{window})")
        has_q = True
    if max_q is not None and len(max_q) > 0:
        sx, sy = _rolling_mean(max_q, window)
        ax_q.plot(sx, sy, color="tab:orange", linewidth=1.8,
                  label=f"Max Q (MA{window})")
        has_q = True
    if not has_q:
        ax_q.text(0.5, 0.5, "No data yet", transform=ax_q.transAxes,
                  ha="center", va="center", fontsize=10, color="gray")
    ax_q.set_title("Q-Value Statistics", fontsize=11, fontweight="bold")
    ax_q.set_xlabel("Episode")
    ax_q.set_ylabel("Q-Value")
    ax_q.tick_params(labelsize=8)
    ax_q.legend(fontsize=7, loc="best")
    ax_q.grid(True, alpha=0.3)

    _plot_metric(axes[0, 3], td_error, "|TD Error|",
                 "Episode", "|TD Error|", color="tab:purple")

    # ── Row 1 ──────────────────────────────────────────────────────────
    _plot_metric(axes[1, 0], advantage, "Mean Advantage  A(s,a)",
                 "Episode", "Advantage", color="tab:green")
    axes[1, 0].axhline(y=0, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6)

    _plot_metric(axes[1, 1], entropy, "Policy Entropy",
                 "Episode", "Entropy (nats)", color="tab:cyan")

    _plot_metric(axes[1, 2], grad_norm, "Gradient Norm",
                 "Episode", r"$||\nabla||_2$", color="tab:brown")

    # Eval Score History (replaces epsilon subplot from reference)
    ax_eval = axes[1, 3]
    if eval_episodes:
        ax_eval.plot(eval_episodes, eval_rewards,
                     color="tab:red", marker="o", markersize=5, linewidth=1.3)
    else:
        ax_eval.text(0.5, 0.5, "No eval yet", transform=ax_eval.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
    ax_eval.set_title("Evaluation Score History", fontsize=11, fontweight="bold")
    ax_eval.set_xlabel("Episode")
    ax_eval.set_ylabel("Eval Reward")
    ax_eval.tick_params(labelsize=8)
    ax_eval.grid(True, alpha=0.3)

    # ── Row 2 ──────────────────────────────────────────────────────────
    _plot_metric(axes[2, 0], iterations, "Episode Iterations (Survival)",
                 "Episode", "Frames", color="tab:blue")
    if eval_episodes:
        axes[2, 0].scatter(eval_episodes, eval_survivals,
                           color="red", marker="s", s=30, zorder=5,
                           label="Eval", edgecolors="black", linewidths=0.3)
        axes[2, 0].legend(fontsize=7, loc="best")
    if iterations is not None and len(iterations) > 0:
        axes[2, 0].axhline(y=max(iterations), color="gray",
                           linestyle=":", linewidth=1, alpha=0.5)

    # Eval Survival History
    ax_es = axes[2, 1]
    if eval_episodes:
        ax_es.plot(eval_episodes, eval_survivals,
                   color="tab:blue", marker="s", markersize=4, linewidth=1.3)
    else:
        ax_es.text(0.5, 0.5, "No eval yet", transform=ax_es.transAxes,
                   ha="center", va="center", fontsize=10, color="gray")
    ax_es.set_title("Eval Survival History", fontsize=11, fontweight="bold")
    ax_es.set_xlabel("Episode")
    ax_es.set_ylabel("Survival (frames)")
    ax_es.tick_params(labelsize=8)
    ax_es.grid(True, alpha=0.3)

    # Loss (Log Scale)
    ax_logl = axes[2, 2]
    if loss is not None and len(loss) > 0:
        sx, sy = _rolling_mean(loss, window)
        ax_logl.semilogy(sx, sy + 1e-10, color="tab:red", linewidth=1.8,
                         label=f"Loss (log, MA{window})")
        ax_logl.legend(fontsize=7, loc="best")
    else:
        ax_logl.text(0.5, 0.5, "No data yet", transform=ax_logl.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
    ax_logl.set_title("Loss (Log Scale)", fontsize=11, fontweight="bold")
    ax_logl.set_xlabel("Episode")
    ax_logl.set_ylabel("Loss")
    ax_logl.tick_params(labelsize=8)
    ax_logl.grid(True, alpha=0.3, which="both")

    # Reward vs Iterations scatter
    ax_sc = axes[2, 3]
    if (rewards is not None and len(rewards) > 0
            and iterations is not None and len(iterations) > 0):
        step = max(1, len(rewards) // 5000)
        ax_sc.scatter(rewards[::step], iterations[::step],
                      alpha=0.3, s=3, color="tab:blue", edgecolors="none")
    else:
        ax_sc.text(0.5, 0.5, "No data yet", transform=ax_sc.transAxes,
                   ha="center", va="center", fontsize=10, color="gray")
    ax_sc.set_title("Reward vs Iterations", fontsize=11, fontweight="bold")
    ax_sc.set_xlabel("Reward")
    ax_sc.set_ylabel("Iterations")
    ax_sc.tick_params(labelsize=8)
    ax_sc.grid(True, alpha=0.3)

    # ── Stage-change markers ───────────────────────────────────────────
    if stage_events:
        for idx, (ep, stage_name) in enumerate(stage_events):
            for ax in axes.flat:
                ax.axvline(ep, color="tab:red", linestyle="--",
                           linewidth=1.0, alpha=0.7,
                           label="stage change" if idx == 0 else None)
            axes[0, 0].text(
                ep, axes[0, 0].get_ylim()[1], stage_name,
                rotation=90, va="top", ha="right",
                fontsize=7, color="tab:red",
            )

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DDQN agent on SimPVZ")
    parser.parse_args()

    from simenv.trainer import train_sim
    train_sim(
        max_episodes=200000,
        buffer_size=100000,
        burn_in=10000,
        batch_size=512,
        lr=1e-4,
        network_update_freq=32,
        network_sync_freq=2000,
        eval_episodes=100,
        plot_callback=plot_training,
    )
