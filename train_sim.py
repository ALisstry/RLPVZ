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
    """Return (x, y) where y is the NaN-aware rolling mean and x is aligned
    to the first index of each window.

    NaN values are ignored in the mean computation, so episodes with no
    training steps do not pull the rolling average toward zero.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n < window:
        return np.arange(n), data
    # NaN-aware via cumsum — prepend 0 so rolling windows are correct
    finite = np.isfinite(data)
    data_filled = np.where(finite, data, 0.0)
    cumsum = np.concatenate([[0.0], np.cumsum(data_filled)])
    cumcount = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
    rolling_sum = cumsum[window:] - cumsum[:n - window + 1]
    rolling_count = cumcount[window:] - cumcount[:n - window + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = rolling_sum / rolling_count
    smoothed[rolling_count == 0] = np.nan
    x = np.arange(window - 1, n)
    return x, smoothed


def plot_training(save_path, rewards, iterations, loss,
                  advantage=None, entropy=None, mean_q=None,
                  max_q=None, td_error=None, grad_norm=None,
                  q_wait=None, delta_mean=None, delta_max=None):
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
    step_window = min(500, max(1, len(loss) if loss is not None and len(loss) > 0 else 500))
    max_episodes = len(rewards)
    max_steps = len(loss) if loss is not None and len(loss) > 0 else 0

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
        f"Sim DDQN Training Dashboard  —  Ep {max_episodes}  "
        f"Steps {max_steps}  |  "
        f"Mean Reward: {np.mean(rewards[-window:]):.1f}",
        fontsize=17, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Helper: standard metric subplot ────────────────────────────────
    def _plot_metric(ax, data, title, xlabel, ylabel, color, *, ma_win=None):
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(title, fontsize=11, fontweight="bold")
            return
        w = ma_win if ma_win is not None else window
        episodes = np.arange(len(data))
        ax.plot(episodes, data, alpha=0.25, color=color, linewidth=0.6)
        sx, sy = _rolling_mean(data, w)
        ax.plot(sx, sy, color=color, linewidth=1.8, label=f"MA{w}")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(len(data), 1))
        ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Row 0 — Episode-level metrics (x-axis: Episode)
    # ═══════════════════════════════════════════════════════════════════
    _plot_metric(axes[0, 0], rewards, "Episode Reward",
                 "Episode", "Reward", color="tab:blue")
    if eval_episodes:
        axes[0, 0].scatter(eval_episodes, eval_rewards,
                           color="red", marker="o", s=30, zorder=5,
                           label="Eval", edgecolors="black", linewidths=0.3)
        axes[0, 0].legend(fontsize=7, loc="best")

    _plot_metric(axes[0, 1], iterations, "Episode Iterations (Survival)",
                 "Episode", "Frames", color="tab:blue")
    if eval_episodes:
        axes[0, 1].scatter(eval_episodes, eval_survivals,
                           color="red", marker="s", s=30, zorder=5,
                           label="Eval", edgecolors="black", linewidths=0.3)
        axes[0, 1].legend(fontsize=7, loc="best")
    if iterations is not None and len(iterations) > 0:
        axes[0, 1].axhline(y=max(iterations), color="gray",
                           linestyle=":", linewidth=1, alpha=0.5)

    # Eval Score History
    ax_er = axes[0, 2]
    if eval_episodes:
        ax_er.plot(eval_episodes, eval_rewards,
                   color="tab:red", marker="o", markersize=5, linewidth=1.3)
    else:
        ax_er.text(0.5, 0.5, "No eval yet", transform=ax_er.transAxes,
                   ha="center", va="center", fontsize=10, color="gray")
    ax_er.set_title("Eval Reward History", fontsize=11, fontweight="bold")
    ax_er.set_xlabel("Episode")
    ax_er.set_ylabel("Eval Reward")
    ax_er.tick_params(labelsize=8)
    ax_er.grid(True, alpha=0.3)

    # Eval Survival History
    ax_es = axes[0, 3]
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

    # ═══════════════════════════════════════════════════════════════════
    # Row 1 — Loss & Value diagnostics (x-axis: Update Step)
    # ═══════════════════════════════════════════════════════════════════
    _plot_metric(axes[1, 0], loss, "Training Loss (Huber)",
                 "Update Step", "Loss", color="tab:red", ma_win=step_window)

    # Loss (Log Scale)
    ax_logl = axes[1, 1]
    if loss is not None and len(loss) > 0:
        sx, sy = _rolling_mean(loss, step_window)
        ax_logl.semilogy(sx, sy + 1e-10, color="tab:red", linewidth=1.8,
                         label=f"Loss (log, MA{step_window})")
        ax_logl.legend(fontsize=7, loc="best")
    else:
        ax_logl.text(0.5, 0.5, "No data yet", transform=ax_logl.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
    ax_logl.set_title("Loss (Log Scale)", fontsize=11, fontweight="bold")
    ax_logl.set_xlabel("Update Step")
    ax_logl.set_ylabel("Loss")
    ax_logl.tick_params(labelsize=8)
    ax_logl.grid(True, alpha=0.3, which="both")

    # Q-Value & Differential Statistics
    ax_q = axes[1, 2]
    has_q = False
    if mean_q is not None and len(mean_q) > 0:
        sx, sy = _rolling_mean(mean_q, step_window)
        ax_q.plot(sx, sy, color="tab:blue", linewidth=1.8,
                  label=f"Mean Q (MA{step_window})")
        has_q = True
    if max_q is not None and len(max_q) > 0:
        sx, sy = _rolling_mean(max_q, step_window)
        ax_q.plot(sx, sy, color="tab:orange", linewidth=1.8,
                  label=f"Max Q (MA{step_window})")
        has_q = True
    if q_wait is not None and len(q_wait) > 0:
        sx, sy = _rolling_mean(q_wait, step_window)
        ax_q.plot(sx, sy, color="tab:green", linewidth=1.5, linestyle="--",
                  label=f"Q(wait) (MA{step_window})")
        has_q = True
    if delta_mean is not None and len(delta_mean) > 0:
        sx, sy = _rolling_mean(delta_mean, step_window)
        ax_q.plot(sx, sy, color="tab:red", linewidth=1.5, linestyle=":",
                  label=f"Δ mean (MA{step_window})")
        ax_q.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
        has_q = True
    if delta_max is not None and len(delta_max) > 0:
        sx, sy = _rolling_mean(delta_max, step_window)
        ax_q.plot(sx, sy, color="tab:purple", linewidth=1.2, linestyle="-.",
                  label=f"Δ max (MA{step_window})")
        has_q = True
    if not has_q:
        ax_q.text(0.5, 0.5, "No data yet", transform=ax_q.transAxes,
                  ha="center", va="center", fontsize=10, color="gray")
    if all(v is not None and len(v) > 0 for v in (mean_q, q_wait)):
        gap = mean_q[-1] - q_wait[-1]
        ax_q.text(0.98, 0.02, f"gap(MeanQ-Qwait)={gap:+.4f}",
                  transform=ax_q.transAxes, ha="right", va="bottom",
                  fontsize=7, color="gray",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax_q.set_title("Q-Value & Differential Stats", fontsize=11, fontweight="bold")
    ax_q.set_xlabel("Update Step")
    ax_q.set_ylabel("Q-Value")
    ax_q.tick_params(labelsize=8)
    ax_q.legend(fontsize=6, loc="best")
    ax_q.grid(True, alpha=0.3)

    _plot_metric(axes[1, 3], td_error, "|TD Error|",
                 "Update Step", "|TD Error|", color="tab:purple", ma_win=step_window)

    # ═══════════════════════════════════════════════════════════════════
    # Row 2 — Policy diagnostics & dynamics (x-axis: Update Step)
    # ═══════════════════════════════════════════════════════════════════
    _plot_metric(axes[2, 0], advantage, "Mean Advantage  A(s,a)",
                 "Update Step", "Advantage", color="tab:green", ma_win=step_window)
    axes[2, 0].axhline(y=0, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6)

    _plot_metric(axes[2, 1], entropy, "Policy Entropy",
                 "Update Step", "Entropy (nats)", color="tab:cyan", ma_win=step_window)

    _plot_metric(axes[2, 2], grad_norm, "Gradient Norm",
                 "Update Step", r"$||\nabla||_2$", color="tab:brown", ma_win=step_window)

    # Reward vs Iterations scatter (episode data, both axes are value-based)
    ax_sc = axes[2, 3]
    if (rewards is not None and len(rewards) > 0
            and iterations is not None and len(iterations) > 0):
        s = max(1, len(rewards) // 5000)
        ax_sc.scatter(rewards[::s], iterations[::s],
                      alpha=0.3, s=3, color="tab:blue", edgecolors="none")
    else:
        ax_sc.text(0.5, 0.5, "No data yet", transform=ax_sc.transAxes,
                   ha="center", va="center", fontsize=10, color="gray")
    ax_sc.set_title("Reward vs Iterations", fontsize=11, fontweight="bold")
    ax_sc.set_xlabel("Reward")
    ax_sc.set_ylabel("Iterations")
    ax_sc.tick_params(labelsize=8)
    ax_sc.grid(True, alpha=0.3)

    # ── Stage-change markers (episode-based subplots only) ────────────
    if stage_events:
        episode_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3]]
        for idx, (ep, stage_name) in enumerate(stage_events):
            for ax in episode_axes:
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

    parser = argparse.ArgumentParser(description="Train DDQN / PPO agent on SimPVZ")
    parser.add_argument(
        "--ppo",
        action="store_true",
        default=False,
        help="Use PPO instead of DDQN (on-policy Actor-Critic)",
    )
    parser.add_argument(
        "--ppo_network",
        type=str,
        default="deepmlp",
        choices=["mlp", "deepmlp", "cnn"],
        help="PPO network type (default: deepmlp)",
    )
    parser.add_argument(
        "--use_factored",
        action="store_true",
        default=False,
        help="Use factored Q-Network (DDQN only)",
    )
    parser.add_argument(
        "--use_differential",
        action="store_true",
        default=False,
        help="Use Differential Q-Network (DDQN only)",
    )
    parser.add_argument(
        "--use_cnn_v2",
        action="store_true",
        default=False,
        help="Use Row-First CNN V2 (DDQN only)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[2048, 2048],
        help="Hidden layer sizes for DDQN (default: 2048 2048)",
    )
    parser.add_argument(
        "--use_per",
        action="store_true",
        default=False,
        help="Use Prioritized Experience Replay (DDQN only)",
    )
    args = parser.parse_args()

    if args.ppo:
        from simenv.ppo import train_ppo
        train_ppo(
            max_episodes=100000,
            network_type=args.ppo_network,
            eval_episodes=100,
            plot_callback=plot_training,
            plot_freq=1000,
        )
    else:
        hidden_sizes = args.hidden_sizes if args.hidden_sizes else [2048, 2048]
        from simenv.trainer import train_sim
        train_sim(
            max_episodes=100000,
            buffer_size=100000,
            burn_in=10000,
            batch_size=512,
            lr=1e-4,
            network_update_freq=32,
            network_sync_freq=2000,
            eval_episodes=100,
            plot_callback=plot_training,
            hidden_sizes=hidden_sizes,
            use_factored=args.use_factored,
            use_differential=args.use_differential,
            use_cnn_v2=args.use_cnn_v2,
            use_per=args.use_per,
        )
