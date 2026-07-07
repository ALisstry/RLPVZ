"""
Periodic training metrics dashboard — saves a 3×4 PNG snapshot to disk.

Adapted for the RLPVZ DDQN training pipeline.  Reads data directly from
:class:`DDQNTrainingStats` and saves to the run's metrics directory.

Usage in AsyncDDQNTrainer::

    from models.ddqn.live_plotter import LivePlotter
    live = LivePlotter(save_path=run_paths.dashboard_path, update_freq=50)
    trainer.train(..., live_plotter=live)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(data, window):
    """Return (x, y) where y is the rolling mean; x is the first index of each window."""
    data = np.asarray(data, dtype=np.float64)
    if len(data) < window:
        return np.arange(len(data)), data
    smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
    x = np.arange(window - 1, len(data))
    return x, smoothed


class LivePlotter:
    """Periodic 3×4 dashboard snapshot saved during training.

    Parameters
    ----------
    save_path : str
        Full path for the output PNG (e.g. ``"run/metrics/training_dashboard.png"``).
    window : int
        Rolling-mean window size for smoothing curves (default 100).
    update_freq : int
        Save a new snapshot every N episodes (default 50).
    max_episodes : int
        Expected total episodes, used to set initial x-axis limits.
    """

    def __init__(self, save_path="training_dashboard.png", window=100,
                 update_freq=50, max_episodes=10000):
        self.save_path = save_path
        self.window = window
        self.update_freq = update_freq
        self.max_episodes = max_episodes
        self._last_update = -1
        self._has_data = False  # True after at least one successful update
        self._epsilon = 1.0    # updated from outside
        self._setup_figure()

    # ── figure setup ──────────────────────────────────────────────────────
    def _setup_figure(self):
        self.fig, self.axes = plt.subplots(3, 4, figsize=(22, 14))
        self.fig.suptitle("DDQN Training Dashboard", fontsize=17,
                          fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── public API ────────────────────────────────────────────────────────
    def set_epsilon(self, epsilon: float):
        self._epsilon = epsilon

    def update(self, stats, episode: int):
        """Redraw the dashboard and save to ``save_path``.

        Called by the training loop.  Automatically throttled to
        ``update_freq``; safe to invoke every episode.
        """
        if episode - self._last_update < self.update_freq:
            return

        d = self._extract_data(stats)
        if d is None:
            return

        self._last_update = episode

        # ── title bar ──────────────────────────────────────────────────
        self.fig.suptitle(
            f"DDQN Training Dashboard  —  Episode {episode}  |  "
            f"Mean Reward: {d['mean_reward']:.1f}  |  "
            f"Epsilon: {d['epsilon']:.4f}",
            fontsize=17, fontweight="bold", y=0.98,
        )

        # ── Row 0 ─────────────────────────────────────────────────────
        self._plot_metric(0, 0, d["rewards"], "Episode Reward",
                          "Episode", "Reward", color="tab:blue")
        if d["eval_x"] is not None and d["eval_rewards"] is not None:
            self.axes[0, 0].scatter(d["eval_x"], d["eval_rewards"],
                                    color="red", marker="o", s=30, zorder=5,
                                    label="Eval", edgecolors="black", linewidths=0.3)
            self._safe_legend(self.axes[0, 0], fontsize=7, loc="best")

        self._plot_metric(0, 1, d["loss"], "Training Loss (MSE)",
                          "Update Step", "Loss", color="tab:red")

        self._plot_q_values(0, 2, d["q_mean"], d["q_max"],
                           d["q_wait"], d["delta_mean"], d["delta_max"])

        self._plot_metric(0, 3, d["td_error"], "TD Error",
                          "Update Step", "|TD Error|", color="tab:purple")

        # ── Row 1 ─────────────────────────────────────────────────────
        self._plot_metric(1, 0, d["advantage"], "Mean Advantage A(s,a)",
                          "Update Step", "Advantage", color="tab:green")
        self.axes[1, 0].axhline(y=0, color="gray", linestyle="--",
                                linewidth=0.8, alpha=0.6)

        self._plot_metric(1, 1, d["entropy"], "Policy Entropy",
                          "Update Step", "Entropy (nats)", color="tab:cyan")

        self._plot_metric(1, 2, d["grad_norm"], "Gradient Norm",
                          "Update Step", r"$||\nabla||_2$", color="tab:brown")

        self._plot_epsilon(1, 3, d["epsilon_hist"])

        # ── Row 2 ─────────────────────────────────────────────────────
        self._plot_metric(2, 0, d["iterations"], "Episode Iterations (Survival)",
                          "Episode", "Frames", color="tab:blue")
        if d["eval_x"] is not None and d["eval_iters"] is not None:
            self.axes[2, 0].scatter(d["eval_x"], d["eval_iters"],
                                    color="red", marker="s", s=30, zorder=5,
                                    label="Eval", edgecolors="black", linewidths=0.3)
            self._safe_legend(self.axes[2, 0], fontsize=7, loc="best")

        self._plot_rolling_mean(2, 1, d["mean_rewards_list"], "Mean Reward (Rolling)",
                                "Episode", "Mean Reward", color="tab:orange")

        self._plot_log_loss(2, 2, d["loss"])

        self._plot_scatter(2, 3, d["rewards"], d["iterations"],
                           "Reward vs Iterations", "Reward", "Iterations")

        # ── save to disk ──────────────────────────────────────────────
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.savefig(self.save_path, dpi=120, bbox_inches="tight")
        self._has_data = True
        print(f"[dashboard] saved → {self.save_path}  (ep {episode})", flush=True)

    def keep_open(self):
        """Save final snapshot and close the figure.

        Only overwrites the save_path if at least one successful update
        has occurred, preventing an empty figure from clobbering a
        previous run's dashboard.
        """
        if self._has_data:
            self.fig.savefig(self.save_path, dpi=120, bbox_inches="tight")
            print(f"[dashboard] final → {self.save_path}", flush=True)
        plt.close(self.fig)

    # ── data extraction ──────────────────────────────────────────────────
    def _extract_data(self, stats):
        try:
            rewards = self._to_array(stats.training_rewards)
            iterations = self._to_array(stats.training_iterations)
            loss = self._to_array(stats.training_loss)
            advantage = self._to_array(stats.training_advantage)
            entropy = self._to_array(stats.training_entropy)
            grad_norm = self._to_array(stats.training_grad_norm)
            q_mean = self._to_array(stats.training_mean_q)
            q_max = self._to_array(stats.training_max_q)
            q_wait = self._to_array(stats.training_q_wait)
            delta_mean = self._to_array(stats.training_delta_mean)
            delta_max = self._to_array(stats.training_delta_max)
            td_error = self._to_array(stats.training_td_errors)
            mean_rewards_list = self._to_array(stats.mean_training_rewards)
            epsilon_hist = self._to_array(getattr(stats, "training_epsilons", None))

            # Eval data
            eval_rewards = self._to_array(stats.eval_rewards)
            eval_steps = self._to_array(stats.eval_steps)
            eval_rewards_array = eval_rewards
            eval_x = eval_steps
            eval_iters = None  # iterations not stored per eval currently

            n_ep = len(rewards) if rewards is not None else 0

            return {
                "rewards": rewards,
                "iterations": iterations,
                "loss": loss,
                "advantage": advantage,
                "entropy": entropy,
                "grad_norm": grad_norm,
                "q_mean": q_mean,
                "q_max": q_max,
                "q_wait": q_wait,
                "delta_mean": delta_mean,
                "delta_max": delta_max,
                "td_error": td_error,
                "epsilon_hist": epsilon_hist,
                "mean_rewards_list": mean_rewards_list,
                "eval_rewards": eval_rewards_array,
                "eval_iters": eval_iters,
                "eval_x": eval_x,
                "epsilon": self._epsilon,
                "mean_reward": (
                    np.mean(rewards[-self.window:])
                    if rewards is not None and len(rewards) > 0
                    else 0.0
                ),
            }
        except Exception:
            return None

    @staticmethod
    def _to_array(data):
        if data is None or (isinstance(data, list) and len(data) == 0):
            return None
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.ravel()
        return arr

    # ── subplot helpers ──────────────────────────────────────────────────
    @staticmethod
    def _safe_legend(ax, **kwargs):
        """Only call legend if there is at least one labeled artist."""
        handles, labels = ax.get_legend_handles_labels()
        if any(not lbl.startswith("_") for lbl in labels):
            ax.legend(**kwargs)

    def _plot_metric(self, row, col, data, title, xlabel, ylabel, color):
        ax = self.axes[row, col]
        ax.clear()
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(title, fontsize=11, fontweight="bold")
            return
        episodes = np.arange(len(data))
        ax.plot(episodes, data, alpha=0.25, color=color, linewidth=0.6)
        sx, sy = _rolling_mean(data, self.window)
        ax.plot(sx, sy, color=color, linewidth=1.8, label=f"MA{self.window}")
        self._safe_legend(ax, fontsize=7, loc="best")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, len(data))
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_q_values(self, row, col, q_mean, q_max, q_wait, delta_mean, delta_max):
        ax = self.axes[row, col]
        ax.clear()
        has_data = False
        if q_mean is not None and len(q_mean) > 0:
            sx, sy = _rolling_mean(q_mean, self.window)
            ax.plot(sx, sy, color="tab:blue", linewidth=1.8,
                    label=f"Mean Q (MA{self.window})")
            has_data = True
        if q_max is not None and len(q_max) > 0:
            sx, sy = _rolling_mean(q_max, self.window)
            ax.plot(sx, sy, color="tab:orange", linewidth=1.8,
                    label=f"Max Q (MA{self.window})")
            has_data = True
        # ── Differential Q: Q_wait (baseline "do nothing" value) ──
        if q_wait is not None and len(q_wait) > 0:
            sx, sy = _rolling_mean(q_wait, self.window)
            ax.plot(sx, sy, color="tab:green", linewidth=1.5, linestyle="--",
                    label=f"Q(wait) (MA{self.window})")
            has_data = True
        # ── Delta mean: average advantage over waiting ──
        if delta_mean is not None and len(delta_mean) > 0:
            sx, sy = _rolling_mean(delta_mean, self.window)
            ax.plot(sx, sy, color="tab:red", linewidth=1.5, linestyle=":",
                    label=f"Δ mean (MA{self.window})")
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
            has_data = True
        # ── Delta max: best action advantage over waiting ──
        if delta_max is not None and len(delta_max) > 0:
            sx, sy = _rolling_mean(delta_max, self.window)
            ax.plot(sx, sy, color="tab:purple", linewidth=1.2, linestyle="-.",
                    label=f"Δ max (MA{self.window})")
            has_data = True
        if not has_data:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("Q-Value & Differential Statistics", fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Q-Value", fontsize=9)
        ax.tick_params(labelsize=8)
        self._safe_legend(ax, fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
        n_q = max(len(q_mean) if q_mean is not None else 0,
                   len(q_max) if q_max is not None else 0)
        if n_q > 0:
            ax.set_xlim(0, n_q)

    def _plot_epsilon(self, row, col, eps_hist):
        ax = self.axes[row, col]
        ax.clear()
        if eps_hist is not None and len(eps_hist) > 0:
            ax.plot(np.arange(len(eps_hist)), eps_hist,
                    color="darkorange", linewidth=1.8)
        else:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("Exploration Rate (Epsilon)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Epsilon", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if eps_hist is not None and len(eps_hist) > 0:
            ax.set_xlim(0, len(eps_hist))

    def _plot_rolling_mean(self, row, col, data, title, xlabel, ylabel, color):
        ax = self.axes[row, col]
        ax.clear()
        if data is not None and len(data) > 0:
            ax.plot(np.arange(len(data)), data, color=color, linewidth=1.8,
                    label=f"Rolling Mean (w={self.window})")
            self._safe_legend(ax, fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if data is not None and len(data) > 0:
            ax.set_xlim(0, len(data))

    def _plot_log_loss(self, row, col, loss):
        ax = self.axes[row, col]
        ax.clear()
        if loss is not None and len(loss) > 0:
            sx, sy = _rolling_mean(loss, self.window)
            ax.semilogy(sx, sy + 1e-10, color="tab:red", linewidth=1.8,
                        label=f"Loss (log, MA{self.window})")
            self._safe_legend(ax, fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("Loss (Log Scale)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, which="both")
        if loss is not None and len(loss) > 0:
            ax.set_xlim(0, len(loss))

    def _plot_scatter(self, row, col, xdata, ydata, title, xlabel, ylabel):
        ax = self.axes[row, col]
        ax.clear()
        if (xdata is not None and len(xdata) > 0
                and ydata is not None and len(ydata) > 0):
            step = max(1, len(xdata) // 5000)
            ax.scatter(xdata[::step], ydata[::step],
                       alpha=0.3, s=3, color="tab:blue", edgecolors="none")
        else:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
