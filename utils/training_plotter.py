import os
from pathlib import Path

import numpy as np


class TrainingCurvePlotter:
    def __init__(self, output_path: str, refresh_freq: int = 20):
        self.output_path = Path(output_path)
        self.refresh_freq = max(0, int(refresh_freq))
        self._enabled = self.refresh_freq > 0
        self._plt = None

        if not self._enabled:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._plt = plt
        except Exception as exc:
            self._enabled = False
            print(f"[Plot] 实时绘图已禁用: {exc}")

    @property
    def enabled(self) -> bool:
        return self._enabled and self._plt is not None

    def maybe_update(
        self,
        step_count: int,
        episode_rewards,
        mean_rewards,
        mean_iterations,
        eval_steps,
        eval_rewards,
        losses,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        if not force and (step_count <= 0 or step_count % self.refresh_freq != 0):
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._plot_reward_trend(mean_rewards, eval_steps, eval_rewards)
        self._plot_episode_rewards(episode_rewards, mean_rewards)
        self._plot_iterations(mean_iterations)
        self._plot_loss(losses)

    def _derived_path(self, suffix: str) -> str:
        return str(self.output_path.with_name(f"{self.output_path.stem}_{suffix}.png"))

    def _plot_reward_trend(self, mean_rewards, eval_steps, eval_rewards):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Reward Trend")

        if mean_rewards:
            ax.plot(
                np.arange(1, len(mean_rewards) + 1),
                mean_rewards,
                color="#1f77b4",
                linewidth=2.2,
                label="mean reward",
            )
        if eval_rewards and eval_steps:
            ax.plot(
                eval_steps,
                eval_rewards,
                color="#d62728",
                linewidth=1.8,
                marker="o",
                markersize=4,
                label="eval reward",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self._derived_path("rewards"))
        plt.close(fig)

    def _plot_episode_rewards(self, episode_rewards, mean_rewards):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Episode Rewards")

        if episode_rewards:
            ax.plot(
                np.arange(1, len(episode_rewards) + 1),
                episode_rewards,
                color="#9aa0a6",
                alpha=0.35,
                linewidth=0.9,
                label="episode reward",
            )
        if mean_rewards:
            ax.plot(
                np.arange(1, len(mean_rewards) + 1),
                mean_rewards,
                color="#1f77b4",
                linewidth=2.0,
                label="mean reward",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self._derived_path("episode_rewards"))
        plt.close(fig)

    def _plot_iterations(self, mean_iterations):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Mean Iterations")

        if mean_iterations:
            ax.plot(
                np.arange(1, len(mean_iterations) + 1),
                mean_iterations,
                color="#2ca02c",
                linewidth=2.0,
                label="mean iterations",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Iterations")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self._derived_path("iterations"))
        plt.close(fig)

    def _plot_loss(self, losses):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Loss")

        if losses:
            loss_arr = np.asarray(losses, dtype=np.float64)
            x = np.arange(1, len(loss_arr) + 1)

            if len(loss_arr) >= 20:
                display_cap = float(np.percentile(loss_arr, 98.0))
            else:
                display_cap = float(np.max(loss_arr))

            max_loss = float(np.max(loss_arr))
            if display_cap <= 0:
                display_cap = max_loss
            if max_loss > 0:
                display_cap = min(max_loss, max(display_cap, max_loss * 0.15))

            clipped = loss_arr > display_cap
            visible_loss = np.minimum(loss_arr, display_cap)

            ax.plot(
                x,
                visible_loss,
                color="#ff7f0e",
                linewidth=1.0,
                alpha=0.85,
                label="loss (display-clipped)",
            )

            window = min(200, max(10, len(loss_arr) // 20))
            if len(loss_arr) >= window:
                kernel = np.ones(window, dtype=np.float64) / window
                smooth = np.convolve(loss_arr, kernel, mode="valid")
                smooth_x = np.arange(window, len(loss_arr) + 1)
                ax.plot(
                    smooth_x,
                    np.minimum(smooth, display_cap),
                    color="#8c564b",
                    linewidth=2.0,
                    label=f"moving avg ({window})",
                )

            if np.any(clipped):
                ax.scatter(
                    x[clipped],
                    np.full(np.count_nonzero(clipped), display_cap),
                    color="#d62728",
                    s=10,
                    alpha=0.8,
                    label="clipped spikes",
                )
                ax.text(
                    0.99,
                    0.97,
                    f"display cap={display_cap:.1f} | max={max_loss:.1f} | clipped={np.count_nonzero(clipped)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
                )

            ax.set_ylim(0, display_cap * 1.05 if display_cap > 0 else 1.0)

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self._derived_path("loss"))
        plt.close(fig)
