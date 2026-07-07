import os
from pathlib import Path

import numpy as np


class TrainingCurvePlotter:
    def __init__(self, output_path: str, refresh_freq: int = 20):
        self.output_path = Path(output_path)
        self.refresh_freq = max(0, int(refresh_freq))
        self._enabled = self.refresh_freq > 0
        self._plt = None
        self._last_update_step = None

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
        td_error_means=None,
        mean_q_values=None,
        max_q_values=None,
        entropy_values=None,
        advantage_values=None,
        grad_norm_values=None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        if not force and step_count <= 0:
            return
        if not force and self._last_update_step is not None:
            if step_count - self._last_update_step < self.refresh_freq:
                return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        plots = [
            ("rewards",       lambda: self._plot_reward_trend(mean_rewards, eval_steps, eval_rewards)),
            ("episode_rewards", lambda: self._plot_episode_rewards(episode_rewards, mean_rewards)),
            ("reward_ma500",  lambda: self._plot_reward_ma500(episode_rewards, mean_rewards, eval_steps, eval_rewards)),
            ("iterations",    lambda: self._plot_iterations(mean_iterations)),
            ("loss",          lambda: self._plot_loss(losses)),
            ("td_error",      lambda: self._plot_td_error(td_error_means)),
            ("q_values",      lambda: self._plot_q_values(mean_q_values, max_q_values)),
            ("entropy",       lambda: self._plot_entropy(entropy_values)),
            ("advantage",     lambda: self._plot_line(
                advantage_values, "Advantage", "#9467bd",
                ylabel="Advantage  Q(s,a) - mean(Q)")),
            ("grad_norm",     lambda: self._plot_line(
                grad_norm_values, "Gradient Norm", "#d62728",
                ylabel="||grad||  (pre-clip)")),
            ("q_vs_reward",   lambda: self._plot_q_vs_reward(
                mean_q_values, mean_rewards, episode_rewards)),
        ]
        for name, plot_fn in plots:
            try:
                plot_fn()
            except Exception as exc:
                print(
                    f"[Plot] {name} 绘图失败 (ep={step_count}): {exc}",
                    flush=True,
                )

        self._last_update_step = step_count

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
        self._legend_if_available(ax)
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
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path("episode_rewards"))
        plt.close(fig)

    def _plot_reward_ma500(self, episode_rewards, mean_rewards,
                           eval_steps, eval_rewards):
        """500-episode moving average reward — 平滑长期趋势，消除短期噪声。"""
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Reward Trend  (500-episode moving average)")

        window = 500
        has_500 = False

        # 500-episode rolling mean
        if episode_rewards and len(episode_rewards) >= window:
            arr = np.asarray(episode_rewards, dtype=np.float64)
            kernel = np.ones(window, dtype=np.float64) / window
            ma500 = np.convolve(arr, kernel, mode="valid")
            ma500_x = np.arange(window, len(arr) + 1)
            ax.plot(
                ma500_x, ma500,
                color="#1f77b4",
                linewidth=2.5,
                label=f"mean reward (MA-500)",
            )
            has_500 = True
        elif episode_rewards is not None:
            # Not enough episodes for MA-500 window
            ax.text(
                0.5, 0.5,
                f"MA-500 requires at least {window} episodes (current: {len(episode_rewards)})",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=11, color="#9aa0a6",
            )

        # 当前窗口均值 (MA-100) 作为对比参考线
        if mean_rewards:
            ax.plot(
                np.arange(1, len(mean_rewards) + 1),
                mean_rewards,
                color="#ff7f0e",
                linewidth=1.2,
                alpha=0.7,
                label="mean reward (MA-100)",
            )

        # 评估奖励散点
        if eval_rewards and eval_steps:
            ax.plot(
                eval_steps,
                eval_rewards,
                color="#d62728",
                linewidth=1.6,
                marker="o",
                markersize=4,
                label="eval reward",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path("reward_ma500"))
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
        self._legend_if_available(ax)
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
                display_low = float(np.percentile(loss_arr, 2.0))
                display_high = float(np.percentile(loss_arr, 98.0))
            else:
                display_low = float(np.min(loss_arr))
                display_high = float(np.max(loss_arr))

            min_loss = float(np.min(loss_arr))
            max_loss = float(np.max(loss_arr))
            if display_low == display_high:
                padding = max(1.0, abs(display_high) * 0.1)
                display_low -= padding
                display_high += padding

            clipped = (loss_arr < display_low) | (loss_arr > display_high)
            visible_loss = np.clip(loss_arr, display_low, display_high)

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
                    np.clip(smooth, display_low, display_high),
                    color="#8c564b",
                    linewidth=2.0,
                    label=f"moving avg ({window})",
                )

            if np.any(clipped):
                ax.scatter(
                    x[clipped],
                    np.clip(loss_arr[clipped], display_low, display_high),
                    color="#d62728",
                    s=10,
                    alpha=0.8,
                    label="clipped spikes",
                )
                ax.text(
                    0.99,
                    0.97,
                    f"display=[{display_low:.3g}, {display_high:.3g}] | raw=[{min_loss:.3g}, {max_loss:.3g}] | clipped={np.count_nonzero(clipped)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
                )

            padding = max(1e-6, (display_high - display_low) * 0.05)
            ax.set_ylim(display_low - padding, display_high + padding)
        else:
            ax.text(
                0.5,
                0.5,
                "No loss values yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path("loss"))
        plt.close(fig)

    def _plot_td_error(self, td_error_means):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("TD Error")

        if td_error_means:
            td_arr = np.asarray(td_error_means, dtype=np.float64)
            x = np.arange(1, len(td_arr) + 1)

            if len(td_arr) >= 20:
                display_low = float(np.percentile(td_arr, 2.0))
                display_high = float(np.percentile(td_arr, 98.0))
            else:
                display_low = float(np.min(td_arr))
                display_high = float(np.max(td_arr))

            min_td = float(np.min(td_arr))
            max_td = float(np.max(td_arr))
            if display_low == display_high:
                padding = max(1.0, abs(display_high) * 0.1)
                display_low -= padding
                display_high += padding

            clipped = (td_arr < display_low) | (td_arr > display_high)
            visible_td = np.clip(td_arr, display_low, display_high)

            ax.plot(
                x,
                visible_td,
                color="#17becf",
                linewidth=1.0,
                alpha=0.85,
                label="mean |TD| (display-clipped)",
            )

            window = min(200, max(10, len(td_arr) // 20))
            if len(td_arr) >= window:
                kernel = np.ones(window, dtype=np.float64) / window
                smooth = np.convolve(td_arr, kernel, mode="valid")
                smooth_x = np.arange(window, len(td_arr) + 1)
                ax.plot(
                    smooth_x,
                    np.clip(smooth, display_low, display_high),
                    color="#e377c2",
                    linewidth=2.0,
                    label=f"moving avg ({window})",
                )

            if np.any(clipped):
                ax.scatter(
                    x[clipped],
                    np.clip(td_arr[clipped], display_low, display_high),
                    color="#d62728",
                    s=10,
                    alpha=0.8,
                    label="clipped spikes",
                )
                ax.text(
                    0.99,
                    0.97,
                    f"display=[{display_low:.3g}, {display_high:.3g}] | raw=[{min_td:.3g}, {max_td:.3g}] | clipped={np.count_nonzero(clipped)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
                )

            padding = max(1e-6, (display_high - display_low) * 0.05)
            ax.set_ylim(display_low - padding, display_high + padding)
        else:
            ax.text(
                0.5,
                0.5,
                "No TD error values yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Mean |TD Error|")
        ax.grid(True, alpha=0.3)
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path("td_error"))
        plt.close(fig)

    def _plot_q_values(self, mean_q_values, max_q_values):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Q-Values")

        has_data = False
        if mean_q_values:
            ax.plot(
                np.arange(1, len(mean_q_values) + 1),
                mean_q_values,
                color="#1f77b4",
                linewidth=1.5,
                alpha=0.8,
                label="mean Q",
            )
            has_data = True
        if max_q_values:
            ax.plot(
                np.arange(1, len(max_q_values) + 1),
                max_q_values,
                color="#ff7f0e",
                linewidth=1.5,
                alpha=0.8,
                label="max Q",
            )
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No Q-value data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11)

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Q-Value")
        ax.grid(True, alpha=0.3)
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path("q_values"))
        plt.close(fig)

    def _plot_entropy(self, entropy_values):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title("Policy Entropy  (softmax over Q-values)")

        if entropy_values:
            ent_arr = np.asarray(entropy_values, dtype=np.float64)
            x = np.arange(1, len(ent_arr) + 1)

            ax.plot(x, ent_arr,
                    color="#2ca02c", linewidth=1.0, alpha=0.7,
                    label="entropy")

            window = min(200, max(10, len(ent_arr) // 20))
            if len(ent_arr) >= window:
                kernel = np.ones(window, dtype=np.float64) / window
                smooth = np.convolve(ent_arr, kernel, mode="valid")
                smooth_x = np.arange(window, len(ent_arr) + 1)
                ax.plot(smooth_x, smooth,
                        color="#8c564b", linewidth=2.0,
                        label=f"moving avg ({window})")

            # Reference: max entropy = ln(n_actions) = ln(451) ≈ 6.11
            max_h = np.log(451)
            ax.axhline(y=max_h, color="#d62728", linewidth=1.0, linestyle="--",
                       alpha=0.6, label=f"max entropy (ln 451 ≈ {max_h:.1f})")
        else:
            ax.text(0.5, 0.5, "No entropy data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11)

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Entropy (nats)")
        ax.grid(True, alpha=0.3)
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path("entropy"))
        plt.close(fig)

    def _plot_line(self, values, title, color, ylabel, moving_avg=True):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.set_title(title)

        if values:
            arr = np.asarray(values, dtype=np.float64)
            x = np.arange(1, len(arr) + 1)
            ax.plot(x, arr, color=color, linewidth=1.0, alpha=0.7, label=ylabel)

            if moving_avg and len(arr) >= 10:
                window = min(200, max(10, len(arr) // 20))
                kernel = np.ones(window, dtype=np.float64) / window
                smooth = np.convolve(arr, kernel, mode="valid")
                smooth_x = np.arange(window, len(arr) + 1)
                ax.plot(smooth_x, smooth,
                        color="#8c564b", linewidth=2.0,
                        label=f"moving avg ({window})")
        else:
            ax.text(0.5, 0.5, f"No {title} data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11)

        ax.set_xlabel("Update Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        self._legend_if_available(ax)
        fig.tight_layout()
        fig.savefig(self._derived_path(title.lower().replace(" ", "_")))
        plt.close(fig)

    def _plot_q_vs_reward(self, mean_q_values, mean_rewards, episode_rewards):
        """Q-value vs actual reward — diagnostic for overestimation."""
        plt = self._plt
        fig, ax1 = plt.subplots(figsize=(10, 4), dpi=120)
        ax1.set_title("Q-Value vs Reward")
        ax2 = ax1.twinx()

        has_data = False

        # Left axis: mean Q (per-update, smoothed)
        if mean_q_values:
            q_arr = np.asarray(mean_q_values, dtype=np.float64)
            q_x = np.arange(1, len(q_arr) + 1)
            # Down-sample for readability: take moving avg with wide window
            window = max(10, len(q_arr) // 40) if len(q_arr) >= 40 else 1
            if window > 1:
                kernel = np.ones(window, dtype=np.float64) / window
                q_smooth = np.convolve(q_arr, kernel, mode="valid")
                q_smooth_x = np.arange(window, len(q_arr) + 1)
                ax1.plot(q_smooth_x, q_smooth, color="#1f77b4", linewidth=2.0,
                         alpha=0.9, label="mean Q (smoothed)")
            else:
                ax1.plot(q_x, q_arr, color="#1f77b4", linewidth=1.0,
                         alpha=0.7, label="mean Q")
            ax1.set_ylabel("Q-Value", color="#1f77b4")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")
            has_data = True

        # Right axis: mean reward (smoothed line)
        if mean_rewards:
            r_arr = np.asarray(mean_rewards, dtype=np.float64)
            updates_per_ep = len(mean_q_values) / max(1, len(r_arr)) if mean_q_values else 1
            r_x = np.arange(1, len(r_arr) + 1) * updates_per_ep
            ax2.plot(r_x, r_arr, color="#ff7f0e", linewidth=2.0,
                     alpha=0.9, label="mean reward")
            ax2.set_ylabel("Reward", color="#ff7f0e")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")
            has_data = True

        if not has_data:
            ax1.text(0.5, 0.5, "No data yet", transform=ax1.transAxes,
                     ha="center", va="center", fontsize=11)

        ax1.set_xlabel("Update Step  (episode rewards ≈ aligned by avg updates/ep)")
        ax1.grid(True, alpha=0.3)

        # Combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles1 or handles2:
            ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        fig.savefig(self._derived_path("q_vs_reward"))
        plt.close(fig)

    @staticmethod
    def _legend_if_available(ax) -> None:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best")
