from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from training.metrics import MetricEvent, TrainingSnapshot


class PPOMetricsCallback(BaseCallback):
    def __init__(self, metrics=None, window=100, verbose=0):
        super().__init__(verbose)
        self.metrics = metrics
        self.window = window
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_iterations = []
        self.mean_rewards = []
        self.mean_iterations = []
        self.losses = []
        self._recent_rewards = deque(maxlen=window)
        self._recent_iterations = deque(maxlen=window)

    def _on_step(self) -> bool:
        if self.metrics is None:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue

            self.episode_count += 1
            reward = float(episode["r"])
            iterations = float(episode["l"])
            self.episode_rewards.append(reward)
            self.episode_iterations.append(iterations)
            self._recent_rewards.append(reward)
            self._recent_iterations.append(iterations)

            mean_reward = sum(self._recent_rewards) / len(self._recent_rewards)
            mean_iteration = sum(self._recent_iterations) / len(self._recent_iterations)
            self.mean_rewards.append(mean_reward)
            self.mean_iterations.append(mean_iteration)

            if self.metrics is not None:
                self.metrics.emit_many(
                    [
                        MetricEvent(
                            source="ppo",
                            name="episode_reward",
                            value=reward,
                            step=self.num_timesteps,
                            episode=self.episode_count,
                        ),
                        MetricEvent(
                            source="ppo",
                            name="episode_iterations",
                            value=iterations,
                            step=self.num_timesteps,
                            episode=self.episode_count,
                        ),
                        MetricEvent(
                            source="ppo",
                            name="mean_reward",
                            value=mean_reward,
                            step=self.num_timesteps,
                            episode=self.episode_count,
                        ),
                        MetricEvent(
                            source="ppo",
                            name="mean_iterations",
                            value=mean_iteration,
                            step=self.num_timesteps,
                            episode=self.episode_count,
                        ),
                    ]
                )
            self._emit_snapshot()

        return True

    def _on_rollout_end(self) -> None:
        if self.metrics is None:
            return

        logger_values = getattr(self.model.logger, "name_to_value", {})
        metric_names = {
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/entropy_loss": "entropy_loss",
            "train/explained_variance": "explained_variance",
            "train/learning_rate": "learning_rate",
            "train/loss": "loss",
            "train/policy_gradient_loss": "policy_loss",
            "train/value_loss": "value_loss",
        }
        for logger_name, metric_name in metric_names.items():
            if logger_name not in logger_values:
                continue
            value = float(logger_values[logger_name])
            if metric_name == "loss":
                self.losses.append(value)
            self.metrics.emit(
                MetricEvent(
                    source="ppo",
                    name=metric_name,
                    value=value,
                    step=self.num_timesteps,
                    episode=self.episode_count or None,
                )
            )
        self._emit_snapshot()

    def _on_training_end(self) -> None:
        self._emit_snapshot(force=True)

    def _emit_snapshot(self, force=False) -> None:
        if self.metrics is None:
            return
        self.metrics.emit_snapshot(
            TrainingSnapshot(
                algo="ppo",
                step_count=self.num_timesteps,
                episode_count=self.episode_count,
                episode_rewards=list(self.episode_rewards),
                mean_rewards=list(self.mean_rewards),
                mean_iterations=list(self.mean_iterations),
                losses=list(self.losses),
                force=force,
            )
        )
