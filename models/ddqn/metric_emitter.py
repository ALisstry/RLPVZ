from training.metrics import MetricEvent


class DDQNMetricEmitter:
    def __init__(self, metrics):
        self.metrics = metrics

    def emit_loss(self, loss_value, transition_count, episode_count):
        self.emit(
            name="q_loss",
            value=loss_value,
            step=transition_count,
            episode=episode_count,
        )

    def emit_episode(self, message, episode_stats, transition_count):
        worker_tags = {
            "worker_id": str(message["worker_id"]),
            "pid": str(message["pid"]),
            "port": str(message["port"]),
        }
        self.emit(
            name="episode_reward",
            value=episode_stats.reward,
            step=transition_count,
            episode=episode_stats.episode,
            tags=worker_tags,
        )
        self.emit(
            name="episode_iterations",
            value=episode_stats.iterations,
            step=transition_count,
            episode=episode_stats.episode,
            tags=worker_tags,
        )
        self.emit(
            name="mean_reward",
            value=episode_stats.mean_reward,
            step=transition_count,
            episode=episode_stats.episode,
        )
        self.emit(
            name="mean_iterations",
            value=episode_stats.mean_iterations,
            step=transition_count,
            episode=episode_stats.episode,
        )

    def emit_eval(self, eval_stats, transition_count):
        self.emit(
            name="eval_reward",
            value=eval_stats.avg_score,
            step=transition_count,
            episode=eval_stats.episode,
        )
        self.emit(
            name="eval_iterations",
            value=eval_stats.avg_iterations,
            step=transition_count,
            episode=eval_stats.episode,
        )

    def emit_snapshot(self, snapshot):
        if self.metrics is not None:
            self.metrics.emit_snapshot(snapshot)

    def emit(self, name, value, step=None, episode=None, tags=None):
        if self.metrics is None:
            return
        self.metrics.emit(
            MetricEvent(
                source="ddqn",
                name=name,
                value=value,
                step=step,
                episode=episode,
                tags=tags or {},
            )
        )
