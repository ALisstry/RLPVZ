from dataclasses import dataclass

from training.metrics import TrainingSnapshot


@dataclass(frozen=True)
class EpisodeStats:
    episode: int
    reward: float
    iterations: float
    mean_reward: float
    mean_iterations: float

    @property
    def progress_line(self) -> str:
        return (
            "Episode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                self.episode,
                self.mean_reward,
                self.mean_iterations,
            )
        )


@dataclass(frozen=True)
class EvalStats:
    episode: int
    avg_score: float
    avg_iterations: float


class DDQNTrainingStats:
    def __init__(self, window: int = 100):
        self.window = window
        self.episode_count = 0
        self.training_rewards = []
        self.training_iterations = []
        self.training_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.eval_episodes = []
        self.sync_eps = []

    def record_episode(self, reward, iterations) -> EpisodeStats:
        self.episode_count += 1
        reward = float(reward)
        iterations = float(iterations)
        self.training_rewards.append(reward)
        self.training_iterations.append(iterations)

        recent_rewards = self.training_rewards[-self.window :]
        recent_iterations = self.training_iterations[-self.window :]
        mean_reward = sum(recent_rewards) / len(recent_rewards)
        mean_iterations = sum(recent_iterations) / len(recent_iterations)
        self.mean_training_rewards.append(mean_reward)
        self.mean_training_iterations.append(mean_iterations)

        return EpisodeStats(
            episode=self.episode_count,
            reward=reward,
            iterations=iterations,
            mean_reward=mean_reward,
            mean_iterations=mean_iterations,
        )

    def record_loss(self, loss_value):
        self.training_loss.append(float(loss_value))

    def record_sync(self):
        self.sync_eps.append(self.episode_count)

    def should_evaluate(self, frequency: int) -> bool:
        return self.episode_count > 0 and self.episode_count % frequency == 0

    def record_eval(self, n_iter: int) -> EvalStats:
        recent_rewards = self.training_rewards[-n_iter:]
        recent_iterations = self.training_iterations[-n_iter:]
        avg_score = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        avg_iterations = (
            sum(recent_iterations) / len(recent_iterations)
            if recent_iterations
            else 0.0
        )
        self.real_rewards.append(float(avg_score))
        self.real_iterations.append(float(avg_iterations))
        self.eval_episodes.append(self.episode_count)
        return EvalStats(
            episode=self.episode_count,
            avg_score=float(avg_score),
            avg_iterations=float(avg_iterations),
        )

    def to_snapshot(self, force=False) -> TrainingSnapshot:
        return TrainingSnapshot(
            algo="ddqn",
            step_count=self.episode_count,
            episode_count=self.episode_count,
            episode_rewards=list(self.training_rewards),
            mean_rewards=list(self.mean_training_rewards),
            mean_iterations=list(self.mean_training_iterations),
            eval_steps=list(self.eval_episodes),
            eval_rewards=list(self.real_rewards),
            losses=list(self.training_loss),
            force=force,
        )
