from dataclasses import dataclass

from training.metrics import MetricEvent, TrainingSnapshot

# 模型状态监视器
# 实现
# DDQNTrainingStats: 维护 episode、loss、eval 等训练统计，并生成 TrainingSnapshot。
# DDQNMetricEmitter: 将 DDQN 训练事件转换为通用 MetricsPipeline 事件。
# DDQNConsoleReporter: 输出 DDQN 控制台进度、eval、checkpoint 和结束信息。
# DDQNWorkerStatus: 跟踪 worker 存活状态，处理 worker warning/error。


@dataclass(frozen=True)
class EpisodeStats:
    episode: int
    reward: float
    iterations: float
    success: bool
    mean_reward: float
    mean_iterations: float
    mean_success_rate: float
    mean_window_size: int
    mean_window_count: int

    @property
    def progress_line(self) -> str:
        return (
            "Episode: {:d} Reward: {:.2f} Iterations: {:.2f} Win: {} | "
            "Mean({:d}/{:d}) Reward: {:.2f} Iterations: {:.2f} WinRate: {:.2f}%".format(
                self.episode,
                self.reward,
                self.iterations,
                self.success,
                self.mean_window_count,
                self.mean_window_size,
                self.mean_reward,
                self.mean_iterations,
                self.mean_success_rate * 100.0,
            )
        )


@dataclass(frozen=True)
class EvalStats:
    episode: int
    avg_score: float
    avg_iterations: float


class DDQNTrainingStats:
    _MAX_LOSS_HISTORY = 20000
    _MAX_EPISODE_HISTORY = 10000

    def __init__(self, window: int = 100):
        self.window = window
        self.episode_count = 0
        self.training_rewards = []
        self.training_iterations = []
        self.training_successes = []
        self.training_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.eval_episodes = []
        self.sync_eps = []

    def record_episode(self, reward, iterations, success) -> EpisodeStats:
        self.episode_count += 1
        reward = float(reward)
        iterations = float(iterations)
        success = bool(success)
        self.training_rewards.append(reward)
        self.training_iterations.append(iterations)
        self.training_successes.append(1.0 if success else 0.0)

        # Cap per-episode lists
        if len(self.training_rewards) > self._MAX_EPISODE_HISTORY:
            self.training_rewards = self.training_rewards[-self._MAX_EPISODE_HISTORY:]
            self.training_iterations = self.training_iterations[-self._MAX_EPISODE_HISTORY:]

        recent_rewards = self.training_rewards[-self.window :]
        recent_iterations = self.training_iterations[-self.window :]
        recent_successes = self.training_successes[-self.window :]
        mean_reward = sum(recent_rewards) / len(recent_rewards)
        mean_iterations = sum(recent_iterations) / len(recent_iterations)
        mean_success_rate = sum(recent_successes) / len(recent_successes)
        self.mean_training_rewards.append(mean_reward)
        self.mean_training_iterations.append(mean_iterations)

        # Cap mean lists
        if len(self.mean_training_rewards) > self._MAX_EPISODE_HISTORY:
            self.mean_training_rewards = self.mean_training_rewards[-self._MAX_EPISODE_HISTORY:]
            self.mean_training_iterations = self.mean_training_iterations[-self._MAX_EPISODE_HISTORY:]

        return EpisodeStats(
            episode=self.episode_count,
            reward=reward,
            iterations=iterations,
            success=success,
            mean_reward=mean_reward,
            mean_iterations=mean_iterations,
            mean_success_rate=mean_success_rate,
            mean_window_size=self.window,
            mean_window_count=len(recent_rewards),
        )

    def record_loss(self, loss_value):
        self.training_loss.append(float(loss_value))
        if len(self.training_loss) > self._MAX_LOSS_HISTORY:
            self.training_loss = self.training_loss[-self._MAX_LOSS_HISTORY:]

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
            name="episode_success",
            value=episode_stats.success,
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
        self.emit(
            name="mean_success_rate",
            value=episode_stats.mean_success_rate,
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


class DDQNConsoleReporter:
    def print_progress(self, episode_stats):
        print(episode_stats.progress_line, flush=True)

    def print_eval(self, eval_stats, progress_line, stage_name=""):
        stage_text = f" | stage={stage_name}" if stage_name else ""
        print(
            f"[Eval] Episode {eval_stats.episode}{stage_text} | "
            f"avg_score={eval_stats.avg_score:.2f} | "
            f"avg_iter={eval_stats.avg_iterations:.2f}",
            flush=True,
        )
        print(flush=True)

    def print_checkpoint(self, episode_count):
        print(
            f"\n[DDQN] 已保存周期 checkpoint: episode {episode_count}",
            flush=True,
        )

    def print_finished(self, solved, episode_count):
        if solved:
            print(f"\nEnvironment solved in {episode_count} episodes.")
        else:
            print("\nEpisode limit reached.")


class DDQNWorkerStatus:
    def __init__(self, worker_count: int):
        self.active_workers = set(range(worker_count))
        self.dead_workers = {}

    @property
    def has_active_workers(self) -> bool:
        return bool(self.active_workers)

    def check_processes(self, processes):
        for worker_id, process in enumerate(processes):
            if (
                worker_id in self.active_workers
                and not process.is_alive()
                and process.exitcode not in (None, 0)
            ):
                self.active_workers.discard(worker_id)
                self.dead_workers[worker_id] = (
                    f"进程异常退出，exitcode={process.exitcode}"
                )
                print(
                    f"\n[DDQN][Worker {worker_id}] 进程异常退出，已从训练中移除",
                    flush=True,
                )

    def handle_error(self, message):
        worker_id = message["worker_id"]
        self.active_workers.discard(worker_id)
        self.dead_workers[worker_id] = message["message"]
        print(
            f"\n[DDQN][Worker {worker_id}] 失败，已从训练中移除: {message['message']} "
            f"(pid={message['pid']}, port={message['port']})",
            flush=True,
        )
        self.raise_if_all_dead()

    def handle_warning(self, message):
        print(
            f"\n[DDQN][Worker {message['worker_id']}] {message['message']} "
            f"(pid={message['pid']}, port={message['port']})",
            flush=True,
        )

    def raise_if_all_dead(self):
        if self.active_workers or not self.dead_workers:
            return
        raise RuntimeError("所有 DDQN worker 都已失效: " + self._failure_summary())

    def _failure_summary(self) -> str:
        return "; ".join(
            f"worker {worker_id}: {reason}"
            for worker_id, reason in sorted(self.dead_workers.items())
        )
