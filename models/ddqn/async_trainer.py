import queue

from .ddqn import experienceReplayBuffer
from .learner import DDQNLearner
from .metric_emitter import DDQNMetricEmitter
from .reporting import DDQNConsoleReporter
from .training_stats import DDQNTrainingStats
from .worker_pool import DDQNWorkerPool
from .worker_status import DDQNWorkerStatus


class AsyncDDQNTrainer:
    def __init__(
        self,
        args,
        instances,
        network,
        metrics=None,
        checkpoint=None,
        env_spec=None,
        scenario_spec=None,
    ):
        self.args = args
        self.instances = instances
        self.network = network
        self.learner = DDQNLearner(
            network=network,
            batch_size=args.ddqn_batch_size,
            gamma=args.ddqn_gamma,
        )
        self.buffer = experienceReplayBuffer(
            memory_size=args.ddqn_buffer_size, burn_in=args.ddqn_burn_in
        )
        self.batch_size = args.ddqn_batch_size
        self.reward_threshold = 30000
        self.stats = DDQNTrainingStats(window=100)

        self.transition_count = 0
        self.solved = False
        self.worker_status = DDQNWorkerStatus(worker_count=len(instances))
        self.checkpoint_freq = max(0, int(getattr(args, "ddqn_checkpoint_freq", 0)))
        self.metric_emitter = DDQNMetricEmitter(metrics)
        self.reporter = DDQNConsoleReporter()
        self.checkpoint = checkpoint
        self.env_spec = env_spec
        self.scenario_spec = scenario_spec

    def train(
        self,
        max_episodes,
        network_update_frequency,
        network_sync_frequency,
        evaluate_frequency,
        evaluate_n_iter,
    ):
        worker_pool = DDQNWorkerPool(
            args=self.args,
            instances=self.instances,
            batch_size=self.batch_size,
            initial_state_dict=self.learner.state_dict_cpu(),
            env_spec=self.env_spec,
            scenario_spec=self.scenario_spec,
        )
        worker_pool.start()

        try:
            self._run_training_loop(
                worker_pool=worker_pool,
                max_episodes=max_episodes,
                network_update_frequency=network_update_frequency,
                network_sync_frequency=network_sync_frequency,
                evaluate_frequency=evaluate_frequency,
                evaluate_n_iter=evaluate_n_iter,
            )
        finally:
            worker_pool.stop()

    def _run_training_loop(
        self,
        worker_pool,
        max_episodes,
        network_update_frequency,
        network_sync_frequency,
        evaluate_frequency,
        evaluate_n_iter,
    ):
        while self.stats.episode_count < max_episodes and not self.solved:
            self._drain_stats_queue(
                worker_pool, evaluate_frequency, evaluate_n_iter
            )

            try:
                transition = worker_pool.transition_queue.get(timeout=1.0)
            except queue.Empty:
                if not self.worker_status.has_active_workers:
                    raise RuntimeError("所有 DDQN worker 都已退出，训练终止")
                continue

            self.buffer.append(*transition)
            self.transition_count += 1

            if self.buffer.burn_in_capacity() < 1:
                continue

            if self.transition_count % network_update_frequency == 0:
                loss_value = self.learner.update(self.buffer)
                if loss_value is not None:
                    self.stats.record_loss(loss_value)
                    self.metric_emitter.emit_loss(
                        loss_value=loss_value,
                        transition_count=self.transition_count,
                        episode_count=self.stats.episode_count,
                    )

            if self.transition_count % network_sync_frequency == 0:
                self.stats.record_sync()
                worker_pool.publish_weights(self.learner.sync_target())

        worker_pool.request_stop()
        self._drain_stats_queue(worker_pool, evaluate_frequency, evaluate_n_iter)
        self._emit_training_metrics(force=True)
        self.reporter.print_finished(self.solved, self.stats.episode_count)

    def _drain_stats_queue(self, worker_pool, evaluate_frequency, evaluate_n_iter):
        self.worker_status.check_processes(worker_pool.workers)

        while True:
            try:
                message = worker_pool.stats_queue.get_nowait()
            except queue.Empty:
                self.worker_status.raise_if_all_dead()
                return

            if message["type"] == "error":
                self.worker_status.handle_error(message)
                continue

            if message["type"] == "warning":
                self.worker_status.handle_warning(message)
                continue

            episode_stats = self.stats.record_episode(
                message["reward"], message["iterations"]
            )
            self.metric_emitter.emit_episode(
                message, episode_stats, self.transition_count
            )

            if (
                self.checkpoint is not None
                and self.checkpoint_freq
                and self.stats.episode_count % self.checkpoint_freq == 0
            ):
                self.checkpoint.save(
                    network=self.network,
                    tag=f"episode_{self.stats.episode_count}",
                )
                self.reporter.print_checkpoint(self.stats.episode_count)

            self._emit_training_metrics()

            progress_line = episode_stats.progress_line
            self.reporter.print_progress(episode_stats)

            if episode_stats.mean_reward >= self.reward_threshold:
                self.solved = True
                return

            if self.stats.should_evaluate(evaluate_frequency):
                eval_stats = self.stats.record_eval(evaluate_n_iter)
                self.metric_emitter.emit_eval(eval_stats, self.transition_count)
                self.reporter.print_eval(eval_stats, progress_line)

    def _emit_training_metrics(self, force=False):
        self.metric_emitter.emit_snapshot(self.stats.to_snapshot(force=force))
