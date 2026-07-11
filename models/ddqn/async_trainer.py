import queue
import os
import time

import numpy as np
import torch

from training.evaluation import EvaluationScheduler
from training.metrics import load_metric_events, load_training_snapshot
from utils.train_utils import get_current_stage_name, load_training_config

from .ddqn import PrioritizedReplayBuffer, UniformReplayBuffer
from .learner import DDQNLearner
from .live_plotter import LivePlotter
from .monitoring import (
    DDQNConsoleReporter,
    DDQNMetricEmitter,
    DDQNTrainingStats,
    DDQNWorkerStatus,
)
from .worker_pool import DDQNWorkerPool


class AsyncDDQNTrainer:
    def __init__(
        self,
        args,
        instances,
        network,
        metrics=None,
        checkpoint=None,
        context=None,
        env_spec=None,
        scenario_spec=None,
        restored_extra=None,  # from load_full_state: {optimizer_state_dict, buffer_data, episode_count, ...}
    ):
        self.args = args
        self.instances = instances
        self.network = network
        self.learner = DDQNLearner(
            network=network,
            batch_size=args.ddqn_batch_size,
            gamma=args.ddqn_gamma,
        )
        # PER hyper-params (can be overridden via CLI / config in the future)
        self._use_per = bool(getattr(args, "ddqn_use_per", False))
        self._per_alpha = float(getattr(args, "ddqn_per_alpha", 0.6))
        self._per_beta_start = float(getattr(args, "ddqn_per_beta", 0.4))
        self._per_epsilon = float(getattr(args, "ddqn_per_epsilon", 1e-6))

        if self._use_per:
            self.buffer = PrioritizedReplayBuffer(
                memory_size=args.ddqn_buffer_size,
                burn_in=args.ddqn_burn_in,
                alpha=self._per_alpha,
                epsilon=self._per_epsilon,
            )
        else:
            self.buffer = UniformReplayBuffer(
                memory_size=args.ddqn_buffer_size,
                burn_in=args.ddqn_burn_in,
            )
        self.batch_size = args.ddqn_batch_size
        self.reward_threshold = 30000
        config = load_training_config(getattr(args, "training_config", None))
        metric_window = max(1, int(config.get("metric_window", 100)))
        snapshot = None
        metric_events = []
        if context is not None and getattr(args, "auto_resume", True):
            snapshot = load_training_snapshot(context.run_paths.metrics_snapshot_path)
            metric_events = load_metric_events(context.run_paths.metrics_csv_path)
        self.stats = DDQNTrainingStats.from_history(
            window=metric_window,
            snapshot=snapshot,
            events=metric_events,
        )

        self.transition_count = max(
            (event.step or 0 for event in metric_events),
            default=0,
        )

        # ── Restore optimizer state, replay buffer & episode count from checkpoint ──
        if restored_extra is not None:
            # PER hyper-params (restore before buffer so they match)
            if self._use_per:
                if restored_extra.get("per_alpha") is not None:
                    self._per_alpha = float(restored_extra["per_alpha"])
                if restored_extra.get("per_beta_start") is not None:
                    self._per_beta_start = float(restored_extra["per_beta_start"])

            if restored_extra.get("optimizer_state_dict"):
                self.learner.network.optimizer.load_state_dict(
                    restored_extra["optimizer_state_dict"]
                )
                print(
                    "[DDQN] optimizer 状态已恢复 (Adam m/v buffers)",
                    flush=True,
                )
            if restored_extra.get("buffer_data"):
                from .checkpoint import _deserialize_buffer
                self.buffer = _deserialize_buffer(
                    restored_extra["buffer_data"], use_per=self._use_per)
                if self.buffer is not None:
                    self.buffer.burn_in = int(
                        getattr(self.buffer, "burn_in", self.args.ddqn_burn_in)
                    )
                    print(
                        f"[DDQN] replay buffer 已恢复: "
                        f"{len(self.buffer)} entries, capacity={self.buffer.memory_size}",
                        flush=True,
                    )
            if restored_extra.get("episode_count", 0) > self.stats.episode_count:
                print(
                    f"[DDQN] episode_count 从 checkpoint 覆盖: "
                    f"{self.stats.episode_count} → {restored_extra['episode_count']}",
                    flush=True,
                )
                self.stats.episode_count = restored_extra["episode_count"]
            if restored_extra.get("transition_count", 0) > self.transition_count:
                self.transition_count = restored_extra["transition_count"]

        if self.stats.episode_count > 0:
            print(
                f"[DDQN] 从 run 记录恢复: episodes={self.stats.episode_count}, "
                f"steps={self.transition_count}",
                flush=True,
            )
        self.solved = False
        self.worker_status = DDQNWorkerStatus(worker_count=len(instances))
        self.checkpoint_freq = max(0, int(getattr(args, "ddqn_checkpoint_freq", 0)))
        self.metric_emitter = DDQNMetricEmitter(metrics)
        self.reporter = DDQNConsoleReporter()
        self.checkpoint = checkpoint
        if context is None and getattr(args, "curriculum", "none") != "none":
            raise ValueError("DDQN curriculum training requires TrainContext")
        self.context = context
        self.env_spec = env_spec
        self.scenario_spec = scenario_spec
        self.eval_scheduler = (
            EvaluationScheduler(context.eval_config)
            if context is not None
            else None
        )
        self._snapshot_freq = max(
            1, int(getattr(args, "ddqn_plot_freq", 20))
        )  # rate-limit snapshot emission
        self._last_snapshot_ep = 0
        self._dashboard_freq = max(
            1, int(getattr(args, "ddqn_plot_freq", 20))
        )
        self.best_eval_checkpoint = None
        self._overfit_test_done = False   # 单步过拟合诊断只跑一次

        # ── Live training dashboard ──
        dashboard_path = (
            context.run_paths.dashboard_path
            if context is not None
            else os.path.join("models_output", "training_dashboard.png")
        )
        max_ep = max(1, int(getattr(args, "ddqn_episodes", 10000)))
        self._live_plotter = LivePlotter(
            save_path=dashboard_path,
            window=100,
            update_freq=self._dashboard_freq,
            max_episodes=max_ep,
        )

    def train(
        self,
        max_episodes,
        network_update_frequency,
        network_sync_frequency,
    ):
        worker_pool = DDQNWorkerPool(
            args=self.args,
            instances=self.instances,
            batch_size=self.batch_size,
            initial_state_dict=self.learner.state_dict_cpu(),
            env_spec=self.env_spec,
            scenario_spec=self.scenario_spec,
            initial_global_episode=self.stats.episode_count,
        )
        worker_pool.start()

        try:
            self._run_training_loop(
                worker_pool=worker_pool,
                max_episodes=max_episodes,
                network_update_frequency=network_update_frequency,
                network_sync_frequency=network_sync_frequency,
            )
        finally:
            worker_pool.stop()

    def _run_training_loop(
        self,
        worker_pool,
        max_episodes,
        network_update_frequency,
        network_sync_frequency,
    ):
        try:
            while self.stats.episode_count < max_episodes and not self.solved:
                self._drain_stats_queue(worker_pool)

                try:
                    transition = worker_pool.transition_queue.get(timeout=1.0)
                except queue.Empty:
                    if not self.worker_status.has_active_workers:
                        raise RuntimeError("所有 DDQN worker 都已退出，训练终止")
                    continue

                # Store contiguous copies to reduce memory fragmentation
                (t_state, t_action, t_reward, t_done,
                 t_next_state, t_mask, t_next_mask) = transition
                self.buffer.append(
                    np.ascontiguousarray(t_state),
                    t_action,
                    t_reward,
                    t_done,
                    np.ascontiguousarray(t_next_state),
                    np.ascontiguousarray(t_mask),
                    np.ascontiguousarray(t_next_mask),
                )
                del t_state, t_next_state, t_mask, t_next_mask, transition
                self.transition_count += 1

                if self.buffer.burn_in_capacity() < 1:
                    continue

                # ── 单步过拟合诊断（burn-in 完成后只跑一次，默认关闭）──
                if not self._overfit_test_done:
                    self._overfit_test_done = True
                    if getattr(self.args, "ddqn_overfit_test", False):
                        self._run_overfit_diagnostic()

                if self.transition_count % network_update_frequency == 0:
                    if self._use_per:
                        beta = self._per_beta_start + (
                            1.0 - self._per_beta_start
                        ) * min(1.0, self.stats.episode_count / max(1, max_episodes))
                    else:
                        beta = 0.4  # ignored by UniformReplayBuffer

                    result = self.learner.update(self.buffer, beta=beta)
                    if result is not None:
                        loss_value, tree_indices, td_errors, q_stats, grad_norm = result
                        self.buffer.update_priorities(tree_indices, td_errors)
                        self.stats.record_loss(loss_value)
                        mean_td = float(np.mean(np.abs(td_errors)))
                        self.stats.record_td_error(mean_td)
                        self.metric_emitter.emit_loss(
                            loss_value=loss_value,
                            transition_count=self.transition_count,
                            episode_count=self.stats.episode_count,
                        )
                        self.metric_emitter.emit_td_error(
                            td_error_mean=mean_td,
                            transition_count=self.transition_count,
                            episode_count=self.stats.episode_count,
                        )
                        self.stats.record_q_stats(
                            mean_q=q_stats["mean_q"],
                            max_q=q_stats["max_q"],
                            entropy=q_stats["entropy"],
                            advantage=q_stats["advantage"],
                            grad_norm=grad_norm,
                            q_wait=q_stats.get("q_wait", 0.0),
                            delta_mean=q_stats.get("delta_mean", 0.0),
                            delta_max=q_stats.get("delta_max", 0.0),
                        )
                        self.metric_emitter.emit_q_stats(
                            mean_q=q_stats["mean_q"],
                            max_q=q_stats["max_q"],
                            entropy=q_stats["entropy"],
                            advantage=q_stats["advantage"],
                            grad_norm=grad_norm,
                            transition_count=self.transition_count,
                            episode_count=self.stats.episode_count,
                            q_wait=q_stats.get("q_wait", 0.0),
                            delta_mean=q_stats.get("delta_mean", 0.0),
                            delta_max=q_stats.get("delta_max", 0.0),
                        )

                if self.transition_count % network_sync_frequency == 0:
                    self.stats.record_sync()
                    worker_pool.publish_weights(
                        self.learner.sync_target(),
                        global_episode=self.stats.episode_count,
                    )
        finally:
            # Drain remaining episode messages BEFORE telling workers to stop,
            # so in-flight episodes get their stats recorded.
            self._drain_stats_queue(worker_pool)
            worker_pool.request_stop()
            self._drain_stats_queue(worker_pool)
            # Always save the final plot, even if training was interrupted.
            self._emit_training_metrics(force=True)
            self._live_plotter.keep_open()
            self.reporter.print_finished(self.solved, self.stats.episode_count)

    def _drain_stats_queue(self, worker_pool):
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
                message["reward"],
                message["iterations"],
                bool(message.get("win") is True),
                message.get("epsilon", 1.0),
            )

            # Keep live dashboard in sync with latest epsilon
            self._live_plotter.set_epsilon(message.get("epsilon", 1.0))
            self._live_plotter.update(self.stats, self.stats.episode_count)

            self.metric_emitter.emit_episode(
                message, episode_stats, self.transition_count
            )
            self._update_curriculum(worker_pool, message, episode_stats)

            if (
                self.checkpoint is not None
                and self.checkpoint_freq
                and self.stats.episode_count % self.checkpoint_freq == 0
            ):
                self.checkpoint.save(
                    network=self.network,
                    tag=f"episode_{self.stats.episode_count}",
                    extra={
                        "optimizer_state_dict": self.learner.network.optimizer.state_dict(),
                        "buffer": self.buffer,
                        "episode_count": self.stats.episode_count,
                        "transition_count": self.transition_count,
                        "per_alpha": self._per_alpha,
                        "per_beta_start": self._per_beta_start,
                    },
                )
                self.reporter.print_checkpoint(self.stats.episode_count)

            self._emit_training_metrics()

            # 每隔 1000 episode 自动保存指标快照副本
            if (self.stats.episode_count > 0
                    and self.stats.episode_count % 1000 == 0
                    and self.context is not None):
                try:
                    import shutil
                    self._emit_training_metrics(force=True)
                    src = self.context.run_paths.metrics_snapshot_path
                    dst = src.replace(
                        ".json", f"_ep{self.stats.episode_count:06d}.json"
                    )
                    shutil.copy2(src, dst)
                except Exception:
                    pass

            progress_line = episode_stats.progress_line
            self.reporter.print_progress(episode_stats, message["worker_id"])

            if episode_stats.mean_reward >= self.reward_threshold:
                self.solved = True
                return

            if self.stats.episode_count % 100 == 0:
                import gc, os

                gc.collect()
                try:
                    import torch, psutil

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    proc = psutil.Process(os.getpid())
                    mem_mb = proc.memory_info().rss / 1024 / 1024
                    buf_pct = len(self.buffer) / max(1, self.buffer.memory_size) * 100
                    qsize = worker_pool.transition_queue.qsize()
                    print(f"\n[MEM] main PID={os.getpid()} RSS={mem_mb:.0f}MB  "
                          f"buffer={buf_pct:.0f}%  queue={qsize}  "
                          f"ep={self.stats.episode_count}  "
                          f"loss_cap={len(self.stats.training_loss)}",
                          flush=True)
                except Exception:
                    pass

            if (
                self.eval_scheduler is not None
                and self.eval_scheduler.should_run(self.stats.episode_count)
            ):
                self._run_worker_eval(worker_pool)
                return  # 重新进入主循环，清空队列后恢复训练

    def _run_overfit_diagnostic(self):
        """单步过拟合诊断：取 buffer 中 1 条 transition，反复训练 100 次。

        验证目标：
          - loss 下降至接近 0 → 网络 + 损失函数实现正确
          - loss 不降 → 网络架构、梯度流或损失计算存在 bug
        """
        print("\n" + "=" * 60, flush=True)
        print("  单步过拟合诊断 (Single-Transition Overfit Test)", flush=True)
        print("=" * 60, flush=True)

        # 保存当前网络权重，测试完恢复
        state_backup = {
            k: v.detach().cpu().clone()
            for k, v in self.network.state_dict().items()
        }

        # 取 1 条 transition（beta=1.0 关闭 IS 修正）
        batch, tree_indices, is_weights = self.buffer.sample_batch(
            batch_size=1, beta=1.0,
        )
        states, actions, rewards, dones, next_states, masks, next_masks = [
            item for item in batch
        ]
        action = actions[0]
        reward = rewards[0]
        done = dones[0]
        print(
            f"  采样 transition: action={action}, reward={reward:.2f}, "
            f"done={done}",
            flush=True,
        )

        # 诊断 1: 检查 state / next_state 是否完全相同
        s0 = np.asarray(states[0])
        s1 = np.asarray(next_states[0])
        state_diff = np.abs(s0 - s1).sum()
        state_same = state_diff < 1e-6
        print(
            f"  |state - next_state|_1 = {state_diff:.2f}"
            f"{'  (IDENTICAL!)' if state_same else ''}",
            flush=True,
        )

        # 诊断 2: 过拟合训练
        losses = self.learner.overfit_test(
            batch, is_weights=None, n_iterations=100,
            log_prefix="  [Overfit]",
        )

        # 恢复权重
        self.learner.network.load_state_dict(state_backup)
        # 同步 target network 也恢复
        self.learner.target_network.load_state_dict(state_backup)

        # 诊断 3: 检查 target vs online Q 值差异
        with torch.no_grad():
            online_q = self.learner.network.get_qvals(states)
            target_q = self.learner.target_network.get_qvals(next_states)
            online_max = float(online_q.max(dim=-1).values.cpu().item())
            target_max = float(target_q.max(dim=-1).values.cpu().item())
        print(
            f"  初始 online max Q = {online_max:.6f} | "
            f"target max Q = {target_max:.6f}",
            flush=True,
        )
        print("=" * 60 + "\n", flush=True)

    def _emit_training_metrics(self, force=False):
        ep = self.stats.episode_count
        if not force and ep - self._last_snapshot_ep < self._snapshot_freq:
            return
        self._last_snapshot_ep = ep
        self.metric_emitter.emit_snapshot(self.stats.to_snapshot(force=force))

    def _run_worker_eval(self, worker_pool):
        """使用训练 Worker 进程运行评估回合（eps=0），数据不入训练 buffer。"""
        eval_episodes = self.eval_scheduler.config.episodes

        # 设置 eval_mode + 计数器，Worker 在下次循环迭代时自动进入评估模式
        worker_pool.eval_slots.value = eval_episodes
        worker_pool.eval_done.value = eval_episodes
        worker_pool.eval_mode.set()
        print("[Eval] 开始评估，等待 Worker 收集结果...", flush=True)

        # 收集 eval 结果
        eval_rewards = []
        eval_wins = []
        eval_iterations = []
        start_time = time.perf_counter()
        while worker_pool.eval_done.value > 0:
            try:
                message = worker_pool.stats_queue.get(timeout=120.0)
            except queue.Empty:
                print("[Eval] 等待 eval episode 超时", flush=True)
                break

            if message["type"] == "eval_episode":
                eval_rewards.append(message["reward"])
                eval_wins.append(message.get("win", False))
                eval_iterations.append(message.get("iterations", 0))
                collected = len(eval_rewards)
                win_mark = "✓" if message.get("win") else " "
                print(
                    f"[Eval {collected}/{eval_episodes}]  "
                    f"reward={message['reward']:.0f}  "
                    f"iter={message.get('iterations', 0)}  "
                    f"win={win_mark}  "
                    f"worker={message.get('worker_id', '?')}",
                    flush=True,
                )
            elif message["type"] == "error":
                self.worker_status.handle_error(message)
            elif message["type"] == "warning":
                self.worker_status.handle_warning(message)
            # 忽略其他类型消息（训练 episode 等）

        # 关闭 eval_mode，Worker 自动切回训练
        worker_pool.eval_mode.clear()
        duration = time.perf_counter() - start_time

        # 记录评估结果
        if eval_rewards:
            mean_reward = sum(eval_rewards) / len(eval_rewards)
            win_rate = sum(1 for w in eval_wins if w) / len(eval_wins)
            mean_iterations = sum(eval_iterations) / len(eval_iterations)
            self.stats.record_eval_result(
                episode=self.stats.episode_count,
                mean_reward=mean_reward,
                win_rate=win_rate,
            )
            stage_name = (
                get_current_stage_name(self.context.curriculum)
                if self.context else ""
            )
            stage_text = f" | stage={stage_name}" if stage_name else ""
            print(
                f"[Eval] Episode {self.stats.episode_count}{stage_text} | "
                f"episodes={len(eval_rewards)} | "
                f"reward={mean_reward:.2f} | "
                f"survival={mean_iterations:.1f} | "
                f"win_rate={win_rate:.2%} | "
                f"duration={duration:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[Eval] Episode {self.stats.episode_count} | "
                "未收集到任何 eval 结果",
                flush=True,
            )

        print("[Eval] 评估结束，恢复训练", flush=True)

    def _update_curriculum(self, worker_pool, message, episode_stats):
        if self.context is None:
            return
        changed, scenario = self.context.update_curriculum(
            {
                "episode_reward": float(message["reward"]),
                "episode_success": bool(message.get("win") is True),
                "episode_count": episode_stats.episode,
                "step": self.transition_count,
            }
        )
        if changed:
            worker_pool.publish_scenario(scenario)
        else:
            worker_pool.acknowledge_episode(int(message["worker_id"]))

