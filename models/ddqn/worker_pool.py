import queue

import numpy as np

from .threshold import Threshold
from .ddqn import QNetwork, copy_state_dict_to_cpu
from training.logging import setup_worker_logging
from training.worker_pool import AsyncWorkerPool


def _parse_worker_hidden_sizes(args) -> list[int] | None:
    """Parse hidden sizes from args for worker-side QNetwork construction.

    Handles YAML list or CLI comma-separated string.
    """
    raw = getattr(args, "ddqn_hidden_sizes", None)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        result = [int(x) for x in raw]
        return result if result else None
    if isinstance(raw, str) and raw.strip():
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [int(p) for p in parts] if parts else None
    return None


class DDQNWorkerPool(AsyncWorkerPool):
    def __init__(
        self,
        args,
        instances,
        batch_size,
        initial_state_dict,
        env_spec=None,
        scenario_spec=None,
    ):
        self.args = args
        self.instances = instances
        self.batch_size = batch_size
        self.initial_state_dict = initial_state_dict
        self.env_spec = env_spec
        self.scenario_spec = scenario_spec

        super().__init__(instances)
        self.transition_queue = self.make_queue(maxsize=max(2048, self.batch_size * 64))
        self.stats_queue = self.make_queue(maxsize=1024)
        self.weight_queues = self.make_per_worker_queues(maxsize=1)

    def start(self):
        self.start_workers(
            target=ddqn_worker_main,
            build_args=lambda worker_id, instance: (
                worker_id,
                self.args,
                instance,
                self.initial_state_dict,
                self.transition_queue,
                self.stats_queue,
                self.weight_queues[worker_id],
                self.stop_event,
                self.env_spec,
                self.scenario_spec,
            ),
            label="DDQN",
        )

    def publish_weights(self, state_dict):
        for weights_queue in self.weight_queues:
            _put_latest_weights(weights_queue, state_dict)


def _build_worker_env(args, instance, env_spec=None, scenario_spec=None):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter

    use_paper = bool(getattr(args, "ddqn_paper_observation", False))
    env = PVZEnv(
        config_path=args.training_config,
        hook_port=instance["port"],
        target_pid=instance["pid"],
        game_speed=args.speed,
        frame_skip=args.frameskip,
        verbose=args.env_console_log_level,
        log_verbose=args.file_log_level,
    )
    return DDQNEnvAdapter(
        env, env_spec=env_spec, scenario_spec=scenario_spec,
        use_paper_observation=use_paper,
    )


def _drain_latest_weights(weights_queue):
    latest = None
    while True:
        try:
            latest = weights_queue.get_nowait()
        except queue.Empty:
            return latest


def _put_latest_weights(weights_queue, state_dict):
    while True:
        try:
            weights_queue.get_nowait()
        except queue.Empty:
            break
    weights_queue.put(copy_state_dict_to_cpu(state_dict))


def ddqn_worker_main(
    worker_id,
    args,
    instance,
    initial_state_dict,
    transition_queue,
    stats_queue,
    weights_queue,
    stop_event,
    env_spec,
    scenario_spec,
):
    env = None
    try:
        setup_worker_logging(args)
        # Enable tracemalloc for leak diagnosis
        import tracemalloc as _tm
        _tm.start()
        env = _build_worker_env(args, instance, env_spec, scenario_spec)
        use_paper = bool(getattr(args, "ddqn_paper_observation", False))
        hidden_sizes = _parse_worker_hidden_sizes(args)
        n_inputs_override = None
        if use_paper:
            from .adapter import paper_state_dim
            n_inputs_override = paper_state_dim(env.rows, env.cols, env.num_cards)

        network = QNetwork(
            env,
            learning_rate=args.ddqn_lr,
            device="cpu",
            hidden_sizes=hidden_sizes,
            n_inputs_override=n_inputs_override,
            create_optimizer=False,
        )
        network.load_state_dict(initial_state_dict)
        network.eval()

        # Match epsilon decay span to the configured episode count
        epsilon_seq_length = max(1, int(getattr(args, "ddqn_episodes", 10000)))
        threshold = Threshold(
            seq_length=epsilon_seq_length,
            start_epsilon=1.0,
            interpolation="exponential",
            end_epsilon=0.05,
        )

        state = _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event)
        episode_reward = 0.0
        local_episode = 0

        while not stop_event.is_set():
            latest_state_dict = _drain_latest_weights(weights_queue)
            if latest_state_dict is not None:
                network.load_state_dict(latest_state_dict)
                del latest_state_dict

            mask = np.array(env.mask_available_actions(), dtype=bool)
            epsilon = threshold.epsilon(local_episode)
            action = network.decide_action(state, mask, epsilon=epsilon)

            try:
                next_state, reward, done, info = env.step(action)
            except Exception as exc:
                stats_queue.put(
                    {
                        "type": "warning",
                        "worker_id": worker_id,
                        "port": instance["port"],
                        "pid": instance["pid"],
                        "message": f"step 失败，准备重置: {repr(exc)}",
                    }
                )
                state = _reset_env_with_retry(
                    env, worker_id, instance, stats_queue, stop_event
                )
                episode_reward = 0.0
                continue
            next_mask = np.array(env.mask_available_actions(), dtype=bool)

            while not stop_event.is_set():
                try:
                    transition_queue.put(
                        (
                            state,
                            action,
                            reward,
                            done,
                            next_state,
                            mask,
                            next_mask,
                        ),
                        timeout=1.0,
                    )
                    break
                except queue.Full:
                    latest_state_dict = _drain_latest_weights(weights_queue)
                    if latest_state_dict is not None:
                        network.load_state_dict(latest_state_dict)
                        del latest_state_dict

            state = next_state.copy()
            episode_reward += reward

            if not done:
                continue

            local_episode += 1
            # Periodic memory diagnostic in worker
            if local_episode % 50 == 0 or local_episode == 1:
                try:
                    import os as _os, gc as _gc, tracemalloc as _tm
                    import psutil as _psutil, logging as _logging
                    _gc.collect()
                    proc = _psutil.Process(_os.getpid())
                    mem_mb = proc.memory_info().rss / 1024 / 1024
                    _logger = _logging.getLogger("ddqn_worker")
                    if _tm.is_tracing():
                        snap = _tm.take_snapshot()
                        top = snap.statistics("lineno")[:6]
                        lines = [f"  {s.count:5d} x {s.size/1024:.0f}KB {s}" for s in top]
                        _logger.warning(
                            f"worker{worker_id} PID={_os.getpid()} "
                            f"RSS={mem_mb:.0f}MB ep={local_episode}\n"
                            + "\n".join(lines))
                    else:
                        _logger.warning(
                            f"worker{worker_id} PID={_os.getpid()} "
                            f"RSS={mem_mb:.0f}MB ep={local_episode}")
                except Exception:
                    pass

            stats_queue.put(
                {
                    "type": "episode",
                    "worker_id": worker_id,
                    "reward": episode_reward,
                    "iterations": env.steps,
                    "epsilon": epsilon,
                    "port": instance["port"],
                    "pid": instance["pid"],
                }
            )
            episode_reward = 0.0
            state = _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event)

    except Exception as exc:
        stats_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "port": instance["port"],
                "pid": instance["pid"],
                "message": repr(exc),
            }
        )
    finally:
        if env is not None:
            env.close()


def _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event):
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            return env.reset()
        except Exception as exc:
            consecutive_failures += 1
            stats_queue.put(
                {
                    "type": "warning",
                    "worker_id": worker_id,
                    "port": instance["port"],
                    "pid": instance["pid"],
                    "message": (
                        f"reset 失败，第 {consecutive_failures} 次重试: {repr(exc)}"
                    ),
                }
            )
            if consecutive_failures >= 5:
                raise RuntimeError(
                    f"reset 连续失败 {consecutive_failures} 次: {repr(exc)}"
                ) from exc
            import time

            time.sleep(min(2.0 * consecutive_failures, 5.0))
    raise RuntimeError("worker 停止前未能完成 reset")
