import queue

import numpy as np

from .threshold import Threshold
from .ddqn import QNetwork
from training.logging import setup_worker_logging
from training.worker_pool import AsyncWorkerPool


CURRICULUM_EPISODE_ACK = "__curriculum_episode_ack__"

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
        initial_global_episode: int = 0,
    ):
        self.args = args
        self.instances = instances
        self.batch_size = batch_size
        self.initial_state_dict = initial_state_dict
        self.env_spec = env_spec
        self.scenario_spec = scenario_spec
        self.initial_global_episode = initial_global_episode

        super().__init__(instances)
        self.transition_queue = self.make_queue(maxsize=max(2048, self.batch_size * 64))
        self.stats_queue = self.make_queue(maxsize=1024)
        self.weight_queues = self.make_per_worker_queues(maxsize=1)
        self.scenario_queues = self.make_per_worker_queues(maxsize=1)

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
                self.scenario_queues[worker_id],
                self.stop_event,
                self.env_spec,
                self.scenario_spec,
                self.initial_global_episode,
            ),
            label="DDQN",
        )

    def publish_weights(self, state_dict, global_episode: int = 0):
        for weights_queue in self.weight_queues:
            _put_latest_weights(weights_queue, (state_dict, global_episode))

    def publish_scenario(self, scenario_spec):
        for scenario_queue in self.scenario_queues:
            while True:
                try:
                    scenario_queue.get_nowait()
                except queue.Empty:
                    break
            scenario_queue.put(scenario_spec)

    def acknowledge_episode(self, worker_id):
        try:
            self.scenario_queues[worker_id].put_nowait(CURRICULUM_EPISODE_ACK)
        except queue.Full:
            pass


def build_ddqn_env(args, instance, worker_id=None, env_spec=None, scenario_spec=None):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter

    env = PVZEnv(
        config_path=args.training_config,
        hook_port=instance["port"],
        target_pid=instance["pid"],
        game_speed=args.speed,
        frame_skip=args.frameskip,
        verbose=args.env_console_log_level,
        log_verbose=args.file_log_level,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
        worker_id=worker_id,
    )
    return DDQNEnvAdapter(env, env_spec=env_spec, scenario_spec=scenario_spec)


def _build_worker_env(args, instance, worker_id=None, env_spec=None, scenario_spec=None):
    return build_ddqn_env(args, instance, worker_id, env_spec, scenario_spec)


def _drain_latest_weights(weights_queue):
    latest = None
    while True:
        try:
            latest = weights_queue.get_nowait()
        except queue.Empty:
            return latest  # (state_dict, global_episode) or None


def _put_latest_weights(weights_queue, payload):
    while True:
        try:
            weights_queue.get_nowait()
        except queue.Empty:
            break
    weights_queue.put(payload)


def _consume_scenario_queue(
    env,
    scenario_queue,
    wait: bool = False,
    timeout: float = 5.0,
) -> None:
    latest_scenario = None
    if wait:
        try:
            item = scenario_queue.get(timeout=timeout)
        except queue.Empty:
            return
        if item != CURRICULUM_EPISODE_ACK:
            latest_scenario = item

    while True:
        try:
            item = scenario_queue.get_nowait()
        except queue.Empty:
            break
        if item != CURRICULUM_EPISODE_ACK:
            latest_scenario = item

    if latest_scenario is not None:
        env.set_pending_scenario(latest_scenario)


def ddqn_worker_main(
    worker_id,
    args,
    instance,
    initial_state_dict,
    transition_queue,
    stats_queue,
    weights_queue,
    scenario_queue,
    stop_event,
    env_spec,
    scenario_spec,
    initial_global_episode: int = 0,
):
    env = None
    try:
        setup_worker_logging(args)
        env = _build_worker_env(args, instance, worker_id, env_spec, scenario_spec)
        hidden_sizes = _parse_worker_hidden_sizes(args)
        from .adapter import typed_onehot_state_dim
        n_inputs_override = typed_onehot_state_dim(
            env.rows, env.cols, env.num_cards)

        use_cnn = getattr(args, "use_cnn", False)
        if use_cnn:
            from .cnn_network import CNNQNetwork
            use_factored = getattr(args, "use_factored", False)
            network = CNNQNetwork(
                env,
                learning_rate=args.ddqn_lr,
                device="cpu",
                create_optimizer=False,
                use_factored=use_factored,
            )
        else:
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

        # inference_mode: stronger than no_grad, saves memory in forward pass
        import torch as _torch
        _inference_ctx = _torch.inference_mode()
        _inference_ctx.__enter__()

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

        global_episode = initial_global_episode
        while not stop_event.is_set():
            _consume_scenario_queue(env, scenario_queue)
            latest_weights = _drain_latest_weights(weights_queue)
            if latest_weights is not None:
                state_dict, global_episode = latest_weights
                network.load_state_dict(state_dict)
                del state_dict, latest_weights

            mask = np.array(env.mask_available_actions(), dtype=bool)
            epsilon = threshold.epsilon(global_episode)
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
                    latest_weights = _drain_latest_weights(weights_queue)
                    if latest_weights is not None:
                        state_dict, _ = latest_weights
                        network.load_state_dict(state_dict)
                        del state_dict, latest_weights

            state = next_state.copy()
            episode_reward += reward

            if not done:
                continue

            local_episode += 1

            stats_queue.put(
                {
                    "type": "episode",
                    "worker_id": worker_id,
                    "reward": episode_reward,
                    "iterations": env.steps,
                    "win": bool(info.get("win") is True),
                    "epsilon": epsilon,
                    "port": instance["port"],
                    "pid": instance["pid"],
                }
            )
            episode_reward = 0.0
            if getattr(args, "curriculum", "none") != "none":
                # episode 结束后等待主进程处理课程更新，确保新场景赶上下次 reset。
                _consume_scenario_queue(env, scenario_queue, wait=True)
            else:
                _consume_scenario_queue(env, scenario_queue)
            state = _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event)

    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        if not stop_event.is_set():
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
            try:
                env.close()
            except KeyboardInterrupt:
                stop_event.set()
            except Exception:
                pass


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
    return None
