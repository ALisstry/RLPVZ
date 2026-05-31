import os

from training.execution import require_execution
from training.registry import AlgorithmSpec


def _build_ddqn_env(args, instance=None, env_spec=None, scenario_spec=None):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter

    if instance is None:
        raise ValueError("DDQN 环境构建需要显式传入 game instance")

    env = PVZEnv(
        config_path=args.training_config,
        hook_port=instance["port"],
        target_pid=instance["pid"],
        game_speed=args.speed,
        frame_skip=args.frameskip,
        verbose=args.env_console_log_level,
        log_verbose=args.file_log_level,
    )
    return DDQNEnvAdapter(env, env_spec=env_spec, scenario_spec=scenario_spec)


def train_ddqn(
    args,
    metrics=None,
    checkpoint=None,
    artifacts=None,
    env_spec=None,
    scenario_spec=None,
    instances=None,
    execution=None,
):
    import torch
    from .async_trainer import AsyncDDQNTrainer
    from .adapter import DDQNSpaceSpec
    from .ddqn import QNetwork

    if instances is None:
        raise ValueError("DDQN 训练需要 TrainContext 提供 game_instances")
    if execution is not None:
        require_execution(execution, "async_worker_pool", "DDQN")
    if env_spec is not None:
        env = DDQNSpaceSpec(env_spec, scenario_spec=scenario_spec)
    else:
        env = _build_ddqn_env(
            args,
            instance=instances[0],
            env_spec=env_spec,
            scenario_spec=scenario_spec,
        )
    if artifacts is not None and hasattr(env, "close"):
        artifacts.env = env

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = QNetwork(env, learning_rate=args.ddqn_lr, device=device)
    if artifacts is not None:
        artifacts.network = network

    load_path = checkpoint.resolve_load_path() if checkpoint is not None else args.ddqn_load_path
    if load_path and os.path.exists(load_path):
        print(f"加载 DDQN 模型: {load_path}")
        state_dict = torch.load(load_path, map_location=device)
        network.load_state_dict(state_dict)

    trainer = AsyncDDQNTrainer(
        args,
        instances,
        network,
        metrics=metrics,
        checkpoint=checkpoint,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
    )

    trainer.train(
        max_episodes=args.ddqn_episodes,
        network_update_frequency=args.ddqn_update_freq,
        network_sync_frequency=args.ddqn_sync_freq,
        evaluate_frequency=args.ddqn_eval_freq,
        evaluate_n_iter=args.ddqn_eval_iters,
    )


class DDQNAlgorithm:
    spec = AlgorithmSpec(
        name="ddqn",
        policy_type="off_policy",
        supported_execution=("async_worker_pool",),
        supports_curriculum=True,
        supports_action_mask=True,
    )

    def __init__(self, args):
        self.args = args

    def describe_config(self) -> list[str]:
        return [
            f"Batch: {self.args.ddqn_batch_size} | Burn-in: {self.args.ddqn_burn_in}",
            f"LR: {self.args.ddqn_lr} | Gamma: {self.args.ddqn_gamma}",
            f"Update: {self.args.ddqn_update_freq} | Sync: {self.args.ddqn_sync_freq}",
        ]

    def train(self, context) -> None:
        train_ddqn(
            context.args,
            metrics=context.metrics,
            checkpoint=context.checkpoint,
            artifacts=context.artifacts,
            env_spec=context.env_spec,
            scenario_spec=context.scenario_spec,
            instances=context.game_instances,
            execution=context.execution,
        )


def create_algorithm(args):
    return DDQNAlgorithm(args)

