import os
import torch

from training.execution import require_execution
from training.registry import AlgorithmSpec

# 实现
# create_algorithm(): 返回当前算法的 Algorithm 实例。供 training.registry 动态创建。
# spec: 声明算法名称、on/off-policy、是否支持课程学习、多进程等。
# describe_config: 返回启动时需要打印的关键算法配置。
# train: 接收 TrainContext，构建环境规格、模型、trainer，并启动训练。


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

    def _build_env(self, instance, env_spec=None, scenario_spec=None):
        from envs import PVZEnv
        from .adapter import DDQNEnvAdapter

        env = PVZEnv(
            config_path=self.args.training_config,
            hook_port=instance["port"],
            target_pid=instance["pid"],
            game_speed=self.args.speed,
            frame_skip=self.args.frameskip,
            verbose=self.args.env_console_log_level,
            log_verbose=self.args.file_log_level,
        )
        return DDQNEnvAdapter(env, env_spec=env_spec, scenario_spec=scenario_spec)

    def train(self, context) -> None:
        from .adapter import DDQNSpaceSpec
        from .async_trainer import AsyncDDQNTrainer
        from .ddqn import QNetwork

        require_execution(context.execution, "async_worker_pool", "DDQN")
        if context.game_instances is None:
            raise ValueError("DDQN 训练需要 TrainContext 提供 game_instances")

        if context.env_spec is not None:
            env = DDQNSpaceSpec(context.env_spec, scenario_spec=context.scenario_spec)
        else:
            env = self._build_env(
                instance=context.game_instances[0],
                env_spec=context.env_spec,
                scenario_spec=context.scenario_spec,
            )
        if hasattr(env, "close"):
            context.artifacts.env = env

        device = "cuda" if torch.cuda.is_available() else "cpu"
        network = QNetwork(env, learning_rate=context.args.ddqn_lr, device=device)
        context.artifacts.network = network

        load_path = context.checkpoint.resolve_load_path()
        if load_path and os.path.exists(load_path):
            print(f"加载 DDQN 模型: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            network.load_state_dict(state_dict)

        trainer = AsyncDDQNTrainer(
            context.args,
            context.game_instances,
            network,
            metrics=context.metrics,
            checkpoint=context.checkpoint,
            env_spec=context.env_spec,
            scenario_spec=context.scenario_spec,
        )

        trainer.train(
            max_episodes=context.args.ddqn_episodes,
            network_update_frequency=context.args.ddqn_update_freq,
            network_sync_frequency=context.args.ddqn_sync_freq,
            evaluate_frequency=context.args.ddqn_eval_freq,
            evaluate_n_iter=context.args.ddqn_eval_iters,
        )


def create_algorithm(args):
    return DDQNAlgorithm(args)
