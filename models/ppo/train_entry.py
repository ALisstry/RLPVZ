from training.registry import AlgorithmSpec
from training.execution import require_execution
from utils.train_utils import print_gpu_memory

# 实现
# create_algorithm: 返回当前算法的 Algorithm 实例，供 training.registry 动态创建。
# spec: 声明算法名称、策略类型、支持的执行策略和能力。
# describe_config: 返回启动时需要打印的关键算法配置。
# train: 接收 TrainContext，构建环境、模型、callback，并启动训练。


class PPOAlgorithm:
    spec = AlgorithmSpec(
        name="ppo",
        policy_type="on_policy",
        supported_execution=("sb3_vec_env",),
        supports_curriculum=True,
        supports_action_mask=True,
    )

    def __init__(self, args):
        self.args = args

    def describe_config(self) -> list[str]:
        return [
            f"网络: {self.args.net}",
            f"Batch: {self.args.batch} | Steps: {self.args.n_steps} | LR: {self.args.lr}",
            f"探索: {self.args.start_ent} → {self.args.end_ent}",
        ]

    def train(self, context) -> None:
        from .env import get_env
        from .model import get_model

        require_execution(context.execution, "sb3_vec_env", "PPO")
        print("建环境...")
        load_path = context.checkpoint.resolve_load_path()
        env = get_env(
            context.args,
            context.game_instances,
            context.env_spec,
            context.scenario_spec,
            load_path=load_path,
        )
        context.artifacts.env = env

        print("创建模型...")
        model = get_model(context.args, env, context.device, load_path=load_path)
        context.artifacts.model = model

        print_gpu_memory()

        from .callbacks import build_callbacks

        callbacks = build_callbacks(
            context.args,
            context.run_paths,
            checkpoint=context.checkpoint,
            metrics=context.metrics,
        )
        run_ppo(model, context.args, callbacks)


def create_algorithm(args):
    return PPOAlgorithm(args)


def run_ppo(model, args, callbacks):
    print("开始训练...\r\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=False,
    )
