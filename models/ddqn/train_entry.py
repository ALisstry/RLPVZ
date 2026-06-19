import os
import torch

from training.execution import require_execution
from training.registry import AlgorithmSpec


# ── helpers ────────────────────────────────────────────────────────────────

def _parse_hidden_sizes(raw) -> list[int] | None:
    """Parse hidden sizes from YAML list or comma-separated CLI string.

    Handles:
      - YAML list: [2048, 2048]  (already a Python list)
      - CLI string: "2048,2048"
      - None / empty → None (uses default)
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        result = [int(x) for x in raw]
        return result if result else None
    if isinstance(raw, str) and raw.strip():
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [int(p) for p in parts] if parts else None
    return None


def _get_paper_observation(args) -> bool:
    """Determine whether paper-format observation should be used."""
    # 1. Explicit CLI flag
    if hasattr(args, "ddqn_paper_observation"):
        return bool(args.ddqn_paper_observation)
    # 2. Training config YAML
    training_args = getattr(args, "training", {}).get("args", {})
    return bool(training_args.get("ddqn_paper_observation", False))


def _build_ddqn_env(args, instance=None, env_spec=None, scenario_spec=None):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter

    if instance is None:
        raise ValueError("DDQN 环境构建需要显式传入 game instance")

    use_paper = _get_paper_observation(args)
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


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN Algorithm
# ═══════════════════════════════════════════════════════════════════════════════

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
        hidden = _parse_hidden_sizes(
            getattr(self.args, "ddqn_hidden_sizes", None))
        hidden_str = ",".join(str(h) for h in hidden) if hidden else "256,128"
        paper_obs = _get_paper_observation(self.args)
        return [
            f"Batch: {self.args.ddqn_batch_size} | Burn-in: {self.args.ddqn_burn_in}",
            f"LR: {self.args.ddqn_lr} | Gamma: {self.args.ddqn_gamma}",
            f"Update: {self.args.ddqn_update_freq} | Sync: {self.args.ddqn_sync_freq}",
            f"Hidden: [{hidden_str}] | PaperObs: {paper_obs}",
        ]

    def _build_env(self, instance, env_spec=None, scenario_spec=None):
        from envs import PVZEnv
        from .adapter import DDQNEnvAdapter

        use_paper = _get_paper_observation(self.args)
        env = PVZEnv(
            config_path=self.args.training_config,
            hook_port=instance["port"],
            target_pid=instance["pid"],
            game_speed=self.args.speed,
            frame_skip=self.args.frameskip,
            verbose=self.args.env_console_log_level,
            log_verbose=self.args.file_log_level,
        )
        return DDQNEnvAdapter(
            env, env_spec=env_spec, scenario_spec=scenario_spec,
            use_paper_observation=use_paper,
        )

    def train(self, context) -> None:
        from .adapter import DDQNSpaceSpec, paper_state_dim
        from .async_trainer import AsyncDDQNTrainer
        from .ddqn import QNetwork

        require_execution(context.execution, "async_worker_pool", "DDQN")
        if context.game_instances is None:
            raise ValueError("DDQN 训练需要 TrainContext 提供 game_instances")

        use_paper = _get_paper_observation(context.args)
        hidden_sizes = _parse_hidden_sizes(
            getattr(context.args, "ddqn_hidden_sizes", None))

        # Build a space-spec for QNetwork construction (no game process needed)
        if context.env_spec is not None:
            env = DDQNSpaceSpec(
                context.env_spec, scenario_spec=context.scenario_spec,
                use_paper_observation=use_paper,
            )
        else:
            env = self._build_env(
                instance=context.game_instances[0],
                env_spec=context.env_spec,
                scenario_spec=context.scenario_spec,
            )
        if hasattr(env, "close"):
            context.artifacts.env = env

        # Compute n_inputs override for paper observation
        n_inputs_override = None
        if use_paper:
            n_inputs_override = paper_state_dim(
                env.rows, env.cols, env.num_cards)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        network = QNetwork(
            env,
            learning_rate=context.args.ddqn_lr,
            device=device,
            hidden_sizes=hidden_sizes,
            n_inputs_override=n_inputs_override,
        )
        context.artifacts.network = network

        load_path = context.checkpoint.resolve_load_path()
        if load_path and os.path.exists(load_path):
            print(f"加载 DDQN 模型: {load_path}")
            state_dict = torch.load(load_path, map_location=device, weights_only=True)
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
