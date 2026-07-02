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
        env_spec=env_spec,
        scenario_spec=scenario_spec,
    )
    return DDQNEnvAdapter(env, env_spec=env_spec, scenario_spec=scenario_spec)


def _print_network_summary(network, use_paper, hidden_sizes, device):
    """Print PyTorch network structure and parameter count."""
    n_outputs = network.n_outputs
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    is_cnn = hasattr(network, '_n_grid_channels')
    is_factored = getattr(network, '_use_factored', False)

    print(f"\n{'='*60}")
    print(f"  DDQN Network Summary{' (CNN)' if is_cnn else ''}{' (Factored)' if is_factored else ''}")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  Observation:   596 dim {'(paper format)' if use_paper else ''}")
    print(f"  Actions:       {n_outputs}")
    if is_factored:
        print(f"  Action heads:  wait(1) + pos(45) + card(10) = 56 → outer-sum → 451")
    if is_cnn:
        print(f"  Architecture:  3x3-CNN + 1x9-CNN | global-MLP -> {'factored heads' if is_factored else 'head'}")
    else:
        hidden_str = " -> ".join(str(h) for h in (hidden_sizes or [256, 128]))
        print(f"  Hidden layers: {hidden_str}")
        print(f"  Activation:    LeakyReLU")
        print(f"  Architecture:  596 -> {hidden_str} -> {n_outputs}")
    print(f"{'='*60}")
    print(f"  Total params:  {total_params:,}")
    print(f"  Trainable:     {trainable_params:,}")
    print(f"{'='*60}")

    # Per-layer details
    print(f"\n  Layer details:")
    print(f"  {'Layer':<25} {'Shape':<30} {'Params':>12}")
    print(f"  {'-'*67}")
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight
            bias = module.bias.numel() if module.bias is not None else 0
            print(f"  {name:<25} [{'x'.join(str(d) for d in w.shape)}]{' +b' if bias else '':<20} {w.numel() + bias:>12,}")
        elif isinstance(module, torch.nn.Conv2d):
            w = module.weight
            bias = module.bias.numel() if module.bias is not None else 0
            print(f"  {name:<25} {'Conv'+str(tuple(w.shape)):<30} {w.numel() + bias:>12,}")
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.Dropout, torch.nn.LeakyReLU)):
            pass  # skip activation/norm layers
    print()


def _print_network_summary(network, use_paper, hidden_sizes, device):
    """Print PyTorch network structure and parameter count."""
    n_outputs = network.n_outputs
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    is_cnn = hasattr(network, '_n_grid_channels')
    is_factored = getattr(network, '_use_factored', False)

    print(f"\n{'='*60}")
    print(f"  DDQN Network Summary{' (CNN)' if is_cnn else ''}{' (Factored)' if is_factored else ''}")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  Observation:   596 dim {'(paper format)' if use_paper else ''}")
    print(f"  Actions:       {n_outputs}")
    if is_factored:
        print(f"  Action heads:  wait(1) + pos(45) + card(10) = 56 → outer-sum → 451")
    if is_cnn:
        print(f"  Architecture:  3x3-CNN + 1x9-CNN | global-MLP -> {'factored heads' if is_factored else 'head'}")
    else:
        hidden_str = " -> ".join(str(h) for h in (hidden_sizes or [256, 128]))
        print(f"  Hidden layers: {hidden_str}")
        print(f"  Activation:    LeakyReLU")
        print(f"  Architecture:  596 -> {hidden_str} -> {n_outputs}")
    print(f"{'='*60}")
    print(f"  Total params:  {total_params:,}")
    print(f"  Trainable:     {trainable_params:,}")
    print(f"{'='*60}")

    # Per-layer details
    print(f"\n  Layer details:")
    print(f"  {'Layer':<25} {'Shape':<30} {'Params':>12}")
    print(f"  {'-'*67}")
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight
            bias = module.bias.numel() if module.bias is not None else 0
            print(f"  {name:<25} [{'x'.join(str(d) for d in w.shape)}]{' +b' if bias else '':<20} {w.numel() + bias:>12,}")
        elif isinstance(module, torch.nn.Conv2d):
            w = module.weight
            bias = module.bias.numel() if module.bias is not None else 0
            print(f"  {name:<25} {'Conv'+str(tuple(w.shape)):<30} {w.numel() + bias:>12,}")
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.Dropout, torch.nn.LeakyReLU)):
            pass  # skip activation/norm layers
    print()


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
        return [
            f"Batch: {self.args.ddqn_batch_size} | Burn-in: {self.args.ddqn_burn_in}",
            f"LR: {self.args.ddqn_lr} | Gamma: {self.args.ddqn_gamma}",
            f"Update: {self.args.ddqn_update_freq} | Sync: {self.args.ddqn_sync_freq}",
            f"Hidden: [{hidden_str}] | Obs: typed_onehot",
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
            env_spec=env_spec,
            scenario_spec=scenario_spec,
        )
        return DDQNEnvAdapter(env, env_spec=env_spec, scenario_spec=scenario_spec)

    def train(self, context) -> None:
        from .adapter import DDQNSpaceSpec, typed_onehot_state_dim
        from .async_trainer import AsyncDDQNTrainer
        from .ddqn import QNetwork

        require_execution(context.execution, "async_worker_pool", "DDQN")
        if context.game_instances is None:
            raise ValueError("DDQN 训练需要 TrainContext 提供 game_instances")

        hidden_sizes = _parse_hidden_sizes(
            getattr(context.args, "ddqn_hidden_sizes", None))

        # Build a space-spec for QNetwork construction (no game process needed)
        if context.env_spec is not None:
            env = DDQNSpaceSpec(
                context.env_spec, scenario_spec=context.scenario_spec,
            )
        else:
            env = self._build_env(
                instance=context.game_instances[0],
                env_spec=context.env_spec,
                scenario_spec=context.scenario_spec,
            )
        if hasattr(env, "close"):
            context.artifacts.env = env

        n_inputs_override = typed_onehot_state_dim(
            env.rows, env.cols, env.num_cards)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_paper_obs = getattr(context.args, "ddqn_paper_observation", False)
        use_cnn = getattr(context.args, "use_cnn", False)
        if use_cnn:
            from .cnn_network import CNNQNetwork
            use_factored = getattr(context.args, "use_factored", False)
            network = CNNQNetwork(
                env,
                learning_rate=context.args.ddqn_lr,
                device=device,
                use_factored=use_factored,
            )
        else:
            use_cnn = getattr(context.args, "use_cnn", False)
        if use_cnn:
            from .cnn_network import CNNQNetwork
            use_factored = getattr(context.args, "use_factored", False)
            network = CNNQNetwork(
                env,
                learning_rate=context.args.ddqn_lr,
                device=device,
                use_factored=use_factored,
            )
        else:
            network = QNetwork(
                    env,
                    learning_rate=context.args.ddqn_lr,
                    device=device,
                    hidden_sizes=hidden_sizes,
                    n_inputs_override=n_inputs_override,
                )
        context.artifacts.network = network

        load_path = context.checkpoint.resolve_load_path()
        restored_extra = None
        if load_path and os.path.exists(load_path):
            print(f"加载 DDQN 模型: {load_path}")
            from .checkpoint import load_full_state
            state_dict, restored_extra = load_full_state(load_path, device=device)
            network.load_state_dict(state_dict)
            if restored_extra is not None:
                print(
                    "[DDQN] 完整状态恢复: "
                    f"optimizer={'✓' if restored_extra.get('optimizer_state_dict') else '✗'}, "
                    f"buffer={'✓' if restored_extra.get('buffer_data') else '✗'}, "
                    f"ep={restored_extra.get('episode_count', 0)}",
                    flush=True,
                )
            else:
                print("[DDQN] 旧版 checkpoint (仅权重)，optimizer/buffer 从零开始")

            _print_network_summary(network, use_paper_obs, hidden_sizes, device)
            if restored_extra is not None:
                print(
                    "[DDQN] 完整状态恢复: "
                    f"optimizer={'✓' if restored_extra.get('optimizer_state_dict') else '✗'}, "
                    f"buffer={'✓' if restored_extra.get('buffer_data') else '✗'}, "
                    f"ep={restored_extra.get('episode_count', 0)}",
                    flush=True,
                )
            else:
                print("[DDQN] 旧版 checkpoint (仅权重)，optimizer/buffer 从零开始")

        _print_network_summary(network, hidden_sizes, device)

        trainer = AsyncDDQNTrainer(
            context.args,
            context.game_instances,
            network,
            metrics=context.metrics,
            checkpoint=context.checkpoint,
            context=context,
            env_spec=context.env_spec,
            scenario_spec=context.scenario_spec,
            restored_extra=restored_extra,
        )

        trainer.train(
            max_episodes=context.args.ddqn_episodes,
            network_update_frequency=context.args.ddqn_update_freq,
            network_sync_frequency=context.args.ddqn_sync_freq,
        )


def create_algorithm(args):
    return DDQNAlgorithm(args)
