import os


def _build_ddqn_env(args, instance=None):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter
    import train_utils

    if instance is None:
        instances = getattr(args, "game_instances", None) or train_utils.resolve_game_instances(
            args
        )
        instance = instances[0]

    env = PVZEnv(
        hook_port=instance["port"],
        target_pid=instance["pid"],
        game_speed=args.speed,
        frame_skip=args.frameskip,
        verbose=args.env_console_log_level,
        log_verbose=args.file_log_level,
    )
    return DDQNEnvAdapter(env)


def train_ddqn(args, metrics_callback=None):
    import torch
    import train_utils
    from .async_trainer import AsyncDDQNTrainer
    from .ddqn import QNetwork

    instances = getattr(args, "game_instances", None) or train_utils.resolve_game_instances(
        args
    )
    env = _build_ddqn_env(args, instance=instances[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = QNetwork(env, learning_rate=args.ddqn_lr, device=device)

    if args.ddqn_load_path and os.path.exists(args.ddqn_load_path):
        print(f"加载 DDQN 模型: {args.ddqn_load_path}")
        state_dict = torch.load(args.ddqn_load_path, map_location=device)
        network.load_state_dict(state_dict)

    trainer = AsyncDDQNTrainer(args, instances, network, metrics_callback=metrics_callback)

    try:
        trainer.train(
            max_episodes=args.ddqn_episodes,
            network_update_frequency=args.ddqn_update_freq,
            network_sync_frequency=args.ddqn_sync_freq,
            evaluate_frequency=args.ddqn_eval_freq,
            evaluate_n_iter=args.ddqn_eval_iters,
        )
    finally:
        train_utils.save_runtime_checkpoint(args, network=network)
        env.close()

