def print_metadata(args, algorithm=None, run_paths=None):
    print("\r\n" + "=" * 60)
    print("PVZ 训练")
    print("=" * 60)

    actual_game_speed = min(args.speed, 10.0)
    _ = actual_game_speed * args.frameskip

    print(f"\r\n配置:")
    print(f"  算法: {args.algo}")
    print(f"  速度: {actual_game_speed}x | 帧跳过: {args.frameskip}")
    if run_paths is not None:
        print(f"  运行目录: {run_paths.run_dir}")
        print(f"  缓存模型: {run_paths.cached_model_path}")
    if getattr(args, "num_envs", 1) > 1:
        print(f"  并行环境: {args.num_envs} | base_port: {args.base_port}")
    print(f"  执行策略: {getattr(args, 'execution', 'auto')}")
    print(f"  课程学习: {getattr(args, 'curriculum', 'none')}")
    if algorithm is not None and hasattr(algorithm, "describe_config"):
        for line in algorithm.describe_config():
            print(f"  {line}")
