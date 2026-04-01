import argparse
from train_config import cfg, DEFAULT_GAME_PATH, MODEL_PATH, LOAD_PATH


def get_args():
    """
    统一封装训练参数：通用 + PPO + DDQN。
    """
    parser = argparse.ArgumentParser(description="PVZ 训练")

    # 通用参数
    common = parser.add_argument_group("common")
    common.add_argument(
        "--algo",
        type=str,
        default="ddqn",
        choices=["ppo", "ddqn"],
        help="训练算法",
    )
    common.add_argument("--port", "-p", type=int, default=12345, help="Hook 端口")
    common.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="并行环境数量，对应多个 PVZ 进程",
    )
    common.add_argument(
        "--base_port",
        type=int,
        default=12345,
        help="多实例自动分配时的起始 Hook 端口",
    )
    common.add_argument(
        "--ports",
        type=str,
        default="",
        help="显式指定多实例端口，逗号分隔，如 12345,12346,12347",
    )
    common.add_argument(
        "--pids",
        type=str,
        default="",
        help="显式指定目标 PVZ 进程 PID，逗号分隔",
    )
    common.add_argument(
        "--auto_start",
        action="store_true",
        default=True,
        help="自动启动游戏和注入DLL (默认启用)",
    )
    common.add_argument(
        "--no_auto_start", action="store_true", help="禁用自动启动 (手动启动游戏和注入)"
    )
    common.add_argument(
        "--game_path",
        type=str,
        default=DEFAULT_GAME_PATH,
        help=f"游戏路径 (默认: {DEFAULT_GAME_PATH})",
    )
    common.add_argument(
        "--wait_time", type=float, default=3.0, help="游戏启动等待时间 (秒)"
    )
    common.add_argument(
        "--env_console_log_level",
        "--env_verbose",
        dest="env_console_log_level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="环境控制台日志级别: 0=静默, 1=关键信息, 2=详细调试",
    )
    common.add_argument(
        "--file_log_level",
        "--log_verbose",
        dest="file_log_level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="文件日志级别: 0=静默, 1=关键信息, 2=详细调试",
    )

    # PPO 参数
    ppo = parser.add_argument_group("ppo")
    add_ppo_args(ppo)

    # DDQN 参数
    ddqn = parser.add_argument_group("ddqn")
    add_ddqn_args(ddqn)

    return parser.parse_args()


def add_ppo_args(ppo):
    ppo.add_argument("--timesteps", "-t", type=int, default=500000, help="训练步数")
    ppo.add_argument(
        "--speed",
        "-s",
        type=float,
        default=cfg.game_speed,
        help=f"游戏速度 (默认: {cfg.game_speed}x, 最高10x)",
    )
    ppo.add_argument(
        "--frameskip", "-f", type=int, default=4, help="帧跳过 (4=适中，视野更远)"
    )
    ppo.add_argument(
        "--batch", "-b", type=int, default=1024, help="Batch size (GPU空闲，增大Batch)"
    )
    ppo.add_argument(
        "--n_steps",
        "-n",
        type=int,
        default=4096,
        help="N steps (针对 RTX 3050 优化，约 80 秒/更新)",
    )
    ppo.add_argument(
        "--n_epochs", type=int, default=20, help="训练轮数 (数据珍贵，多练几轮)"
    )
    ppo.add_argument(
        "--net",
        type=str,
        default="large",
        choices=["small", "medium", "large", "xlarge", "huge"],
        help="网络大小",
    )
    ppo.add_argument("--lr", type=float, default=3e-4, help="学习率")

    # PPO 优化
    ppo.add_argument(
        "--start_ent", type=float, default=0.15, help="初始探索系数 (0.15=较少随机)"
    )
    ppo.add_argument("--end_ent", type=float, default=0.01, help="最终探索系数")
    ppo.add_argument(
        "--ent_decay",
        type=str,
        default="linear",
        choices=["linear", "exponential", "cosine"],
        help="探索衰减方式",
    )
    ppo.add_argument("--diversify", type=float, default=0.0, help="多样化概率 (0-1)")
    ppo.add_argument(
        "--no_diversify", action="store_true", default=True, help="禁用多样化"
    )
    ppo.add_argument(
        "--no_failure_priority", action="store_true", help="禁用失败优先学习"
    )

    # PPO 加载与保存
    ppo.add_argument(
        "--load",
        type=str,
        default=LOAD_PATH,
        help=f"加载已有模型继续训练 (默认: {LOAD_PATH})",
    )
    ppo.add_argument(
        "--save_path",
        type=str,
        default=MODEL_PATH,
        help=f"模型保存路径 (默认: {MODEL_PATH})",
    )
    ppo.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="自动加载最新模型继续训练 (默认启用)",
    )
    ppo.add_argument(
        "--no_auto_resume", action="store_true", help="禁用自动加载，从零开始训练"
    )
    ppo.add_argument("--save_freq", type=int, default=10000, help="自动保存频率 (步)")

    # PPO 特征抽取器
    ppo.add_argument(
        "--no_attn", action="store_true", help="禁用注意力特征抽取器，使用默认MLP"
    )


def add_ddqn_args(group):
    group.add_argument(
        "--ddqn_episodes", type=int, default=200000, help="DDQN episodes"
    )
    group.add_argument("--ddqn_gamma", type=float, default=0.99, help="DDQN gamma")
    group.add_argument(
        "--ddqn_batch_size", type=int, default=32, help="DDQN batch size"
    )
    group.add_argument(
        "--ddqn_buffer_size", type=int, default=50000, help="DDQN replay buffer size"
    )
    group.add_argument(
        "--ddqn_burn_in", type=int, default=10000, help="DDQN burn-in steps"
    )
    group.add_argument("--ddqn_lr", type=float, default=1e-3, help="DDQN learning rate")
    group.add_argument(
        "--ddqn_update_freq",
        type=int,
        default=32,
        help="DDQN network update frequency",
    )
    group.add_argument(
        "--ddqn_sync_freq",
        type=int,
        default=2000,
        help="DDQN target sync frequency",
    )
    group.add_argument(
        "--ddqn_eval_freq",
        type=int,
        default=20,
        help="DDQN evaluation frequency (episodes)",
    )
    group.add_argument(
        "--ddqn_eval_iters",
        type=int,
        default=5,
        help="DDQN evaluation iterations",
    )
    group.add_argument(
        "--ddqn_checkpoint_freq",
        type=int,
        default=500,
        help="DDQN checkpoint 保存频率（按 episode 计，0 表示禁用）",
    )
    group.add_argument(
        "--ddqn_plot_freq",
        type=int,
        default=20,
        help="DDQN 训练曲线刷新频率（按 episode 计，0 表示禁用）",
    )
    group.add_argument(
        "--ddqn_plot_path",
        type=str,
        default="models_output/ddqn/training_curve.png",
        help="DDQN 训练曲线输出路径",
    )
    group.add_argument(
        "--ddqn_save_path",
        type=str,
        default="models/ddqn_model.pt",
        help="DDQN model save path",
    )
    group.add_argument(
        "--ddqn_load_path",
        type=str,
        default=None,
        help="DDQN model load path",
    )
