import argparse

# 实现
# add_args: 注册该算法支持的 CLI 参数。


def add_args(group):
    group.add_argument("--timesteps", "-t", type=int, default=argparse.SUPPRESS, help="训练步数")
    group.add_argument(
        "--batch", "-b", type=int, default=argparse.SUPPRESS, help="Batch size (GPU空闲，增大Batch)",
    )
    group.add_argument(
        "--n_steps",
        "-n",
        type=int,
        default=argparse.SUPPRESS,
        help="N steps (针对 RTX 3050 优化，约 80 秒/更新)",
    )
    group.add_argument(
        "--n_epochs", type=int, default=argparse.SUPPRESS, help="训练轮数 (数据珍贵，多练几轮)"
    )
    group.add_argument(
        "--net",
        type=str,
        default=argparse.SUPPRESS,
        choices=["small", "medium", "large", "xlarge", "huge"],
        help="网络大小",
    )
    group.add_argument("--lr", type=float, default=argparse.SUPPRESS, help="学习率")

    group.add_argument(
        "--start_ent", type=float, default=argparse.SUPPRESS, help="初始探索系数 (0.15=较少随机)",
    )
    group.add_argument("--end_ent", type=float, default=argparse.SUPPRESS, help="最终探索系数")
    group.add_argument(
        "--ent_decay",
        type=str,
        default=argparse.SUPPRESS,
        choices=["linear", "exponential", "cosine"],
        help="探索衰减方式",
    )
    group.add_argument("--diversify", type=float, default=argparse.SUPPRESS, help="多样化概率 (0-1)")
    group.add_argument(
        "--no_diversify", action="store_true", default=argparse.SUPPRESS, help="禁用多样化"
    )
    group.add_argument(
        "--no_failure_priority", action="store_true", default=argparse.SUPPRESS, help="禁用失败优先学习"
    )

    group.add_argument(
        "--load",
        type=str,
        default=argparse.SUPPRESS,
        help="加载已有模型继续训练；默认使用公共缓存自动恢复",
    )
    group.add_argument(
        "--save_path",
        type=str,
        default=argparse.SUPPRESS,
        help="额外模型保存路径；默认只保存到公共输出目录",
    )
    group.add_argument("--save_freq", type=int, default=argparse.SUPPRESS, help="自动保存频率 (步)")
    group.add_argument(
        "--ppo_plot_freq",
        type=int,
        default=argparse.SUPPRESS,
        help="PPO 训练曲线刷新频率（按 step/episode 计，0 表示禁用）",
    )
    group.add_argument(
        "--ppo_plot_path",
        type=str,
        default=argparse.SUPPRESS,
        help="PPO 训练曲线输出路径，默认使用公共输出目录",
    )

    group.add_argument(
        "--no_attn", action="store_true", default=argparse.SUPPRESS, help="禁用注意力特征抽取器，使用默认MLP"
    )
