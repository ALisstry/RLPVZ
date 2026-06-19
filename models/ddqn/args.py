import argparse

# 实现
# add_args: 注册该算法支持的 CLI 参数。


def add_args(group):
    group.add_argument(
        "--ddqn_episodes", type=int, default=argparse.SUPPRESS, help="DDQN episodes"
    )
    group.add_argument("--ddqn_gamma", type=float, default=argparse.SUPPRESS, help="DDQN gamma")
    group.add_argument(
        "--ddqn_batch_size", type=int, default=argparse.SUPPRESS, help="DDQN batch size"
    )
    group.add_argument(
        "--ddqn_buffer_size", type=int, default=argparse.SUPPRESS, help="DDQN replay buffer size"
    )
    group.add_argument(
        "--ddqn_burn_in", type=int, default=argparse.SUPPRESS, help="DDQN burn-in steps"
    )
    group.add_argument("--ddqn_lr", type=float, default=argparse.SUPPRESS, help="DDQN learning rate")
    group.add_argument(
        "--ddqn_update_freq",
        type=int,
        default=argparse.SUPPRESS,
        help="DDQN network update frequency",
    )
    group.add_argument(
        "--ddqn_sync_freq",
        type=int,
        default=argparse.SUPPRESS,
        help="DDQN target sync frequency",
    )
    group.add_argument(
        "--ddqn_eval_freq",
        type=int,
        default=argparse.SUPPRESS,
        help="DDQN evaluation frequency (episodes)",
    )
    group.add_argument(
        "--ddqn_eval_iters",
        type=int,
        default=argparse.SUPPRESS,
        help="DDQN evaluation iterations",
    )
    group.add_argument(
        "--ddqn_checkpoint_freq",
        type=int,
        default=argparse.SUPPRESS,
        help="DDQN checkpoint 保存频率（按 episode 计，0 表示禁用）",
    )
    group.add_argument(
        "--ddqn_plot_freq",
        type=int,
        default=argparse.SUPPRESS,
        help="DDQN 训练曲线刷新频率（按 episode 计，0 表示禁用）",
    )
    group.add_argument(
        "--ddqn_plot_path",
        type=str,
        default=argparse.SUPPRESS,
        help="DDQN 训练曲线输出路径，默认使用公共输出目录",
    )
    group.add_argument(
        "--ddqn_save_path",
        type=str,
        default=argparse.SUPPRESS,
        help="DDQN 额外模型保存路径；默认只保存到公共输出目录",
    )
    group.add_argument(
        "--ddqn_load_path",
        type=str,
        default=argparse.SUPPRESS,
        help="DDQN model load path",
    )
    group.add_argument(
        "--ddqn_hidden_sizes",
        type=str,
        default=argparse.SUPPRESS,
        help="DDQN hidden layer sizes, comma-separated (e.g. 2048,2048)",
    )
    group.add_argument(
        "--ddqn_paper_observation",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use paper-format 596-dim one-hot observation",
    )
    group.add_argument(
        "--no_ddqn_paper_observation",
        dest="ddqn_paper_observation",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Use legacy flat observation (default)",
    )
