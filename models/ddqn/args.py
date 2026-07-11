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
    # PER (Prioritized Experience Replay)
    group.add_argument(
        "--ddqn_per_alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="PER prioritization exponent (0=uniform, 1=full priority, default 0.6)",
    )
    group.add_argument(
        "--ddqn_per_beta",
        type=float,
        default=argparse.SUPPRESS,
        help="PER IS correction start value (anneals to 1.0, default 0.4)",
    )
    group.add_argument(
        "--ddqn_per_epsilon",
        type=float,
        default=argparse.SUPPRESS,
        help="PER small positive constant to avoid zero priority (default 1e-6)",
    )
    # ── Differential Q-Network ──
    group.add_argument(
        "--use_differential",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use Differential Q-Network: Q(s,a) = Q(s,wait) + Δ(s,a), Δ(s,wait) ≡ 0",
    )
    group.add_argument(
        "--use_cnn",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use CNN-based Q-Network with dual-branch grid processing (V1)",
    )
    group.add_argument(
        "--use_cnn_v2",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use Row-First CNN (V2): 1×5 row-encoder → 3×3 spatial → GAP (single-branch, ~0.5M params)",
    )
    group.add_argument(
        "--use_dueling",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use Dueling DQN architecture (Wang et al., 2016)",
    )
    group.add_argument(
        "--use_factored",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use 3-Factor Q-Network: Q(card,row,col) = q_card + q_row + q_col (MLP)",
    )
