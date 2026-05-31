import argparse
from training.curriculum import CURRICULUM_CHOICES
from training.defaults import DEFAULT_GAME_PATH, DEFAULT_GAME_SPEED
from training.execution import EXECUTION_CHOICES
from training.registry import add_algorithm_args, available_algorithms


def get_args():
    """
    统一封装训练参数：通用 + PPO + DDQN。
    """
    parser = argparse.ArgumentParser(description="PVZ 训练")
    _add_common_args(parser)
    add_algorithm_args(parser)

    return parser.parse_args()


def _add_common_args(parser):
    common = parser.add_argument_group("common")
    common.add_argument(
        "--algo",
        type=str,
        default="ddqn",
        choices=available_algorithms(),
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
        "--training_config",
        type=str,
        default="config/training_config.yaml",
        help="训练环境配置文件路径",
    )
    common.add_argument(
        "--speed",
        "-s",
        type=float,
        default=DEFAULT_GAME_SPEED,
        help=f"游戏速度 (默认: {DEFAULT_GAME_SPEED}x, 最高10x)",
    )
    common.add_argument(
        "--frameskip",
        "-f",
        type=int,
        default=4,
        help="帧跳过 (4=适中，视野更远)",
    )
    common.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="自动加载最新模型继续训练 (默认启用)",
    )
    common.add_argument(
        "--no_auto_resume",
        action="store_true",
        help="禁用自动加载，从零开始训练",
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
    common.add_argument(
        "--execution",
        type=str,
        default="auto",
        choices=EXECUTION_CHOICES,
        help="训练执行策略，auto 表示使用算法默认策略",
    )
    common.add_argument(
        "--curriculum",
        type=str,
        default="none",
        choices=CURRICULUM_CHOICES,
        help="课程学习策略；当前仅启用 none，保留公共策略入口",
    )
