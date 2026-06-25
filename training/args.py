import argparse

from training.curriculum import CURRICULUM_CHOICES
from training.execution import EXECUTION_CHOICES
from training.registry import add_algorithm_args, available_algorithms
import training.constants as const
from utils.train_utils import load_training_config


def get_args(argv=None):
    """
    统一封装训练参数：通用 + 算法插件。
    """
    # 读取 config file 中的配置
    config_path, cli_algo = _preparse_config(argv)
    config = load_training_config(config_path)
    config_defaults = _build_config_defaults(config, config_path, cli_algo)

    # 设置默认值
    parser = argparse.ArgumentParser(description="PVZ 训练")
    _add_common_args(parser)
    add_algorithm_args(parser)
    parser.set_defaults(**config_defaults)

    # 应用传入值
    return parser.parse_args(argv)


# 读取 training_config 和 algo
def _preparse_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training_config", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--algo",
        type=str,
        choices=available_algorithms(),
        default=argparse.SUPPRESS,
    )
    parsed, _ = parser.parse_known_args(argv)
    return getattr(parsed, "training_config", const.CONFIG_PATH), getattr(
        parsed, "algo", None
    )


def _build_config_defaults(config, config_path, cli_algo=None):
    training = config.get("training", {})
    defaults = dict(training.get("args", {}))
    algo = cli_algo or defaults.get("algo")
    if not algo:
        raise ValueError("training_config.yaml 缺少训练算法配置")
    if algo not in available_algorithms():
        raise ValueError(f"Unsupported algorithm in training_config.yaml: {algo}")

    defaults.update(training.get(algo, {}))
    defaults["algo"] = algo
    defaults["training_config"] = config_path
    return defaults


def _add_common_args(parser):
    common = parser.add_argument_group("common")
    common.add_argument(
        "--algo",
        type=str,
        choices=available_algorithms(),
        default=argparse.SUPPRESS,
        help="训练算法",
    )
    common.add_argument(
        "--port", "-p", type=int, default=argparse.SUPPRESS, help="Hook 端口"
    )
    common.add_argument(
        "--num_envs",
        type=int,
        default=argparse.SUPPRESS,
        help="并行环境数量，对应多个 PVZ 进程",
    )
    common.add_argument(
        "--base_port",
        type=int,
        default=argparse.SUPPRESS,
        help="多实例自动分配时的起始 Hook 端口",
    )
    common.add_argument(
        "--ports",
        type=str,
        default=argparse.SUPPRESS,
        help="显式指定多实例端口，逗号分隔，如 12345,12346,12347",
    )
    common.add_argument(
        "--pids",
        type=str,
        default=argparse.SUPPRESS,
        help="显式指定目标 PVZ 进程 PID，逗号分隔",
    )
    common.add_argument(
        "--auto_start",
        dest="no_auto_start",
        action="store_false",
        default=argparse.SUPPRESS,
        help="根据 --num_envs 自动启动对应数量的游戏进程并注入 DLL",
    )
    common.add_argument(
        "--no_auto_start",
        dest="no_auto_start",
        action="store_true",
        default=argparse.SUPPRESS,
        help="禁用自动启动 (需手动启动游戏进程和注入 DLL)",
    )
    common.add_argument(
        "--game_path",
        type=str,
        default=argparse.SUPPRESS,
        help="游戏路径",
    )
    common.add_argument(
        "--wait_time",
        type=float,
        default=argparse.SUPPRESS,
        help="游戏启动等待时间 (秒)",
    )
    common.add_argument(
        "--training_config",
        type=str,
        default=argparse.SUPPRESS,
        help="训练环境配置文件路径",
    )
    common.add_argument(
        "--speed",
        "-s",
        type=float,
        default=argparse.SUPPRESS,
        help="游戏速度 (最高10x)",
    )
    common.add_argument(
        "--frameskip",
        "-f",
        type=int,
        default=argparse.SUPPRESS,
        help="帧跳过",
    )
    common.add_argument(
        "--auto_resume",
        dest="no_auto_resume",
        action="store_false",
        default=argparse.SUPPRESS,
        help="自动加载最新模型继续训练",
    )
    common.add_argument(
        "--no_auto_resume",
        dest="no_auto_resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="禁用自动加载，从零开始训练",
    )
    common.add_argument(
        "--env_console_log_level",
        "--env_verbose",
        dest="env_console_log_level",
        type=int,
        default=argparse.SUPPRESS,
        choices=[0, 1, 2],
        help="环境控制台日志级别: 0=静默, 1=关键信息, 2=详细调试",
    )
    common.add_argument(
        "--file_log_level",
        "--log_verbose",
        dest="file_log_level",
        type=int,
        default=argparse.SUPPRESS,
        choices=[0, 1, 2],
        help="文件日志级别: 0=静默, 1=关键信息, 2=详细调试",
    )
    common.add_argument(
        "--execution",
        type=str,
        default=argparse.SUPPRESS,
        choices=EXECUTION_CHOICES,
        help="训练执行策略，auto 表示使用算法默认策略",
    )
    common.add_argument(
        "--curriculum",
        type=str,
        default=argparse.SUPPRESS,
        choices=CURRICULUM_CHOICES,
        help="课程学习策略: none=关闭, stage_gate=按阶段指标升阶",
    )
