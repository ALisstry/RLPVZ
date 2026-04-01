import os
import sys
import time
import logging
import subprocess
from datetime import datetime

import psutil
import torch
from stable_baselines3.common.vec_env import VecNormalize

from hook_client import inject_dll
from hook_client.injector import find_pvz_process, list_pvz_processes
from train_config import DEFAULT_GAME_PATH, MODEL_PATH
from callbacks import (
    MemoryResetCallback,
    AutoCollectCallback,
    SimpleMonitorCallback,
    AsyncSingleModelCallback,
    HeatmapCallback,
    DetailedLogCallback,
    DynamicEntropyCallback,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLKIT_PATH = os.path.join(
    PROJECT_ROOT, "PvZ_Toolkit_v1.22.0", "PvZ_Toolkit_v1.22.0_ZH.exe"
)
V5_MANAGER_PATH = os.path.join(PROJECT_ROOT, "v5ProcessManager", "V5.exe")

_tee_stdout = None
_tee_stderr = None
_active_log_file = None


class _TeeStream:
    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def launch_game_and_inject(
    game_path: str = None,
    wait_time: float = 3.0,
    port: int = 12345,
    pid: int = None,
) -> bool:
    """
    自动启动游戏并注入 Hook DLL

    Args:
        game_path: 游戏可执行文件路径
                                 wait_time: 启动后等待时间（秒）
        port: Hook 服务端口

    Returns:
        是否成功
    """
    if game_path is None:
        game_path = DEFAULT_GAME_PATH

    # 检查游戏是否已运行
    pid = pid or find_pvz_process()
    if not pid:
        # 检查游戏文件是否存在
        if not os.path.exists(game_path):
            print(f"[错误] 游戏文件不存在: {game_path}")
            return False
        try:
            # 启动游戏（不等待，在后台运行）
            subprocess.Popen(
                [game_path],
                cwd=os.path.dirname(game_path),
                creationflags=subprocess.DETACHED_PROCESS,
            )
        except Exception as e:
            print(f"[错误] 启动游戏失败: {e}")
            return False

        time.sleep(wait_time)

        pid = find_pvz_process()
        if not pid:
            print("[错误] 游戏启动失败")
            return False

    # 注入 DLL
    if inject_dll(pid=pid, port=port):
        print(f"[Hook] DLL注入成功 (port {port})")
        # 等待 Hook 初始化
        time.sleep(1.0)
        return True
    else:
        print("[错误] DLL注入失败")
        return False


def _is_process_running(exe_path: str) -> bool:
    exe_name = os.path.basename(exe_path).lower()
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info["name"] or "").lower() == exe_name:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _launch_background_exe(exe_path: str, label: str) -> bool:
    if not os.path.exists(exe_path):
        print(f"[前置程序] 未找到 {label}: {exe_path}")
        return False

    if _is_process_running(exe_path):
        print(f"[前置程序] {label} 已在运行")
        return True

    try:
        subprocess.Popen(
            [exe_path],
            cwd=os.path.dirname(exe_path),
            creationflags=subprocess.DETACHED_PROCESS,
        )
        print(f"[前置程序] 已启动 {label}")
        return True
    except Exception as exc:
        print(f"[前置程序] 启动 {label} 失败: {exc}")
        return False


def ensure_prelaunch_tools():
    toolkit_ok = _launch_background_exe(TOOLKIT_PATH, "PvZ Toolkit")
    manager_ok = _launch_background_exe(V5_MANAGER_PATH, "V5 Process Manager")
    return toolkit_ok and manager_ok


def wait_for_pvz_processes(expected_count: int, poll_interval: float = 1.0):
    expected_count = max(1, int(expected_count))
    while True:
        pids = list_pvz_processes()
        if len(pids) >= expected_count:
            print(f"[PVZ] 已发现 {len(pids)} 个进程: {pids}")
            return pids
        print(
            f"[PVZ] 当前仅发现 {len(pids)}/{expected_count} 个进程，等待中...",
            flush=True,
        )
        time.sleep(poll_interval)


def print_detected_pvz_processes():
    pids = list_pvz_processes()
    print(f"[PVZ] 当前发现的所有进程 PID: {pids}")
    return pids


def find_latest_model():
    """查找最新的模型文件"""
    # 优先查找统一路径
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    # 兼容旧版本：搜索其他可能的模型位置
    import glob

    patterns = [
        "models/advanced_*/final_model.zip",
        "models/*/final_model.zip",
        "models/*.zip",
    ]

    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))

    if not all_models:
        return None

    # 按修改时间排序，返回最新的
    latest = max(all_models, key=os.path.getmtime)
    return latest


def get_model_output_dir(algo: str) -> str:
    return os.path.join("models_output", algo)


def get_cached_model_path(algo: str) -> str:
    if algo == "ddqn":
        return os.path.join(get_model_output_dir(algo), "latest_model.pt")
    return os.path.join(get_model_output_dir(algo), "latest_model.zip")


def get_cached_vecnormalize_path() -> str:
    return os.path.join(get_model_output_dir("ppo"), "latest_model_vecnormalize.pkl")


def prepare_resume_paths(args):
    if args.algo == "ddqn":
        if args.ddqn_load_path:
            print(f"使用参数指定 DDQN 模型路径：{args.ddqn_load_path}")
            return
        if args.no_auto_resume:
            print("自动恢复已禁用，DDQN 从零开始训练")
            return

        cached_path = get_cached_model_path("ddqn")
        if os.path.exists(cached_path):
            args.ddqn_load_path = cached_path
            print(f"自动恢复 DDQN: {cached_path}")
        else:
            print("未找到 DDQN 缓存模型，从零开始训练")
        return

    if args.load:
        print(f"使用参数指定模型路径：{args.load}")
        return

    if args.no_auto_resume:
        print("自动恢复已禁用，从零开始训练")
        return

    cached_path = get_cached_model_path("ppo")
    if os.path.exists(cached_path):
        args.load = cached_path
        print(f"自动恢复 PPO: {cached_path}")
        return

    load_path = find_latest_model()
    if load_path:
        args.load = load_path
        print(f"自动恢复: 找到最新模型 {load_path}")
    else:
        print("未找到已有模型，从零开始训练")


def setup_logging(args):
    from utils.logger import get_logger, LogLevel

    global _tee_stdout, _tee_stderr, _active_log_file

    if getattr(args, "log_file_path", None):
        log_file = args.log_file_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        args.log_file_path = log_file

    level_map = {
        0: LogLevel.WARNING,
        1: LogLevel.INFO,
        2: LogLevel.DEBUG,
    }
    logger = get_logger(level=level_map.get(getattr(args, "file_log_level", 1), LogLevel.INFO), file_path=log_file)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "file_log_level", 1) >= 2 else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

    if _active_log_file != log_file:
        log_handle = open(log_file, "a", encoding="utf-8")
        _tee_stdout = _TeeStream(sys.__stdout__, log_handle)
        _tee_stderr = _TeeStream(sys.__stderr__, log_handle)
        sys.stdout = _tee_stdout
        sys.stderr = _tee_stderr
        _active_log_file = log_file

    print(f"\r\n[日志] 调试信息将保存到: {log_file}")
    return logger


def setup_worker_logging(args):
    if getattr(args, "log_file_path", None):
        from utils.logger import get_logger, LogLevel

        level_map = {
            0: LogLevel.WARNING,
            1: LogLevel.INFO,
            2: LogLevel.DEBUG,
        }
        get_logger(
            level=level_map.get(getattr(args, "file_log_level", 1), LogLevel.INFO),
            file_path=args.log_file_path,
        )
        logging.basicConfig(
            level=logging.DEBUG if getattr(args, "file_log_level", 1) >= 2 else logging.INFO,
            format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.FileHandler(args.log_file_path, encoding="utf-8")],
            force=True,
        )


def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[设备] {torch.cuda.get_device_name(0)}")
        # 性能加速设置
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = "cpu"
        print("[设备] CPU")
    return device


def print_metadata(args):
    print("\r\n" + "=" * 60)
    print("高级 PVZ 训练 - 三大优化")
    print("=" * 60)

    actual_game_speed = min(args.speed, 10.0)  # tick_ms 最小为1，最高是10x
    _ = actual_game_speed * args.frameskip

    print(f"\r\n配置:")
    print(f"  速度: {actual_game_speed}x | 帧跳过: {args.frameskip} | 网络: {args.net}")
    print(f"  Batch: {args.batch} | Steps: {args.n_steps} | LR: {args.lr}")
    print(f"  探索: {args.start_ent} → {args.end_ent}")
    if getattr(args, "num_envs", 1) > 1:
        print(f"  并行环境: {args.num_envs} | base_port: {args.base_port}")


def ensure_game_ready(args):
    if not args.no_auto_start:
        ensure_prelaunch_tools()
        wait_for_pvz_processes(args.num_envs, poll_interval=max(0.5, args.wait_time))
        print_detected_pvz_processes()

    try:
        instances = resolve_game_instances(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}")
        return False
    args.game_instances = instances

    if args.no_auto_start:
        print_detected_pvz_processes()

    if len(instances) == 1 and not args.no_auto_start:
        instance = instances[0]
        if not launch_game_and_inject(
            game_path=args.game_path,
            wait_time=args.wait_time,
            port=instance["port"],
            pid=instance["pid"],
        ):
            print("无法启动游戏或注入 DLL，训练终止")
            print("  请使用 --no_auto_start 选项手动启动游戏和注入")
            return False
        instance["pid"] = find_pvz_process() if instance["pid"] is None else instance["pid"]
        return True

    if not args.no_auto_start:
        for instance in instances:
            pid = instance["pid"]
            if pid is None:
                print("多实例模式下请先启动所有 PVZ 进程，或使用 --pids 显式指定")
                return False
            if not inject_dll(pid=pid, port=instance["port"]):
                print(f"无法注入 DLL: pid={pid}, port={instance['port']}")
                return False
        print(
            "[Hook] 多实例注入完成: "
            + ", ".join(
                f"pid={instance['pid']} port={instance['port']}" for instance in instances
            )
        )
    else:
        print("自动启动已禁用，请确保游戏已启动并注入 DLL")
    return True


def _parse_int_list(raw_value):
    if not raw_value:
        return []
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def resolve_game_instances(args):
    requested = max(1, int(getattr(args, "num_envs", 1)))
    explicit_pids = _parse_int_list(getattr(args, "pids", ""))
    explicit_ports = _parse_int_list(getattr(args, "ports", ""))

    if explicit_pids and len(explicit_pids) != requested:
        raise ValueError("--pids 数量必须与 --num_envs 一致")
    if explicit_ports and len(explicit_ports) != requested:
        raise ValueError("--ports 数量必须与 --num_envs 一致")

    if explicit_ports:
        ports = explicit_ports
    elif requested == 1:
        ports = [int(getattr(args, "port", getattr(args, "base_port", 12345)))]
    else:
        base_port = int(getattr(args, "base_port", getattr(args, "port", 12345)))
        ports = [base_port + idx for idx in range(requested)]

    if explicit_pids:
        pids = explicit_pids
    else:
        discovered = list_pvz_processes()
        if len(discovered) < requested:
            if requested == 1 and not getattr(args, "no_auto_start", False):
                pids = [None]
            else:
                raise RuntimeError(
                    f"仅发现 {len(discovered)} 个 PVZ 进程，但需要 {requested} 个"
                )
        else:
            pids = discovered[:requested]

    return [
        {"index": idx, "pid": pid, "port": port}
        for idx, (pid, port) in enumerate(zip(pids, ports))
    ]


def resolve_load_path(args):
    """获取模型 checkpoint 路径"""
    if args.load:
        print(f"使用参数指定模型路径：{args.load}")
        return args.load

    if args.no_auto_resume:
        print("自动恢复已禁用，从零开始训练")
        return None

    load_path = find_latest_model()
    if load_path:
        print(f"自动恢复: 找到最新模型 {load_path}")
    else:
        print(f"未找到已有模型，从零开始训练")
    return load_path


def save_runtime_checkpoint(args, model=None, env=None, network=None, tag=None):
    os.makedirs(get_model_output_dir(args.algo), exist_ok=True)

    if args.algo == "ddqn":
        import torch
        from models.ddqn.ddqn import copy_state_dict_to_cpu

        if network is None:
            return None

        cpu_state_dict = copy_state_dict_to_cpu(network.state_dict())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        explicit_path = getattr(args, "ddqn_save_path", None)
        cached_path = get_cached_model_path("ddqn")
        tagged_path = None
        if tag:
            tagged_path = os.path.join(get_model_output_dir("ddqn"), f"{tag}.pt")
        if explicit_path:
            os.makedirs(os.path.dirname(explicit_path) or ".", exist_ok=True)
            torch.save(cpu_state_dict, explicit_path)
        torch.save(cpu_state_dict, cached_path)
        if tagged_path:
            torch.save(cpu_state_dict, tagged_path)
        print(f"\n模型已保存: {cached_path}")
        return cached_path

    if model is None:
        return None

    def _find_vec_normalize(current_env):
        if current_env is None:
            return None
        if isinstance(current_env, VecNormalize):
            return current_env
        for attr in ("venv", "env"):
            nested = getattr(current_env, attr, None)
            if nested is not None:
                found = _find_vec_normalize(nested)
                if found is not None:
                    return found
        return None

    explicit_path = getattr(args, "save_path", None)
    cached_path = get_cached_model_path("ppo")
    tagged_path = None
    if tag:
        tagged_path = os.path.join(get_model_output_dir("ppo"), f"{tag}.zip")
    if explicit_path:
        os.makedirs(os.path.dirname(explicit_path) or ".", exist_ok=True)
        model.save(explicit_path)
    model.save(cached_path)
    if tagged_path:
        model.save(tagged_path)

    vec_normalize = _find_vec_normalize(env)
    if vec_normalize is not None:
        try:
            if explicit_path:
                explicit_vec_path = os.path.splitext(explicit_path)[0] + "_vecnormalize.pkl"
                vec_normalize.save(explicit_vec_path)
            vec_normalize.save(get_cached_vecnormalize_path())
            if tagged_path:
                tagged_vec_path = os.path.splitext(tagged_path)[0] + "_vecnormalize.pkl"
                vec_normalize.save(tagged_vec_path)
        except Exception:
            pass

    print(f"\n模型已保存: {cached_path}")
    return cached_path


def build_training_metrics_callback(args):
    if args.algo != "ddqn" or getattr(args, "ddqn_plot_freq", 0) <= 0:
        return None

    from utils.training_plotter import TrainingCurvePlotter

    plotter = TrainingCurvePlotter(
        output_path=getattr(args, "ddqn_plot_path", "models_output/ddqn/training_curve.png"),
        refresh_freq=getattr(args, "ddqn_plot_freq", 20),
    )

    def on_metrics(snapshot):
        plotter.maybe_update(
            step_count=snapshot["step_count"],
            episode_rewards=snapshot["episode_rewards"],
            mean_rewards=snapshot["mean_rewards"],
            mean_iterations=snapshot["mean_iterations"],
            eval_steps=snapshot["eval_steps"],
            eval_rewards=snapshot["eval_rewards"],
            losses=snapshot["losses"],
            force=snapshot.get("force", False),
        )

    return on_metrics


def print_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存: {allocated:.2f} GB")


def build_callbacks(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/advanced_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    callbacks = [
        MemoryResetCallback(verbose=0),  # 🆕 记忆重置 (在episode开始时)
        AutoCollectCallback(),  # 强制自动收集
        SimpleMonitorCallback(),  # 简洁监控：连胜/连败
        AsyncSingleModelCallback(
            save_freq=args.save_freq, save_path=args.save_path, verbose=1
        ),  # 异步保存
        HeatmapCallback(
            save_path="heatmap.html", refresh_rate=10, verbose=1
        ),  # 实时热力图 (开启 verbose 以显示 Attention Debug)
        DetailedLogCallback(log_freq=500),  # 新增：详细数据日志
    ]

    dynamic_entropy = DynamicEntropyCallback(
        start_ent_coef=args.start_ent,
        end_ent_coef=args.end_ent,
        decay_type=args.ent_decay,
        total_timesteps=args.timesteps,
        warmup_steps=min(10000, args.timesteps // 10),
        verbose=0,  # 静默
    )
    callbacks.append(dynamic_entropy)
    return callbacks


def train_model(model, env, args, callbacks):
    print(f"开始训练...\r\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,  # 关闭进度条，用自己的输出
        )
    except KeyboardInterrupt:
        print("\r\n 训练被中断")
    finally:
        save_runtime_checkpoint(args, model=model, env=env)
        env.close()

