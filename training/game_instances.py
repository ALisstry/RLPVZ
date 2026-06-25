import os
import subprocess
import time

import psutil

from hook_client import inject_dll
from hook_client.injector import find_pvz_process, list_pvz_processes, find_new_pvz_process

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLKIT_PATH = os.path.join(
    PROJECT_ROOT, "PvZ_Toolkit_v1.22.0", "PvZ_Toolkit_v1.22.0_ZH.exe"
)


def prepare_game_instances(args):
    """准备游戏实例：解析配置，必要时自动启动游戏进程并注入 DLL。"""

    instances = _resolve_game_instances(args)
    if instances is None:
        return None

    if args.no_auto_start:
        pids = list_pvz_processes()
        print(f"[PVZ] 当前发现的所有进程 PID: {pids}")
        if any(i["pid"] is None for i in instances):
            print("[警告] no_auto_start 模式下需要手动启动所有 PVZ 进程或使用 --pids")
        return instances

    # Auto-start: 先启动 PvZ Toolkit，再根据实例数自动启动游戏进程并注入 DLL
    _launch_toolkit()

    multi = len(instances) > 1
    known_pids = set(list_pvz_processes())
    for i, instance in enumerate(instances):
        label = f"[{i + 1}/{len(instances)}]"
        new_pid = _launch_game_and_inject(
            game_path=args.game_path,
            wait_time=args.wait_time,
            port=instance["port"],
            pid=instance["pid"],
            label=label,
            force_new=multi,
        )
        if new_pid is None:
            print(f"[错误] {label} 实例启动失败")
            return None
        instance["pid"] = new_pid
        known_pids.add(new_pid)

    print(
        "[Hook] 所有实例注入完成: "
        + ", ".join(
            f"pid={i['pid']} port={i['port']}" for i in instances
        )
    )
    return instances


def _launch_game_and_inject(
    game_path: str = None,
    wait_time: float = 3.0,
    port: int = 12345,
    pid: int = None,
    label: str = "",
    force_new: bool = False,
) -> int | None:
    """启动游戏（若未运行）并注入 DLL，成功返回 PID，失败返回 None。

    如果已经给定了 pid，则直接对那个进程注入。
    force_new=True 时跳过复用已有进程逻辑，始终启动新游戏进程。
    """
    prefix = f"{label} " if label else ""

    if pid is not None:
        # 显式指定了 PID，直接注入
        if inject_dll(pid=pid, port=port):
            print(f"{prefix}[Hook] DLL 注入成功 (pid={pid}, port={port})")
            time.sleep(1.0)
            return pid
        print(f"{prefix}[错误] DLL 注入失败 (pid={pid})")
        return None

    # 单实例模式：优先复用已有进程
    if not force_new:
        existing = find_pvz_process()
        if existing is not None:
            print(f"{prefix}[PVZ] 发现已有进程 PID={existing}，直接注入")
            if inject_dll(pid=existing, port=port):
                print(f"{prefix}[Hook] DLL 注入成功 (pid={existing}, port={port})")
                time.sleep(1.0)
                return existing
            print(f"{prefix}[错误] 对已有进程注入失败 (pid={existing})")
            return None

    # 需要启动新游戏进程
    if not game_path:
        print(
            f"{prefix}[错误] 未配置游戏路径，"
            "请在 training_config.yaml 的 training.args.game_path 中设置"
        )
        return None
    if not os.path.exists(game_path):
        print(f"{prefix}[错误] 游戏文件不存在: {game_path}")
        return None

    # 记录启动前的 PID 列表，用于后续识别新进程
    before_pids = set(list_pvz_processes())

    try:
        subprocess.Popen(
            [game_path],
            cwd=os.path.dirname(game_path),
            creationflags=subprocess.DETACHED_PROCESS,
        )
    except Exception as e:
        print(f"{prefix}[错误] 启动游戏失败: {e}")
        return None

    print(f"{prefix}[游戏] 已启动 {game_path}，等待初始化...")
    time.sleep(wait_time)

    # 找到新启动的进程 PID
    new_pid = find_new_pvz_process(before_pids)
    if new_pid is None:
        print(f"{prefix}[错误] 未能检测到新启动的 PVZ 进程")
        return None

    print(f"{prefix}[PVZ] 新进程 PID={new_pid}")

    # 注入 DLL
    if inject_dll(pid=new_pid, port=port):
        print(f"{prefix}[Hook] DLL 注入成功 (pid={new_pid}, port={port})")
        time.sleep(1.0)
        return new_pid

    print(f"{prefix}[错误] DLL 注入失败 (pid={new_pid})")
    return None


def _resolve_game_instances(args):
    """解析游戏实例列表 [(pid, port), ...]。

    auto_start 模式：未指定 --pids 时所有 pid 为 None（由后续启动流程填充）。
    no_auto_start 模式：从当前运行的进程中发现。
    """
    requested = max(1, int(getattr(args, "num_envs", 1)))
    explicit_pids = _parse_int_list(getattr(args, "pids", ""))
    explicit_ports = _parse_int_list(getattr(args, "ports", ""))

    if explicit_pids and len(explicit_pids) != requested:
        raise ValueError("--pids 数量必须与 --num_envs 一致")
    if explicit_ports and len(explicit_ports) != requested:
        raise ValueError("--ports 数量必须与 --num_envs 一致")

    # 解析端口
    if explicit_ports:
        ports = explicit_ports
    elif requested == 1:
        ports = [int(getattr(args, "port", getattr(args, "base_port", 12345)))]
    else:
        base_port = int(getattr(args, "base_port", getattr(args, "port", 12345)))
        ports = [base_port + idx for idx in range(requested)]

    no_auto = bool(getattr(args, "no_auto_start", False))

    # 解析 PID
    if explicit_pids:
        pids = explicit_pids
    elif no_auto:
        # 手动模式：从当前运行的进程中获取
        discovered = list_pvz_processes()
        if len(discovered) < requested:
            raise RuntimeError(
                f"仅发现 {len(discovered)} 个 PVZ 进程，但需要 {requested} 个"
            )
        pids = discovered[:requested]
    else:
        # 自动启动模式：pid 留空，后续由 launch 流程填充
        pids = [None] * requested

    return [
        {"index": idx, "pid": pid, "port": port}
        for idx, (pid, port) in enumerate(zip(pids, ports))
    ]


def _launch_toolkit():
    """启动 PvZ Toolkit（游戏辅助工具），不启动多开器。"""
    _launch_background_exe(TOOLKIT_PATH, "PvZ Toolkit")


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


def _parse_int_list(raw_value):
    if not raw_value:
        return []
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]
