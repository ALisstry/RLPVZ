import os
import subprocess
import time

import psutil

from hook_client import inject_dll
from hook_client.injector import find_pvz_process, list_pvz_processes
from .defaults import DEFAULT_GAME_PATH


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLKIT_PATH = os.path.join(
    PROJECT_ROOT, "PvZ_Toolkit_v1.22.0", "PvZ_Toolkit_v1.22.0_ZH.exe"
)
V5_MANAGER_PATH = os.path.join(PROJECT_ROOT, "v5ProcessManager", "V5.exe")


def prepare_game_instances(args):
    if not args.no_auto_start:
        ensure_prelaunch_tools()
        wait_for_pvz_processes(args.num_envs, poll_interval=max(0.5, args.wait_time))
        print_detected_pvz_processes()

    try:
        instances = resolve_game_instances(args)
    except (RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}")
        return None

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
            return None
        instance["pid"] = find_pvz_process() if instance["pid"] is None else instance["pid"]
        return instances

    if not args.no_auto_start:
        for instance in instances:
            pid = instance["pid"]
            if pid is None:
                print("多实例模式下请先启动所有 PVZ 进程，或使用 --pids 显式指定")
                return None
            if not inject_dll(pid=pid, port=instance["port"]):
                print(f"无法注入 DLL: pid={pid}, port={instance['port']}")
                return None
        print(
            "[Hook] 多实例注入完成: "
            + ", ".join(
                f"pid={instance['pid']} port={instance['port']}" for instance in instances
            )
        )
    else:
        print("自动启动已禁用，请确保游戏已启动并注入 DLL")
    return instances


def launch_game_and_inject(
    game_path: str = None,
    wait_time: float = 3.0,
    port: int = 12345,
    pid: int = None,
) -> bool:
    if game_path is None:
        game_path = DEFAULT_GAME_PATH

    pid = pid or find_pvz_process()
    if not pid:
        if not os.path.exists(game_path):
            print(f"[错误] 游戏文件不存在: {game_path}")
            return False
        try:
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

    if inject_dll(pid=pid, port=port):
        print(f"[Hook] DLL注入成功 (port {port})")
        time.sleep(1.0)
        return True

    print("[错误] DLL注入失败")
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
