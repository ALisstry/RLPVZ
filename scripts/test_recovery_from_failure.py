"""测试：两个 PVZ 进程，一个失败后重启，另一个不受影响。
Usage:
    python scripts/test_recovery_from_failure.py
"""
import time
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hook_client import HookClient
from hook_client.injector import inject_dll, list_pvz_processes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_PATH = os.path.join(PROJECT_ROOT, "gameobj", "PlantsVsZombies.exe")
PAPER_PLANTS = [1, 0, 4, 3, 7, 17, 2, 5, 39, 21]

PORT_A = 12345
PORT_B = 12346
SPEED = 10.0
MODE = 6  # 白天困难


def _name(ui):
    return {0: "加载", 1: "主菜单", 2: "选卡", 3: "游戏中",
            4: "ZOMBIES_WON", 5: "AWARD"}.get(ui, "未知")


def launch_pvz():
    """启动一个 PVZ 进程，返回 PID。"""
    before = set(list_pvz_processes())
    subprocess.Popen([GAME_PATH], cwd=os.path.dirname(GAME_PATH))
    for _ in range(15):
        time.sleep(1)
        new_pids = set(list_pvz_processes()) - before
        if new_pids:
            return new_pids.pop()
    return None


def main():
    # ── 1. 启动两个 PVZ 进程 ──
    print("启动 PVZ 进程 A ...")
    pid_a = launch_pvz()
    if pid_a is None:
        print("进程 A 启动失败"); return
    print(f"进程 A pid={pid_a}")

    print("启动 PVZ 进程 B ...")
    pid_b = launch_pvz()
    if pid_b is None:
        print("进程 B 启动失败"); return
    print(f"进程 B pid={pid_b}")

    # ── 2. 分别注入 DLL + 连接（指定 PID，不混用）──
    print(f"\n--- 进程 A (pid={pid_a}): 注入 port={PORT_A} ---")
    if not inject_dll(pid=pid_a, port=PORT_A):
        print("A DLL 注入失败"); return
    time.sleep(0.5)
    client_a = HookClient(port=PORT_A)
    if not client_a.connect():
        print("A 连接失败"); return
    client_a.set_tick_ms(max(1, int(10.0 / SPEED)))

    print(f"--- 进程 B: 注入 port={PORT_B} ---")
    if not inject_dll(pid=pid_b, port=PORT_B):
        print("B DLL 注入失败"); return
    time.sleep(0.5)
    client_b = HookClient(port=PORT_B)
    if not client_b.connect():
        print("B 连接失败"); return
    client_b.set_tick_ms(max(1, int(10.0 / SPEED)))

    # ── 3. 进程 A 进入游戏；进程 B 留在选卡界面 ──
    print("\n=== 进程 A: 进入游戏 ===")
    if client_a.auto_start_game(mode=MODE, cards=PAPER_PLANTS, timeout=15.0):
        ui_a = 3
    else:
        ui_a = client_a.get_ui()
    print(f"进程 A UI={ui_a} ({_name(ui_a)})")

    print("=== 进程 B: 进入选卡界面 ===")
    ui_b = client_b.get_ui()
    if ui_b in (0, 1):
        if ui_b == 0:
            client_b.click(400, 300); time.sleep(1)
        client_b.start_game(MODE); time.sleep(1.5)
        ui_b = client_b.get_ui()
    if ui_b == 2:
        client_b.select_cards(PAPER_PLANTS); time.sleep(0.5)
    print(f"进程 B UI={ui_b} ({_name(ui_b)})")

    if ui_a != 3:
        print("进程 A 未能进入游戏"); return
    if ui_b not in (1, 2):
        print("进程 B 未能进入选卡"); return

    # ── 4. 等待进程 A 失败 ──
    print("\n等待进程 A 失败...")
    while True:
        time.sleep(0.5)
        ui_a = client_a.get_ui()
        if ui_a != 3:
            break
    print(f"进程 A UI={ui_a} ({_name(ui_a)})")
    if ui_a != 4:
        print("进程 A 不是 ZOMBIES_WON"); return

    # 确认进程 B 仍在选卡界面
    ui_b = client_b.get_ui()
    print(f"进程 B 当前 UI={ui_b} ({_name(ui_b)})")

    # ── 5. 只重启进程 A ──
    print(f"\n=== 重启进程 A (pid={pid_a}) ===")
    import psutil
    try:
        p = psutil.Process(pid_a)
        p.terminate()
        p.wait(timeout=10)
        print(f"进程 A pid={pid_a} 已终止")
    except Exception as e:
        print(f"终止失败: {e}")

    # 启动新 A（记录当前 PID 集合，只找新增的进程）
    known = set(list_pvz_processes())
    subprocess.Popen([GAME_PATH], cwd=os.path.dirname(GAME_PATH))
    print("等待新进程 A 启动...")
    new_pid = None
    for _ in range(15):
        time.sleep(1)
        new_pids = set(list_pvz_processes()) - known
        if new_pids:
            new_pid = new_pids.pop()
            break
    if new_pid is None:
        print("新进程 A 启动失败"); return
    print(f"新进程 A pid={new_pid}")

    # 注入 + 连接
    if not inject_dll(pid=new_pid, port=PORT_A):
        print("新 A DLL 注入失败"); return
    time.sleep(1.0)
    client_a = HookClient(port=PORT_A)
    if not client_a.connect():
        print("新 A 连接失败"); return
    client_a.set_tick_ms(max(1, int(10.0 / SPEED)))

    # 等待游戏初始化
    print("等待游戏初始化 (5s)...")
    time.sleep(5)
    print(f"新 A 当前 UI={client_a.get_ui()}")

    # 使用 auto_start_game 一键导航到游戏中
    ok = client_a.auto_start_game(mode=MODE, cards=PAPER_PLANTS, timeout=15.0)
    if not ok:
        print(f"auto_start_game 失败，等待 3s 后重试...")
        time.sleep(3)
        ok = client_a.auto_start_game(mode=MODE, cards=PAPER_PLANTS, timeout=15.0)
    if ok:
        print("✅ 新进程 A 已进入游戏")
    else:
        print(f"❌ auto_start_game 失败，UI={client_a.get_ui()}")

    # ── 6. 验证进程 B 未受影响 ──
    ui_b = client_b.get_ui()
    print(f"\n进程 B 最终 UI={ui_b} ({_name(ui_b)})")
    all_pids = list_pvz_processes()
    print(f"当前所有 PVZ 进程: {all_pids}")
    print(f"进程 B pid={pid_b} 仍在运行: {pid_b in all_pids}")

    if ui_b in (1, 2):
        print("✅ 进程 B 未受影响")
    else:
        print("❌ 进程 B 状态异常")


if __name__ == "__main__":
    main()
