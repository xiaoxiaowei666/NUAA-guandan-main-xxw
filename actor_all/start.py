# -*- coding: utf-8 -*-
"""
一键启动 4 桌掼蛋（每桌 1 RL + 3 EggPan 规则），所有客户端后台运行，不弹窗。
"""
import subprocess
import sys
import os
import time

# ===== 配置 =====
PYTHON = r"D:\conda_envs\egg\python.exe"
TCLI   = "clients/tcli.py"
PROJECT_ROOT = r"C:\Users\24704\Desktop\毕设\NUAA-guandan-main"

# 四张桌子的端口
TABLES = [23456, 23457, 23458, 23459]

def start(port, mode, seat, extra_args=[]):
    """启动一个 tcli 客户端，不弹出窗口，返回 Popen 对象"""
    cmd = [
        PYTHON, TCLI, mode, str(seat),
        "--host", "127.0.0.1",
        "--port", str(port),
    ] + extra_args
    # CREATE_NO_WINDOW 在 Windows 下隐藏窗口，非 Windows 系统可用 0
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace",
    )
    return proc

def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print("正在启动 4 张桌子（每桌 1 强化学习 + 3 EggPan 规则）...")
    print("=" * 60)

    # 1. 先启动所有强化学习客户端（座位1）
    rl_procs = []
    for port in TABLES:
        print(f"启动 Table RL (端口 {port})...")
        proc = start(port, "reinforcement", 1,
                     extra_args=["--learner_host", "127.0.0.1", "--learner_port", "10002"])
        rl_procs.append(proc)

    print("\n所有 RL 客户端已启动，等待 5 秒后启动规则客户端...\n")
    time.sleep(3)

    # 2. 启动所有 EggPan 规则客户端（座位 2~4，无渲染）
    rule_procs = []
    for port in TABLES:
        for seat in (2, 3, 4):
            print(f"启动 EggPan 规则 (端口 {port}, 座位 {seat})...")
            proc = start(port, "rule", seat,
                         extra_args=["-c", "EggPan"])  # 无 -r，不渲染
            rule_procs.append(proc)

    print("\n" + "=" * 60)
    print(f"全部客户端已启动！")
    print(f"  强化学习: {len(rl_procs)} 个")
    print(f"  EggPan 规则: {len(rule_procs)} 个")
    print("所有进程在后台运行，关闭本窗口不会影响对局。")
    print("如需终止，请在任务管理器中结束 python 进程。")
    print("=" * 60)

    # 可选：一直保持脚本运行，直到用户按 Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在终止所有客户端...")
        for p in rl_procs + rule_procs:
            p.terminate()
        print("已全部终止。")

if __name__ == "__main__":
    main()