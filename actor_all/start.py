# -*- coding: utf-8 -*-
"""
一键启动 4 桌掼蛋（1 RL + 3 EggPan 规则），后台运行，监控进程状态。
"""
import subprocess
import sys
import os
import time
import threading
import signal

# ===== 配置 =====
PYTHON = r"D:\conda_envs\egg\python.exe"
TCLI   = "clients/tcli.py"
PROJECT_ROOT = r"C:\Users\24704\Desktop\毕设\NUAA-guandan-main"

TABLES = [23456, 23457, 23458, 23459]

def start(port, mode, seat, extra_args=[]):
    """启动一个 tcli 客户端，不弹出窗口，返回 Popen 对象"""
    cmd = [
        PYTHON, TCLI, mode, str(seat),
        "--host", "127.0.0.1",
        "--port", str(port),
    ] + extra_args
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

def monitor_processes(procs, labels):
    """监控进程列表，直到所有进程退出或手动终止。"""
    while True:
        for i, (proc, label) in enumerate(zip(procs, labels)):
            poll = proc.poll()
            if poll is not None:
                # 进程已退出
                if poll == 0:
                    print(f"[{label}] 正常退出 (returncode=0)")
                elif poll < 0:
                    print(f"[{label}] 被信号终止 (signal={-poll})")
                else:
                    print(f"[{label}] 异常退出 (returncode={poll})")
                # 从监控列表中移除
                procs.pop(i)
                labels.pop(i)
                break   # 因为改变了列表长度，重新开始循环
        if not procs:
            print("\n所有子进程已退出。")
            break
        time.sleep(1)

def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print("正在启动 4 张桌子（每桌 1 强化学习 + 3 TOP 规则）...")
    print("=" * 60)

    all_procs = []
    all_labels = []

    # 1. 强化学习客户端
    for port in TABLES:
        label = f"Table RL (port {port})"
        proc = start(port, "reinforcement", 1,
                     extra_args=["--learner_host", "127.0.0.1", "--learner_port", "10002"])
        all_procs.append(proc)
        all_labels.append(label)
        print(f"启动 {label}")

    print("\n所有 RL 客户端已启动，等待 3 秒后启动规则客户端...\n")
    time.sleep(3)

    # 2. EggPan 规则客户端
    for port in TABLES:
        for seat in (2, 3, 4):
            label = f"TOP (port {port}, seat {seat})"
            proc = start(port, "rule", seat, extra_args=["-c", "TOP"])
            all_procs.append(proc)
            all_labels.append(label)
            print(f"启动 {label}")

    print("\n" + "=" * 60)
    print(f"全部 {len(all_procs)} 个客户端已启动，开始监控...")
    print("按 Ctrl+C 可终止所有客户端。")
    print("=" * 60)

    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_processes, args=(all_procs, all_labels))
    monitor_thread.daemon = True
    monitor_thread.start()

    # 主线程等待 Ctrl+C
    try:
        while monitor_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在终止所有客户端...")
        for p in all_procs:
            p.terminate()
        time.sleep(2)   # 等待进程彻底结束
        print("已全部终止。")

if __name__ == "__main__":
    main()