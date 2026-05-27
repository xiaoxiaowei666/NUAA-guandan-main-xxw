# -*- coding: utf-8 -*-
"""
启动 4 桌分布式模仿学习（每桌 1 个 ImitationDist + 3 个 TOP 规则）
使用 tcli.py 启动规则客户端，使用 tcli_imitation.py 启动分布式DAgger客户端
"""

import subprocess
import sys
import os
import time

PYTHON = r"D:\conda_envs\egg\python.exe"          # 你的 Python 解释器路径
IMITATION_CLIENT = "clients/tcli_imitation.py"    # 分布式模仿学习客户端脚本
RULE_CLIENT = "clients/tcli.py"                   # 规则客户端脚本
PROJECT_ROOT = r"C:\Users\24704\Desktop\毕设\NUAA-guandan-main"   # 项目根目录

TABLES = [23456, 23457, 23458, 23459]

def start_imitation(port, seat, extra_args=[]):
    """启动一个分布式模仿学习客户端"""
    cmd = [
        PYTHON, IMITATION_CLIENT, "imitation_dist", str(seat),
        "--host", "127.0.0.1",
        "--port", str(port),
    ] + extra_args
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace"
    )

def start_rule(port, seat):
    """启动一个 EggPan 规则客户端"""
    cmd = [
        PYTHON, RULE_CLIENT, "rule", str(seat),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", "TOP"
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace"
    )

def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print("正在启动 4 桌分布式模仿学习...")
    print("=" * 60)

    # 1. 启动所有 imitation_dist 客户端（座位 1）
    im_procs = []
    for port in TABLES:
        print(f"启动 Table 模仿学习客户端 (端口 {port}, 座位 1)...")
        proc = start_imitation(port, 1,
                               extra_args=["--learner_host", "127.0.0.1",
                                           "--learner_port", "10003"])
        im_procs.append(proc)

    print("\n所有模仿学习客户端已启动，等待 3 秒后启动规则客户端...\n")
    time.sleep(3)

    # 2. 启动 EggPan 规则客户端（座位 2~4）
    rule_procs = []
    for port in TABLES:
        for seat in (2, 3, 4):
            print(f"启动 TOP 规则 (端口 {port}, 座位 {seat})...")
            proc = start_rule(port, seat)
            rule_procs.append(proc)

    print("\n" + "=" * 60)
    print(f"全部客户端已启动！")
    print(f"  模仿学习: {len(im_procs)} 个")
    print(f"  TOP 规则: {len(rule_procs)} 个")
    print("所有进程在后台运行，关闭本窗口不会影响对局。")
    print("如需终止，请在任务管理器中结束 python 进程。")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在终止所有客户端...")
        for p in im_procs + rule_procs:
            p.terminate()
        print("已全部终止。")


if __name__ == "__main__":
    main()