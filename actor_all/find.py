# -*- coding: utf-8 -*-
"""
只启动 4 个 RL 客户端，全部连接到单桌 23456，不启动规则客户端
"""
import subprocess
import sys
import os
import time

# ===== 配置 =====
PYTHON = r"D:\conda_envs\egg\python.exe"
TCLI   = "clients/tcli.py"
PROJECT_ROOT = r"C:\Users\24704\Desktop\毕设\NUAA-guandan-main"

# 固定连接端口 23456
PORT = 23456
# 启动 4 个 RL 客户端
RL_COUNT = 4

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace",
    )
    return proc

def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print(f"正在启动 {RL_COUNT} 个 RL 客户端，全部连接到端口 {PORT}...")
    print("不启动任何规则客户端")
    print("=" * 60)

    # 只启动 RL 客户端，全部连 23456，座位自动填 1/2/3/4
    rl_procs = []
    for i in range(RL_COUNT):
        seat = i + 1  # 座位 1,2,3,4
        print(f"启动 RL 客户端 {i+1} (端口 {PORT}, 座位 {seat})...")
        proc = start(PORT, "reinforcement", seat,
                     extra_args=["--learner_host", "127.0.0.1", "--learner_port", "10002"])
        rl_procs.append(proc)
        time.sleep(0.2)

    print("\n" + "=" * 60)
    print(f"✅ 全部启动完成！")
    print(f"  RL 客户端: {len(rl_procs)} 个 (连接 23456)")
    print(f"  规则客户端: 0 个")
    print("所有进程后台运行，关闭本窗口不会停止运行")
    print("按 Ctrl+C 可一键终止所有进程")
    print("=" * 60)

    # 保持运行，Ctrl+C 关闭
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在终止所有 RL 客户端...")
        for p in rl_procs:
            p.terminate()
        print("已全部终止。")

if __name__ == "__main__":
    main()