# -*- coding: utf-8 -*-
"""
自博弈启动脚本 (MCTS 版) — 1 桌 TOP 强对手 + 3 桌自博弈旧模型对手。
RL 客户端使用 MCTS 增强推理 (tcli_mcts.py)。
不启动游戏服务器，只连接已有端口。
"""

import subprocess
import sys
import os
import glob
import time

# ===== 配置 =====
PYTHON = r"D:\conda_envs\egg\python.exe"
TCLI_MCTS = "clients/tcli_mcts.py"
SP_CLI = "clients/selfplay_opponent.py"
PROJECT_ROOT = r"C:\Users\24704\Desktop\毕设\NUAA-guandan-main"

TABLES = [23456, 23457, 23458, 23459]

LEARNER_HOST = "127.0.0.1"
LEARNER_PUB_PORT = 10002

MODEL_POOL_DIR = os.path.join(PROJECT_ROOT, "model", "selfplay_checkpoints")

# MCTS 参数（批量计算优化，CPU 每步约 1-3 秒）
MCTS_SIMS = 30       # 每次决策的模拟次数
MCTS_DEPTH = 3       # rollout 深度
MCTS_DET = 2         # determinization 采样次数
TD_N_STEP = 3        # TD(n) 多步回报步数（1=TD(0), 3=TD(3)）


def start_process(script, *args, extra_args=None):
    cmd = [PYTHON, script] + list(args)
    if extra_args:
        cmd.extend(extra_args)
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace",
    )


def start_mcts_client(port, seat=1):
    """使用 MCTS 客户端 (tcli_mcts.py)"""
    return start_process(
        TCLI_MCTS, "reinforcement_mcts", str(seat),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--learner_host", LEARNER_HOST,
        "--learner_port", str(LEARNER_PUB_PORT),
        "--mcts_sims", str(MCTS_SIMS),
        "--mcts_depth", str(MCTS_DEPTH),
        "--mcts_det", str(MCTS_DET),
        "--n_step", str(TD_N_STEP),
    )


def start_top_opponent(port, seat):
    return start_process(
        "clients/tcli.py", "rule", str(seat),
        "--host", "127.0.0.1",
        "--port", str(port),
        extra_args=["-c", "TOP"],
    )


def start_sp_opponent(port, seat, model_index=0, epsilon=0.0):
    return start_process(
        SP_CLI, str(seat),
        "--model_index", str(model_index),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--epsilon", str(epsilon),
    )


def get_saved_models():
    if not os.path.isdir(MODEL_POOL_DIR):
        return []
    files = glob.glob(os.path.join(MODEL_POOL_DIR, "*.pth"))
    def extract_step(path):
        try:
            return int(os.path.basename(path).replace(".pth", ""))
        except Exception:
            return 0
    return sorted(files, key=extract_step)


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print("自博弈训练系统 (MCTS版) 启动中...")
    print(f"  MCTS: sims={MCTS_SIMS}, depth={MCTS_DEPTH}, det={MCTS_DET}")
    print("=" * 60)

    all_procs = []
    all_labels = []

    # ---- 1. MCTS RL 客户端 (4 桌，每桌 seat=1) ----
    print("\n[1/3] 启动 MCTS RL 客户端...")
    for port in TABLES:
        label = f"MCTS-RL (port {port})"
        proc = start_mcts_client(port, seat=1)
        all_procs.append(proc)
        all_labels.append(label)
        print(f"  {label}")

    print("  等待 3 秒后启动对手...")
    time.sleep(3)

    # ---- 2. 桌子 1: TOP 强对手（1桌对抗） ----
    print("\n[2/3] 启动桌子 1 的 TOP 对手...")
    top_tables = TABLES[:1]
    for port in top_tables:
        for seat in (2, 3, 4):
            label = f"TOP (port {port}, seat {seat})"
            proc = start_top_opponent(port, seat)
            all_procs.append(proc)
            all_labels.append(label)
            print(f"  {label}")

    # ---- 3. 桌子 2-4: 自博弈对手（3桌热加载） ----
    print("\n[3/3] 启动桌子 2-4 的自博弈对手（热加载模型池）...")
    saved_models = get_saved_models()
    print(f"  模型池: {len(saved_models)} 个 checkpoint")

    sp_tables = TABLES[1:]
    for i, table_port in enumerate(sp_tables):
        model_index = i  # 不同桌用不同版本，增加对手多样性
        if not saved_models:
            print(f"  Table {table_port}: 模型池为空，跳过（等待 Learner 产生 checkpoint）")
            continue
        print(f"  Table {table_port}: model_index={model_index} ({os.path.basename(saved_models[model_index]) if model_index < len(saved_models) else 'latest'})")
        for seat in (2, 3, 4):
            label = f"SP (port {table_port}, seat {seat})"
            proc = start_sp_opponent(table_port, seat, model_index=model_index)
            all_procs.append(proc)
            all_labels.append(label)
            print(f"    {label}")

    print("\n" + "=" * 60)
    print("全部启动完毕！")
    print(f"  RL 客户端: MCTS 增强 (tcli_mcts.py)")
    print(f"  桌子 1 ({TABLES[0]}): MCTS-RL + 3 TOP (强对抗，1桌)")
    print(f"  桌子 2-4 ({TABLES[1]}, {TABLES[2]}, {TABLES[3]}): MCTS-RL + 3 SelfPlay (3桌自博弈)")
    print("  按 Ctrl+C 终止所有进程")
    print("=" * 60)

    try:
        while True:
            for i, (proc, label) in enumerate(zip(all_procs, all_labels)):
                poll = proc.poll()
                if poll is not None:
                    print(f"[{label}] 退出 (code={poll})")
                    all_procs.pop(i)
                    all_labels.pop(i)
                    break
            if not all_procs:
                print("\n所有子进程已退出。")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在终止所有进程...")
        for p in all_procs:
            p.terminate()
        time.sleep(2)
        print("已全部终止。")


if __name__ == "__main__":
    main()
