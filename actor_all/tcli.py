# -*- coding: utf-8 -*-
"""
精简版客户端入口 (tcli.py)
模式：
    rule           - 基于规则 / 自定义教练
    reinforcement  - 强化学习推理（被动接收 Learner 广播的权重，主动发送经验）
"""

import sys
import os
import json
import time
import random
import pickle
import argparse
import threading

import zmq
import numpy as np
import torch
from ws4py.client.threadedclient import WebSocketClient
from colorama import Back, Style
from clients.state import State
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach import LoadCoach
from util import *
from model import ActionValueNet


# ===================== 公共基类 =====================
class BaseClient(WebSocketClient):
    def __init__(self, url, render=False):
        super().__init__(url)
        self.state = State(render)
        self.render = render

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Connection closed", code, reason)


# ===================== Rule 模式 =====================
def run_demo(args):
    CLIENT_ARGS = {
        'url': f'ws://{args.host}:{args.port}/game/client{args.pos}',
        'render': args.render
    }
    try:
        ws = LoadCoach(args.client)(**CLIENT_ARGS)
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()


# ===================== Reinforcement 模式（推理 + 经验发送） =====================
class InferenceClient(BaseClient):
    def __init__(self, url, args):
        super().__init__(url, args.render)
        self.args = args
        self.device = args.device

        # 定义模型（与 gene_client 不同：这里只用于推理，由 Learner 更新权重）
        self.model = ActionValueNet().to(self.device)
        self.model.eval()

        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.episode = 0

        # 经验收集相关（完全复刻 gene_client.py 的 ReinforcementAction）
        self.episode_transitions = []   # 当前局的过渡存储
        self.last_obs = None
        self.last_history = None
        self.last_act = None
        self.last_action_list = None    # 保存当前步的动作列表（用于 next_step 的 actionList）

        # ZMQ SUB：被动接收 Learner 广播的权重
        self.zmq_ctx = zmq.Context()
        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{args.learner_host}:{args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # ZMQ PUSH：向 Learner 发送经验
        self.push_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{args.learner_host}:5555")   # Learner 的 PULL 端口

        # 后台监听权重
        self.weights_lock = threading.Lock()
        self.stop_listener = False
        self.listener_thread = threading.Thread(target=self._weights_listener, daemon=True)
        self.listener_thread.start()

    def _weights_listener(self):
        while not self.stop_listener:
            try:
                if self.sub_socket.poll(timeout=500):
                    msg = self.sub_socket.recv()
                    state_dict = pickle.loads(msg)
                    with self.weights_lock:  # 请求锁，保证与推理互斥
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(self.device)
                        self.model.load_state_dict(state_dict)
                    print("成功更新模型权重（来自Learner广播）")
            except Exception as e:
                print(f"权重监听异常: {e}")

    def get_reward(self, order):
        """根据排名计算奖励（和 gene_client 一致）"""
        myPos = self.state._myPos
        friendPos = (myPos + 2) % 4
        myRank = order.index(myPos)
        friendRank = order.index(friendPos)
        myRank, friendRank = sorted((myRank, friendRank))
        reward_map = {(0,1):5, (0,2):3, (0,3):1, (1,2):-1, (1,3):-3, (2,3):-5}
        return reward_map.get((myRank, friendRank), 0)

    def received_message(self, message):
        msg = json.loads(str(message))
        self.state.parse(msg)

        if msg["stage"] == "beginning":
            # 重置本局经验
            self.episode_transitions.clear()
            self.last_obs = None
            self.last_history = None
            self.last_act = None
            self.last_action_list = None
            self.history_action = [['PASS', 'PASS', 'PASS']]
            self.episode += 1

        elif msg["stage"] in ("episodeOver", "gameOver"):
            final_reward = self.get_reward(msg["order"])
            # 把最终奖励写入本局所有过渡
            self.apply_final_reward(final_reward)
            # 发送整局轨迹给 Learner
            if self.episode_transitions:
                self.send_experience()
            # 清空临时存储
            self.episode_transitions.clear()
            self.last_obs = None
            self.last_history = None
            self.last_act = None

        if "actionList" in msg:
            act_idx = self.select_action(msg)
            self.send(json.dumps({"actIndex": act_idx}))

    def select_action(self, msg):
        """与 gene_client.ReinforcementAction.parse 完全对齐的动作选择 + 经验记录"""
        action_list = msg["actionList"]
        act_range = msg["indexRange"]

        # 找到 PASS 的索引（PASS 动作列表第一个元素是 'PASS'）
        pass_idx = None
        for i, act in enumerate(action_list):
            if act[0] == 'PASS':
                pass_idx = i
                break

        # Q 值计算（使用 self.model，不是 self.ValueNet）
        state = StateCatEmbedding(msg).to(self.device)
        history = self.MapHistoryToLSTM().float().to(self.device)

        # ---------- 加锁：防止推理时权重被后台更新 ----------
        with self.weights_lock:
            q_vals = []
            with torch.no_grad():
                for i in range(self.act_range + 1):
                    act_emb = ActionEmbedding(msg, i).to(self.device)
                    inp = torch.cat((state.flatten(), act_emb)).unsqueeze(0)
                    q = self.model(inp, history).sum().item()
                    q_vals.append(q)
        # ---------- 锁释放 ----------

        # ε-贪婪选择
        if random.random() > self.args.epsilon:
            action_idx = int(np.argmax(q_vals))
        else:
            action_idx = random.randint(0, act_range)

        # 强制干预：如果选了 PASS 且还有其它合法动作（防止无限跳过）
        if action_idx == pass_idx and act_range > 0:
            if random.random() < 0.8:
                non_pass_indices = [i for i in range(act_range + 1) if i != pass_idx]
                action_idx = max(non_pass_indices, key=lambda i: q_vals[i])

        # 动作后处理（转成标准格式）
        act = process_card_list(action_list[action_idx])

        # ---------- 经验记录 ----------
        if self.last_obs is not None and self.last_history is not None:
            transition = (
                self.last_obs.cpu(),
                self.last_history.cpu(),
                self.last_act,
                0.0,                 # 中间奖励暂为0，终局时覆盖
                state.cpu(),         # next_obs
                self.last_action_list,  # 上一步的 actionList
                history.cpu(),       # next_history
                False                # 非终止
            )
            self.episode_transitions.append(transition)

        # 更新本次信息作为下一次的 last_*
        self.last_obs = state
        self.last_history = history
        self.last_act = act
        self.last_action_list = action_list   # 保存原始动作列表
        self.history_action.append(act)

        return action_idx

    def apply_final_reward(self, final_reward):
        """将最终奖励赋值给本局所有过渡，并标记最后一个 done"""
        if not self.episode_transitions:
            return
        for i, trans in enumerate(self.episode_transitions):
            t = list(trans)
            t[3] = final_reward          # 用终局奖励替换
            if i == len(self.episode_transitions) - 1:
                t[7] = True
            self.episode_transitions[i] = tuple(t)

    def send_experience(self):
        """将本局轨迹序列化并通过 PUSH 发送到 Learner"""
        data = []
        for t in self.episode_transitions:
            obs_np = t[0].numpy() if torch.is_tensor(t[0]) else t[0]
            hist_np = t[1].numpy() if torch.is_tensor(t[1]) else t[1]
            next_obs_np = t[4].numpy() if torch.is_tensor(t[4]) else t[4]
            next_hist_np = t[6].numpy() if torch.is_tensor(t[6]) else t[6]
            data.append((
                obs_np, hist_np, t[2], t[3],
                next_obs_np, t[5], next_hist_np, t[7]
            ))
        try:
            self.push_socket.send(pickle.dumps(data))
            print(f"已发送 {len(data)} 条经验至 Learner")
        except Exception as e:
            print(f"发送经验失败: {e}")

    def MapHistoryToLSTM(self):
        return torch.stack(
            [encode_card(a).flatten() for a in self.history_action], dim=0
        ).unsqueeze(0)

    def close(self):
        self.stop_listener = True
        if hasattr(self, 'listener_thread'):
            self.listener_thread.join(timeout=2)
        self.sub_socket.close()
        self.push_socket.close()
        self.zmq_ctx.term()
        super().close()


# ===================== 主入口 =====================
def main():
    parser = argparse.ArgumentParser(description="t-cli 客户端")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    # Rule 模式
    rule_parser = subparsers.add_parser("rule", help="规则/自定义教练")
    rule_parser.add_argument("pos", type=int, help="座位号")
    rule_parser.add_argument("-c", "--client", default="Demo", help="教练名称")
    rule_parser.add_argument("-r", "--render", action="store_true")
    rule_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rule_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")

    # Reinforcement 推理模式
    rl_parser = subparsers.add_parser("reinforcement", help="强化学习推理（被动接收 Learner 广播，主动发送经验）")
    rl_parser.add_argument("pos", type=int, help="座位号")
    rl_parser.add_argument("-r", "--render", action="store_true")
    rl_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rl_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    rl_parser.add_argument("--device", default="cuda", help="推理设备")
    rl_parser.add_argument("--epsilon", type=float, default=0.1, help="探索率")
    rl_parser.add_argument("--learner_host", default="127.0.0.1", help="Learner IP")
    rl_parser.add_argument("--learner_port", type=int, default=10000, help="Learner PUB 端口（用于接收权重）")

    args = parser.parse_args()

    if args.mode == "rule":
        run_demo(args)
    elif args.mode == "reinforcement":
        url = f"ws://{args.host}:{args.port}/game/client{args.pos}"
        client = InferenceClient(url, args)
        try:
            client.connect()
            client.run_forever()
        except KeyboardInterrupt:
            client.close()
    else:
        print("未知模式")


if __name__ == "__main__":
    main()