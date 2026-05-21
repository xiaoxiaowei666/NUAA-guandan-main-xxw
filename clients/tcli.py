# -*- coding: utf-8 -*-
"""
精简版客户端入口 (tcli.py)
模式：
    rule           - 基于规则 / 自定义教练
    reinforcement  - 强化学习推理（被动接收 Learner 广播的权重，主动发送经验）
                     启动后向 Learner 发送就绪消息，以便 Learner 累计连接数
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

# 注意导入路径：根据你的项目结构，可能需调整
sys.path.append(os.path.abspath('.'))
from coach import LoadCoach
from coach.TOP.safety_filter import compute_safety_mask
from state import State
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

        # 定义模型（由 Learner 更新权重）
        self.model = ActionValueNet().to(self.device)
        self.model.eval()

        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.episode = 0

        # 经验收集相关
        self.episode_transitions = []
        self.last_obs = None
        self.last_history = None
        self.last_act = None
        self.last_action_list = None
        self.last_phi = None          # 上一步状态势能，用于势能塑形奖励

        # ZMQ SUB：被动接收 Learner 广播的权重
        self.zmq_ctx = zmq.Context()
        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{args.learner_host}:{args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # ZMQ PUSH：向 Learner 发送经验（5555）
        self.push_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{args.learner_host}:5555")

        # ZMQ PUSH：向 Learner 发送就绪消息（5556）
        self.ready_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.ready_socket.connect(f"tcp://{args.learner_host}:5556")
        self.ready_socket.send(b"ready")          # 启动时发送一次
        print("已发送就绪信号至 Learner")

        # 后台监听权重
        self.weights_lock = threading.Lock()
        self.stop_listener = False
        self.listener_thread = threading.Thread(target=self._weights_listener, daemon=True)
        self.listener_thread.start()

    def _weights_listener(self):
        print("权重监听线程已启动")
        while not self.stop_listener:
            try:
                if self.sub_socket.poll(timeout=500):
                    msg = self.sub_socket.recv()
                    state_dict = pickle.loads(msg)
                    with self.weights_lock:
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(self.device)
                        self.model.load_state_dict(state_dict)
                    print("成功更新模型权重（来自Learner广播）")
            except Exception as e:
                print(f"权重监听异常: {e}")

    def get_reward(self, order):
        myPos = self.state._myPos
        friendPos = (myPos + 2) % 4
        myRank = order.index(myPos)
        friendRank = order.index(friendPos)
        myRank, friendRank = sorted((myRank, friendRank))
        reward_map = {(0,1):5, (0,2):3, (0,3):1, (1,2):-1, (1,3):-3, (2,3):-5}
        return reward_map.get((myRank, friendRank), 0)

    # ---------- 势能塑形奖励 ----------
    GAMMA_SHAPING = 0.98  # 与 Learner 端 gamma 保持一致

    def compute_potential(self, msg):
        """
        Φ(s): 态势评估函数。
        利用 publicInfo（队友/对手剩余牌）、greaterPos（控场权）等完整状态信息。
        越高 = 局势越有利。
        """
        my_pos = self.state._myPos      # 从 State 对象取，beginning 阶段已解析
        teammate_pos = (my_pos + 2) % 4
        hand_cards = msg.get('handCards', [])
        public_info = msg.get('publicInfo', [])

        phi = 0.0

        # 1. 自己手牌进度：剩得越少越好
        phi -= len(hand_cards) * 0.06

        # 2. 队友手牌进度：队友剩得越少越好（权重为自己的一半）
        if public_info and teammate_pos < len(public_info):
            teammate_rest = public_info[teammate_pos].get('rest', len(hand_cards))
            phi -= teammate_rest * 0.03

        # 3. 对手手牌进度：对手剩得多对我们有利
        if public_info:
            for i in range(4):
                if i != my_pos and i != teammate_pos and i < len(public_info):
                    opp_rest = public_info[i].get('rest', 27)
                    phi += opp_rest * 0.02

        # 4. 控场权：我方或队友控场优于对手控场
        greater_pos = self._get_greater_pos(msg)
        if greater_pos >= 0:
            if greater_pos == my_pos or greater_pos == teammate_pos:
                phi += 0.1   # 我方控场
            else:
                phi -= 0.1   # 对手控场
        # greaterPos == -1：自由出牌 / 进贡 / 还贡，不加不减

        return phi

    @staticmethod
    def _get_greater_pos(msg):
        """从消息中提取当前最大牌的出牌者位置"""
        gp = msg.get('greaterPos', -1)
        ga = msg.get('greaterAction', -1)
        if isinstance(gp, int) and 0 <= gp <= 3:
            return gp
        if isinstance(ga, int) and 0 <= ga <= 3:
            return ga
        return -1

    def received_message(self, message):
        msg = json.loads(str(message))
        self.state.parse(msg)
        msg.setdefault("myPos", self.state._myPos)

        if msg["stage"] == "beginning":
            self.episode_transitions.clear()
            self.last_obs = None
            self.last_history = None
            self.last_act = None
            self.last_action_list = None
            self.last_phi = None
            self.history_action = [['PASS', 'PASS', 'PASS']]
            self.played_cards = torch.zeros(4, 15, dtype=torch.long)
            self.episode += 1

        elif msg["stage"] == "episodeOver":
            final_reward = self.get_reward(msg["order"])
            self.apply_final_reward(final_reward)
            if self.episode_transitions:
                self.send_experience()
            self.episode_transitions.clear()
            self.last_obs = None
            self.last_history = None
            self.last_act = None
            self.last_phi = None

        # 收到 notify/play 时累计已出牌统计（别人出牌）
        if msg.get("type") == "notify" and msg.get("stage") == "play":
            cur_action = msg.get("curAction")
            if cur_action:
                cards = process_card_list(cur_action)
                self.played_cards = self.played_cards + encode_card(cards)

        if "actionList" in msg:
            msg["playedCards"] = self.played_cards
            act_idx = self.select_action(msg)
            # 自己出牌也计入已出牌统计（进贡/还贡除外）
            if msg.get("stage") == "play":
                chosen_action = msg["actionList"][act_idx]
                cards = process_card_list(chosen_action)
                if cards != ('PASS', 'PASS', 'PASS'):
                    self.played_cards = self.played_cards + encode_card(cards)
            self.send(json.dumps({"actIndex": act_idx}))

    def select_action(self, msg):
        action_list = msg["actionList"]
        act_range = msg["indexRange"]

        # 查找 PASS 索引
        pass_idx = None
        for i, act in enumerate(action_list):
            if act[0] == 'PASS':
                pass_idx = i
                break

        state = StateCatEmbedding(msg).to(self.device)
        history = self.MapHistoryToLSTM().float().to(self.device)

        q_vals = []
        with torch.no_grad():
            for i in range(act_range + 1):
                act_emb = ActionEmbedding(msg, i).to(self.device)
                inp = torch.cat((state.flatten(), act_emb)).unsqueeze(0)
                q = self.model(inp, history).sum().item()
                q_vals.append(q)

        # ---- TOP 结构安全过滤 ----
        safety_mask = compute_safety_mask(msg, action_list, self.state._myPos)

        if random.random() > self.args.epsilon:
            # Greedy: 在安全候选里选最优，同时禁止 PASS 崩塌
            safe_non_pass = [i for i in range(act_range + 1)
                             if safety_mask[i] and i != pass_idx]
            # 如果所有非 PASS 都被过滤了，回退到不过滤
            if not safe_non_pass and act_range > 0:
                safe_non_pass = [i for i in range(act_range + 1) if i != pass_idx]
            if safe_non_pass:
                action_idx = max(safe_non_pass, key=lambda i: q_vals[i])
            else:
                action_idx = int(np.argmax(q_vals))
        else:
            # Exploration: 只在安全候选里随机选
            safe_indices = [i for i in range(act_range + 1) if safety_mask[i]]
            if not safe_indices:
                safe_indices = list(range(act_range + 1))
            action_idx = random.choice(safe_indices)

        act = process_card_list(action_list[action_idx])

        # 势能塑形奖励: F(s,s') = γΦ(s') - Φ(s)
        phi_curr = self.compute_potential(msg)

        # 记录经验
        if self.last_obs is not None and self.last_history is not None:
            shaping_r = 0.0
            if self.last_phi is not None:
                shaping_r = self.GAMMA_SHAPING * phi_curr - self.last_phi
            transition = (
                self.last_obs.cpu(),
                self.last_history.cpu(),
                self.last_act,
                shaping_r,
                state.cpu(),
                action_list,
                history.cpu(),
                False
            )
            self.episode_transitions.append(transition)

        self.last_obs = state
        self.last_history = history
        self.last_act = act
        self.last_action_list = action_list
        self.last_phi = phi_curr
        self.history_action.append(act)

        return action_idx

    PASS_PENALTY = 0.05  # PASS 动作的微小惩罚

    def apply_final_reward(self, final_reward):
        """MC 模式：每一步直接吃终局奖励 + 势能塑形奖励，全部 done=True"""
        if not self.episode_transitions:
            return
        for i, trans in enumerate(self.episode_transitions):
            t = list(trans)
            t[3] = t[3] + final_reward - (self.PASS_PENALTY if t[2][0] == 'PASS' else 0.0)
            t[7] = True
            self.episode_transitions[i] = tuple(t)

    def send_experience(self):
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
        # 后台线程发送，避免阻塞 WebSocket 回调导致整桌冻结
        threading.Thread(
            target=self._send_experience_async, args=(data,), daemon=True
        ).start()

    def _send_experience_async(self, data):
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
        self.ready_socket.close()
        self.zmq_ctx.term()
        super().close()


# ===================== 主入口 =====================
def main():
    parser = argparse.ArgumentParser(description="t-cli 客户端")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    # Rule 模式
    rule_parser = subparsers.add_parser("rule", help="规则/自定义教练")
    rule_parser.add_argument("pos", type=int, help="座位号")
    rule_parser.add_argument("-c", "--client", default="TOP", help="教练名称")
    rule_parser.add_argument("-r", "--render", action="store_true")
    rule_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rule_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")

    # Reinforcement 推理模式
    rl_parser = subparsers.add_parser("reinforcement", help="强化学习推理（被动接收 Learner 广播，主动发送经验）")
    rl_parser.add_argument("pos", type=int, help="座位号")
    rl_parser.add_argument("-r", "--render", action="store_true")
    rl_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rl_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    rl_parser.add_argument("--device", default="cpu", help="推理设备")
    rl_parser.add_argument("--epsilon", type=float, default=0.25, help="探索率")
    rl_parser.add_argument("--learner_host", default="127.0.0.1", help="Learner IP")
    rl_parser.add_argument("--learner_port", type=int, default=10002, help="Learner PUB 端口（用于接收权重）")

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