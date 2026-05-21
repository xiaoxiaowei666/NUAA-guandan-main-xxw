# -*- coding: utf-8 -*-
"""
MCTS 增强版客户端 (tcli_mcts.py)
在 InferenceClient 基础上用 MCTS 搜索替代单步贪心 Q 值选择。
模式：
    reinforcement_mcts  - 强化学习 + MCTS 推理
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

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from coach import LoadCoach
from coach.TOP.safety_filter import compute_safety_mask
from state import State
from util import *
from model import ActionValueNet

# 导入 MCTS
from actor_all.mcts import MCTS


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


# ===================== Rule 模式（与 tcli.py 一致） =====================
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


# ===================== MCTS 增强推理客户端 =====================
class MCTSInferenceClient(BaseClient):
    """
    MCTS 增强推理客户端。
    与 InferenceClient 的区别：select_action 使用 MCTS 搜索，
    保留经验收集逻辑用于继续训练。
    """

    def __init__(self, url, args):
        super().__init__(url, args.render)
        self.args = args
        self.device = args.device

        # 模型（由 Learner 更新权重）
        self.model = ActionValueNet().to(self.device)
        self.model.eval()

        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.episode = 0

        # 经验收集
        self.episode_transitions = []
        self.last_obs = None
        self.last_history = None
        self.last_act = None
        self.last_action_list = None
        self.last_phi = None          # 势能塑形奖励

        # ZMQ 连接
        self.zmq_ctx = zmq.Context()
        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{args.learner_host}:{args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        self.push_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{args.learner_host}:5555")

        self.ready_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.ready_socket.connect(f"tcp://{args.learner_host}:5556")
        self.ready_socket.send(b"ready")
        print("已发送就绪信号至 Learner")

        # 后台权重监听
        self.weights_lock = threading.Lock()
        self.stop_listener = False
        self.listener_thread = threading.Thread(target=self._weights_listener, daemon=True)
        self.listener_thread.start()

        # MCTS 搜索器
        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            num_simulations=args.mcts_sims,
            max_depth=args.mcts_depth,
            c_puct=args.mcts_c_puct,
            temperature=args.mcts_temperature,
            num_determinizations=args.mcts_det,
            epsilon=args.mcts_eps,
        )
        print(f"MCTS 搜索器已初始化: sims={args.mcts_sims}, "
              f"depth={args.mcts_depth}, det={args.mcts_det}")

    # ---------- 权重监听 ----------
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
                    # 更新 MCTS 中的模型引用
                    self.mcts.model = self.model
                    print("成功更新模型权重（来自Learner广播）")
            except Exception as e:
                print(f"权重监听异常: {e}")

    # ---------- 奖励 ----------
    def get_reward(self, order):
        myPos = self.state._myPos
        friendPos = (myPos + 2) % 4
        myRank = order.index(myPos)
        friendRank = order.index(friendPos)
        myRank, friendRank = sorted((myRank, friendRank))
        reward_map = {(0, 1): 5, (0, 2): 3, (0, 3): 1,
                      (1, 2): -1, (1, 3): -3, (2, 3): -5}
        return reward_map.get((myRank, friendRank), 0)

    # ---------- 势能塑形奖励 ----------
    GAMMA_SHAPING = 0.98

    def compute_potential(self, msg):
        """
        Φ(s): 态势评估函数。
        利用 publicInfo（队友/对手剩余牌）、greaterPos（控场权）等完整状态信息。
        """
        my_pos = self.state._myPos      # 从 State 对象取，beginning 阶段已解析
        teammate_pos = (my_pos + 2) % 4
        hand_cards = msg.get('handCards', [])
        public_info = msg.get('publicInfo', [])

        phi = 0.0

        # 自己手牌进度
        phi -= len(hand_cards) * 0.06

        # 队友手牌进度
        if public_info and teammate_pos < len(public_info):
            teammate_rest = public_info[teammate_pos].get('rest', len(hand_cards))
            phi -= teammate_rest * 0.03

        # 对手手牌进度
        if public_info:
            for i in range(4):
                if i != my_pos and i != teammate_pos and i < len(public_info):
                    opp_rest = public_info[i].get('rest', 27)
                    phi += opp_rest * 0.02

        # 控场权
        greater_pos = self._get_greater_pos(msg)
        if greater_pos >= 0:
            if greater_pos == my_pos or greater_pos == teammate_pos:
                phi += 0.1
            else:
                phi -= 0.1

        return phi

    @staticmethod
    def _get_greater_pos(msg):
        gp = msg.get('greaterPos', -1)
        ga = msg.get('greaterAction', -1)
        if isinstance(gp, int) and 0 <= gp <= 3:
            return gp
        if isinstance(ga, int) and 0 <= ga <= 3:
            return ga
        return -1

    # ---------- 消息处理 ----------
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

        # 累计已出牌统计
        if msg.get("type") == "notify" and msg.get("stage") == "play":
            cur_action = msg.get("curAction")
            if cur_action:
                cards = process_card_list(cur_action)
                self.played_cards = self.played_cards + encode_card(cards)

        if "actionList" in msg:
            msg["playedCards"] = self.played_cards
            act_idx = self.select_action(msg)
            if msg.get("stage") == "play":
                chosen_action = msg["actionList"][act_idx]
                cards = process_card_list(chosen_action)
                if cards != ('PASS', 'PASS', 'PASS'):
                    self.played_cards = self.played_cards + encode_card(cards)
            self.send(json.dumps({"actIndex": act_idx}))

    # ---------- 动作选择（MCTS 核心） ----------
    def select_action(self, msg):
        action_list = msg["actionList"]
        act_range = msg["indexRange"]

        state = StateCatEmbedding(msg).to(self.device)
        history = self.MapHistoryToLSTM().float().to(self.device)

        # 打印 MCTS 搜索信息
        t_start = time.time()

        best_idx, probs = self.mcts.search(
            state_msg=msg,
            action_list=action_list,
            act_range=act_range,
            history_tensor=history,
            played_cards_tensor=self.played_cards.clone(),
            hand_cards=msg.get("handCards", []),
        )

        # ---- TOP 结构安全过滤 ----
        safety_mask = compute_safety_mask(msg, action_list, self.state._myPos)
        if not safety_mask[best_idx]:
            # MCTS 选中的动作不安全，回退到安全候选里 MCTS 访问次数最高的
            safe_indices = [i for i, (v, s) in enumerate(zip(probs.values() if hasattr(probs, 'values') else probs, safety_mask))
                          if safety_mask[i]]
            # probs is a dict {idx: prob}
            safe_probs = {i: probs.get(i, 0) for i in probs if safety_mask[i]}
            if safe_probs:
                best_idx = max(safe_probs, key=safe_probs.get)
            elif any(safety_mask):
                best_idx = next(i for i, s in enumerate(safety_mask) if s)
            # else: keep original best_idx as fallback

        elapsed = time.time() - t_start
        if self.episode % 10 == 0:
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_str = " | ".join(
                f"a{i}={action_list[i][0] if i < len(action_list) else '?'}({p:.2f})"
                for i, p in top3
            )
            print(f"[MCTS] ep={self.episode} | "
                  f"best={best_idx}({action_list[best_idx][0]}) | "
                  f"top3: {top3_str} | {elapsed:.2f}s")

        act = process_card_list(action_list[best_idx])

        # 势能塑形奖励
        phi_curr = self.compute_potential(msg)

        # 经验收集
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

        return best_idx

    # ---------- 经验处理 ----------
    PASS_PENALTY = 0.05

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
    parser = argparse.ArgumentParser(description="t-cli-mcts MCTS增强客户端")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    # Rule 模式（与 tcli.py 一致）
    rule_parser = subparsers.add_parser("rule", help="规则/自定义教练")
    rule_parser.add_argument("pos", type=int, help="座位号")
    rule_parser.add_argument("-c", "--client", default="TOP", help="教练名称")
    rule_parser.add_argument("-r", "--render", action="store_true")
    rule_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rule_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")

    # MCTS 增强强化学习模式
    rl_parser = subparsers.add_parser("reinforcement_mcts", help="强化学习 + MCTS 推理")
    rl_parser.add_argument("pos", type=int, help="座位号")
    rl_parser.add_argument("-r", "--render", action="store_true")
    rl_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rl_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    rl_parser.add_argument("--device", default="cpu", help="推理设备")
    rl_parser.add_argument("--learner_host", default="127.0.0.1", help="Learner IP")
    rl_parser.add_argument("--learner_port", type=int, default=10002, help="Learner PUB 端口")

    # MCTS 参数（已针对 CPU 批量计算优化）
    rl_parser.add_argument("--mcts_sims", type=int, default=30,
                           help="每次决策的 MCTS 模拟次数 (默认 30)")
    rl_parser.add_argument("--mcts_depth", type=int, default=3,
                           help="MCTS rollout 最大深度 (默认 3)")
    rl_parser.add_argument("--mcts_c_puct", type=float, default=1.4,
                           help="MCTS UCB 探索系数 (默认 1.4)")
    rl_parser.add_argument("--mcts_temperature", type=float, default=0.5,
                           help="先验 softmax 温度 (默认 0.5)")
    rl_parser.add_argument("--mcts_det", type=int, default=2,
                           help="Determinization 采样次数 (默认 2)")
    rl_parser.add_argument("--mcts_eps", type=float, default=0.1,
                           help="MCTS rollout 中的探索率 (默认 0.1)")

    # TD(n) 参数
    rl_parser.add_argument("--gamma", type=float, default=0.98,
                           help="折扣因子 (默认 0.98)")
    rl_parser.add_argument("--n_step", type=int, default=3,
                           help="TD 步数: 1=TD(0), 3=TD(3), -1=MC全链路/DouZero模式 (默认 3)")

    args = parser.parse_args()

    if args.mode == "rule":
        run_demo(args)
    elif args.mode == "reinforcement_mcts":
        url = f"ws://{args.host}:{args.port}/game/client{args.pos}"
        client = MCTSInferenceClient(url, args)
        try:
            client.connect()
            client.run_forever()
        except KeyboardInterrupt:
            client.close()
    else:
        print("未知模式")


if __name__ == "__main__":
    main()
