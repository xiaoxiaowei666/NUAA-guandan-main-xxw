# -*- coding: utf-8 -*-
"""
分布式模仿学习客户端 (tcli_imitation.py)
模式: imitation_dist  — 分布式 DAgger：接收 Learner 广播权重，发送专家样本

说明：
- 模型初始化为随机权重，后续完全由 Learner 通过 PUB 广播更新，客户端不加载任何本地模型文件。
- 专家策略通过 --expert 参数指定，动态加载 coach/<专家名>/action.py 中的 Action 类（需提供 parse_AI 方法）。
- 客户端连接游戏服务器，根据 DAgger 策略选择动作，并收集 (状态, 历史, 动作嵌入列表, 专家动作索引) 样本，每局结束发送给 Learner。

用法示例：
    python tcli_imitation.py imitation_dist 1 --learner_host 127.0.0.1 --learner_port 10003 --expert EggPan
"""

import sys
import os
import json
import random
import pickle
import argparse
import threading
import importlib

import zmq
import numpy as np
import torch
from ws4py.client.threadedclient import WebSocketClient
from colorama import Back, Style

sys.path.append(os.path.abspath('.'))
from state import State
from util import *
from model import ActionValueNet


class ImitationAction:
    """
    负责模型预测、专家查询、样本收集，不包含训练。
    """
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # 模型：随机初始化，权重由 Learner 广播
        self.model = ActionValueNet().to(self.device)
        self.model.eval()

        # 游戏状态
        self.action = []
        self.act_range = -1
        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.count = 0

        # 加载专家策略（只取 Action 类，不需要 WebSocket 连接）
        self.expert = self._load_expert(args.expert)

        # DAgger 概率
        self.use_expert_prob = args.expert_init

        # 当前局的样本缓冲
        self.buffer = []

    def _load_expert(self, expert_name):
        """
        动态加载 coach.<专家名>.action 模块中的 Action 类。
        专家类必须实现方法：parse_AI(msg, myPos) -> int （动作索引）
        """
        if expert_name == "EggPan":
            from coach.EggPan.action import Action
            return Action(render=False)
        else:
            try:
                mod = importlib.import_module(f"coach.{expert_name}.action")
                # 假设模块有一个 Action 类
                return mod.Action(render=False)
            except Exception as e:
                raise RuntimeError(
                    f"无法加载专家 '{expert_name}'，请确保 coach/{expert_name}/action.py 存在并包含 Action 类。错误: {e}"
                )

    def MapHistoryToLSTM(self):
        ret = torch.stack([encode_card(a).flatten() for a in self.history_action], dim=0)
        ret = ret.unsqueeze(0)
        return ret

    def add_to_buffer(self, msg, expert_idx):
        """将当前状态、历史、动作嵌入及专家索引加入缓冲（转为numpy以备发送）"""
        state_tensor = StateCatEmbedding(msg).cpu()
        history_tensor = self.MapHistoryToLSTM().cpu().float()
        action_embs = [
            encode_card(process_card_list(msg["actionList"][i])).flatten().cpu()
            for i in range(msg["indexRange"] + 1)
        ]
        self.buffer.append((
            state_tensor.numpy(),
            history_tensor.numpy(),
            [emb.numpy() for emb in action_embs],
            expert_idx
        ))

    def select_action_by_model(self, msg):
        """用当前模型进行 softmax 概率采样"""
        state = StateCatEmbedding(msg).to(self.device)
        history = self.MapHistoryToLSTM().float().to(self.device)
        q_vals = []
        with torch.no_grad():
            for i in range(msg["indexRange"] + 1):
                emb = encode_card(process_card_list(msg["actionList"][i])).flatten().to(self.device)
                inp = torch.cat((state.flatten(), emb), dim=0).unsqueeze(0)
                q = self.model(inp, history).sum()
                q_vals.append(q)
        q_tensor = torch.stack(q_vals)
        probs = torch.softmax(q_tensor, dim=0).detach().cpu().numpy()
        return np.random.choice(len(q_vals), p=probs)

    def parse(self, msg, render=False):
        """每步决策，并记录专家样本"""
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        if render:
            print(Back.BLUE, f"可选动作范围: 0 至 {self.act_range}", Style.RESET_ALL)

        # 专家动作
        expert_idx = self.expert.parse_AI(msg, msg.get("myPos", 0))
        expert_idx = np.clip(expert_idx, 0, self.act_range).tolist()

        # 记录样本
        self.add_to_buffer(msg, expert_idx)

        # 混合策略
        if random.random() < self.use_expert_prob:
            index = expert_idx
        else:
            index = self.select_action_by_model(msg)
            index = min(index, self.act_range)

        # 更新历史
        self.history_action.append(process_card_list(msg["actionList"][index]))
        self.count += 1
        return index

    def decay_expert_prob(self):
        self.use_expert_prob = max(
            self.args.min_expert_prob,
            self.use_expert_prob * self.args.expert_decay
        )

    def reset_episode(self):
        self.buffer.clear()
        self.history_action = [['PASS', 'PASS', 'PASS']]

    def load_weights(self, state_dict):
        """从 Learner 接收新权重并加载到模型"""
        with torch.no_grad():
            # 将权重转移到当前设备
            for k, v in state_dict.items():
                state_dict[k] = v.to(self.device)
            self.model.load_state_dict(state_dict)


class ImitationDistClient(WebSocketClient):
    """WebSocket 客户端，封装游戏通信与 ZMQ 连接"""
    def __init__(self, url, args):
        super().__init__(url)
        self.args = args
        self.state = State(args.render)
        self.render = args.render
        self.episode = 0

        self.action = ImitationAction(args)

        # ZMQ 上下文
        self.zmq_ctx = zmq.Context()
        # SUB 接收权重
        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{args.learner_host}:{args.learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        # PUSH 发送专家样本 (5557)
        self.push_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{args.learner_host}:5557")
        # PUSH 就绪信号 (5558)
        self.ready_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.ready_socket.connect(f"tcp://{args.learner_host}:5558")
        self.ready_socket.send(b"ready")
        print("已发送就绪信号至 Learner")

        # 权重监听线程
        self.stop_listener = False
        self.listener_thread = threading.Thread(target=self._weights_listener, daemon=True)
        self.listener_thread.start()

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Connection closed", code, reason)

    def _weights_listener(self):
        print("权重监听线程已启动")
        while not self.stop_listener:
            try:
                if self.sub_socket.poll(timeout=500):
                    msg = self.sub_socket.recv()
                    state_dict = pickle.loads(msg)
                    self.action.load_weights(state_dict)
            except Exception as e:
                print(f"权重监听异常: {e}")

    def received_message(self, message):
        msg = json.loads(str(message))
        self.state.parse(msg)

        if msg["stage"] == "beginning":
            self.action.reset_episode()
            self.episode += 1

        elif msg["stage"] == "episodeOver":
            if self.action.buffer:
                self.send_expert_samples()
            self.action.reset_episode()
            self.action.decay_expert_prob()

        if "actionList" in msg:
            act_idx = self.action.parse(msg, self.render)
            self.send(json.dumps({"actIndex": act_idx}))

    def send_expert_samples(self):
        try:
            self.push_socket.send(pickle.dumps(self.action.buffer))
            print(f"已发送 {len(self.action.buffer)} 条专家样本")
        except Exception as e:
            print(f"发送失败: {e}")

    def close(self):
        self.stop_listener = True
        if hasattr(self, 'listener_thread'):
            self.listener_thread.join(timeout=2)
        self.sub_socket.close()
        self.push_socket.close()
        self.ready_socket.close()
        self.zmq_ctx.term()
        super().close()


def main():
    parser = argparse.ArgumentParser(description="分布式模仿学习客户端")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    im_parser = subparsers.add_parser("imitation_dist", help="分布式DAgger客户端")
    im_parser.add_argument("pos", type=int, help="座位号")
    im_parser.add_argument("-r", "--render", action="store_true")
    im_parser.add_argument("--host", default="127.0.0.1")
    im_parser.add_argument("--port", type=int, default=23456)
    im_parser.add_argument("--device", default="cuda")
    im_parser.add_argument("--learner_host", default="127.0.0.1")
    im_parser.add_argument("--learner_port", type=int, default=10003)
    im_parser.add_argument("--expert_init", type=float, default=1.0, help="初始专家概率")
    im_parser.add_argument("--expert_decay", type=float, default=0.995, help="专家概率衰减因子")
    im_parser.add_argument("--min_expert_prob", type=float, default=0.1, help="最低专家概率")
    im_parser.add_argument("--expert", default="EggPan", help="专家教练名称（对应 coach/<Name>/action.py）")

    args = parser.parse_args()

    if args.mode == "imitation_dist":
        url = f"ws://{args.host}:{args.port}/game/client{args.pos}"
        client = ImitationDistClient(url, args)
        try:
            client.connect()
            client.run_forever()
        except KeyboardInterrupt:
            client.close()
    else:
        print("未知模式")

if __name__ == "__main__":
    main()