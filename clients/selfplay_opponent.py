# -*- coding: utf-8 -*-
"""
自博弈对手客户端 — 加载固定模型 checkpoint，确定性打牌，不学习、不发送经验。
用作 RL 训练的陪练对手。

用法:
    python clients/selfplay_opponent.py <model_path> <seat> --host 127.0.0.1 --port 23456 --device cpu
"""

import sys
import os
import json
import argparse
import random
import glob
import traceback

import numpy as np
import torch
from ws4py.client.threadedclient import WebSocketClient

sys.path.append(os.path.abspath('.'))
from state import State
from util import StateCatEmbedding, ActionEmbedding, encode_card, process_card_list
from model import ActionValueNet


class SelfPlayOpponent(WebSocketClient):
    """加载固定模型 checkpoint，纯推理打牌，无 ZMQ、无学习。
    每局开始时自动扫描模型池，加载最新匹配模型（支持热切换）。
    """

    MODEL_DIR = "model/selfplay_checkpoints"

    def __init__(self, url, model_index=0, device='cpu', epsilon=0.0, render=False):
        super().__init__(url)
        self.state = State(render)
        self.render = render
        self.device = device
        self.epsilon = epsilon

        self.model_index = model_index
        self.current_model_path = None
        self.model = None

        self._reload_model()

        self.history_action = [['PASS', 'PASS', 'PASS']]

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("连接关闭", code, reason)

    def _reload_model(self):
        """扫描模型池，按 model_index 选择模型，仅在路径变化时重新加载。"""
        files = glob.glob(os.path.join(self.MODEL_DIR, "*.pth"))
        if not files:
            if self.model is None:
                raise FileNotFoundError(f"模型池为空: {self.MODEL_DIR}")
            return
        # 按编号从新到旧排序
        def extract_num(p):
            try:
                return int(os.path.basename(p).replace(".pth", ""))
            except Exception:
                return 0
        files.sort(key=extract_num, reverse=True)
        idx = min(self.model_index, len(files) - 1)
        chosen = files[idx]

        if chosen == self.current_model_path:
            return  # 模型没变，跳过

        ckpt = torch.load(chosen, map_location=self.device)
        model_class = ckpt.get("model_class", ActionValueNet)
        if self.model is None:
            self.model = model_class().to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.current_model_path = chosen
        print(f"[热加载] 模型: {os.path.basename(chosen)}")

    def received_message(self, message):
        try:
            msg = json.loads(str(message))
            self.state.parse(msg)
            msg.setdefault("myPos", self.state._myPos)

            if msg["stage"] == "beginning":
                self.history_action = [['PASS', 'PASS', 'PASS']]
                self.played_cards = torch.zeros(4, 15, dtype=torch.long)
                self._reload_model()

            if msg.get("type") == "notify" and msg.get("stage") == "play":
                cur_action = msg.get("curAction")
                if cur_action:
                    cards = process_card_list(cur_action)
                    self.played_cards = self.played_cards + encode_card(cards)

            if "actionList" in msg:
                msg["playedCards"] = self.played_cards
                act_idx = self._select_action(msg)
                if msg.get("stage") == "play":
                    chosen_action = msg["actionList"][act_idx]
                    cards = process_card_list(chosen_action)
                    if cards != ('PASS', 'PASS', 'PASS'):
                        self.played_cards = self.played_cards + encode_card(cards)
                self.send(json.dumps({"actIndex": act_idx}))
        except Exception:
            traceback.print_exc()
            raise

    def _select_action(self, msg):
        action_list = msg["actionList"]
        act_range = msg["indexRange"]

        pass_idx = None
        for i, act in enumerate(action_list):
            if act[0] == 'PASS':
                pass_idx = i
                break

        state = StateCatEmbedding(msg).to(self.device)
        history = self._map_history().float().to(self.device)

        q_vals = []
        with torch.no_grad():
            for i in range(act_range + 1):
                act_emb = ActionEmbedding(msg, i).to(self.device)
                inp = torch.cat((state.flatten(), act_emb)).unsqueeze(0)
                q = self.model(inp, history).sum().item()
                q_vals.append(q)

        if random.random() > self.epsilon:
            action_idx = int(np.argmax(q_vals))
        else:
            action_idx = random.randint(0, act_range)

        if action_idx == pass_idx and act_range > 0:
            non_pass = [i for i in range(act_range + 1) if i != pass_idx]
            non_pass_qs = [q_vals[i] for i in non_pass]
            action_idx = non_pass[int(np.argmax(non_pass_qs))]

        act = process_card_list(action_list[action_idx])
        self.history_action.append(act)
        return action_idx

    def _map_history(self):
        return torch.stack(
            [encode_card(a).flatten() for a in self.history_action], dim=0
        ).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="自博弈对手客户端（每局自动热加载最新模型）")
    parser.add_argument("seat", type=int, help="座位号 (1-4)")
    parser.add_argument("--model_index", type=int, default=0, help="模型选择序号 (0=最新, 1=第二新, ...)")
    parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--epsilon", type=float, default=0.0, help="探索率 (默认 0 = 纯确定性)")
    parser.add_argument("-r", "--render", action="store_true", help="渲染画面")
    args = parser.parse_args()

    url = f"ws://{args.host}:{args.port}/game/client{args.seat}"
    client = SelfPlayOpponent(url, args.model_index, args.device, args.epsilon, args.render)
    try:
        client.connect()
        client.run_forever()
    except KeyboardInterrupt:
        client.close()


if __name__ == "__main__":
    main()
