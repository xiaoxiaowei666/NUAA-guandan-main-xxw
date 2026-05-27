# -*- coding: utf-8 -*-
"""
统一游戏客户端入口（gene_client）
支持模式：
    demo          基于规则的演示或自定义教练
    imitation     模仿学习（DAgger）
    reinforcement 强化学习（DQN / DouZero-MC）
    test          加载模型进行测试
"""

import sys
import os
import json
import time
import random
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ws4py.client.threadedclient import WebSocketClient
from colorama import Back, Style

sys.path.append(os.path.abspath('.'))
from coach import LoadCoach
from coach.TOP.candidate_filter import compute_candidate_list
from coach.TOP.action import Action as TOPAction
from coach.TOP.state import State
from util import *
from model import ActionValueNet
from coach.EggPan.action import Action


# ===================== 工具函数 =====================
def now_str():
    """返回形如 2026_4_27_17_6_24 的时间字符串（无前导零）"""
    now = datetime.now()
    return f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")


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


# ===================== Demo 模式 =====================
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


# ===================== Imitation 模式 =====================
class DaggerDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        state, action_embs, hist_actions, expert_idx = self.data[idx]
        return state, action_embs, hist_actions, expert_idx

class ImitationClient(BaseClient):
    def __init__(self, url, args, check_path):
        super().__init__(url, args.render)
        self.args = args
        self.check_path = check_path
        self.episode = 0
        self.action = ImitationAction(args, check_path)

    def received_message(self, message):
        msg = json.loads(str(message))
        self.state.parse(msg)
        msg.setdefault("myPos", self.state._myPos)

        if msg.get("stage") == "beginning":
            self.action.history_action = [['PASS', 'PASS', 'PASS']]
            self.played_cards = torch.zeros(4, 15, dtype=torch.long)
            self.episode += 1

        elif msg.get("stage") in ("episodeOver", "gameOver"):
            if self.episode % self.args.dagger_interval == 0:
                print(Back.YELLOW +
                      f"[Episode {self.episode}] 全量训练, 数据集: {len(self.action.dataset)}" +
                      Style.RESET_ALL)
                self.action.train_from_dataset(epochs=self.args.dagger_epochs)
            self.action.use_expert_prob = max(self.args.min_expert_prob,
                                              self.action.use_expert_prob * self.args.expert_decay)

        if msg.get("type") == "notify" and msg.get("stage") == "play":
            cur_action = msg.get("curAction")
            if cur_action:
                cards = process_card_list(cur_action)
                self.played_cards = self.played_cards + encode_card(cards)

        if "actionList" in msg:
            msg["playedCards"] = getattr(self, "played_cards", torch.zeros(4, 15, dtype=torch.long))
            act_idx = self.action.parse(msg, self.render)
            if msg.get("stage") == "play":
                chosen_action = msg["actionList"][act_idx]
                cards = process_card_list(chosen_action)
                if cards != ('PASS', 'PASS', 'PASS'):
                    self.played_cards = self.played_cards + encode_card(cards)
            self.send(json.dumps({"actIndex": act_idx}))


class ImitationAction:
    def __init__(self, args, check_path):
        self.args = args
        self.check_path = check_path
        self.action = []
        self.act_range = -1
        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.count = 0
        self.dataset = []
        self.max_dataset_size = 100000
        self.use_expert_prob = args.expert_init

        # ----- 模型加载 -----
        if args.model and args.model != "None":
            check_path(args.model)
            self.state_dict = torch.load(args.model)
        else:
            self.state_dict = {}

        model_class = self.state_dict.get("model_class", ActionValueNet)
        self.ValueNet = model_class().to(args.device)
        if "model_state_dict" in self.state_dict:
            self.ValueNet.load_state_dict(self.state_dict["model_state_dict"])
            print(Back.GREEN, f"成功载入模型: {args.model}", Style.RESET_ALL)
            time.sleep(2)

        self.optimizer = torch.optim.Adam(self.ValueNet.parameters(), lr=args.lr)
        self.eggpan_action = Action(render=args.render)

        # ----- 日志文件 -----
        self.log_file = os.path.join(check_path, "value.log")

    def write_log(self, message):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def MapHistoryToLSTM(self):
        ret = torch.stack([encode_card(a).flatten() for a in self.history_action], dim=0).unsqueeze(0)
        return ret

    def add_to_dataset(self, msg, expert_idx):
        # PASS 过滤：可选少则跳过；多选时专家选PASS仅10%概率加入
        if msg["indexRange"] <= 1:
            return
        if msg["actionList"][expert_idx][0] == 'PASS' and random.random() > 0.1:
            return

        state = StateCatEmbedding(msg).cpu()
        action_embs = [encode_card(process_card_list(msg["actionList"][i])).flatten().cpu()
                       for i in range(len(msg["actionList"]))]
        hist = [list(a) for a in self.history_action]
        self.dataset.append((state, action_embs, hist, expert_idx))
        if len(self.dataset) > self.max_dataset_size:
            self.dataset.pop(0)

    def select_action_by_model(self, msg):
        state = StateCatEmbedding(msg).to(self.args.device)
        history = self.MapHistoryToLSTM().float().to(self.args.device)
        q_vals = []
        for i in range(len(msg["actionList"])):
            emb = encode_card(process_card_list(msg["actionList"][i])).flatten().to(self.args.device)
            inp = torch.cat((state.flatten(), emb), dim=0).unsqueeze(0)
            q = self.ValueNet(inp, history).sum()
            q_vals.append(q)
        probs = torch.softmax(torch.stack(q_vals), dim=0).detach().cpu().numpy()
        return np.random.choice(len(msg["actionList"]), p=probs)

    def parse(self, msg, render=True):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        expert_idx = self.eggpan_action.parse_AI(msg, msg.get("myPos", 0))
        expert_idx = np.clip(expert_idx, 0, self.act_range).tolist()
        self.add_to_dataset(msg, expert_idx)

        if random.random() < self.use_expert_prob:
            index = expert_idx
        else:
            index = self.select_action_by_model(msg)
            index = min(index, self.act_range)

        self.history_action.append(process_card_list(msg["actionList"][index]))

        if self.count % self.args.log_interval == 0:
            self.write_log(f"[count={self.count}] expert_prob={self.use_expert_prob:.4f}, dataset_size={len(self.dataset)}")

        self.count += 1
        return index

    def train_from_dataset(self, epochs=3):
        if len(self.dataset) < self.args.min_dataset_size:
            return
        self.ValueNet.train()
        processed = []
        for state_t, action_embs, hist_actions, expert_idx in self.dataset:
            state_t = state_t.to(self.args.device)
            hist_tensor = torch.stack([encode_card(a).flatten() for a in hist_actions], dim=0).unsqueeze(0).float().to(self.args.device)
            action_embs_dev = [emb.to(self.args.device) for emb in action_embs]
            processed.append((state_t, action_embs_dev, hist_tensor, expert_idx))

        total_loss = 0.0
        for epoch in range(epochs):
            random.shuffle(processed)
            for start in range(0, len(processed), self.args.batch_size):
                batch = processed[start:start+self.args.batch_size]
                self.optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
                for state_t, action_embs, hist_t, expert_idx in batch:
                    q_vals = []
                    for emb in action_embs:
                        inp = torch.cat((state_t.flatten(), emb), dim=0).unsqueeze(0)
                        q = self.ValueNet(inp, hist_t).sum()
                        q_vals.append(q)
                    probs = torch.softmax(torch.stack(q_vals), dim=0)
                    loss = -torch.log(probs[expert_idx] + 1e-8)
                    batch_loss = batch_loss + loss
                    total_loss += loss.item()
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ValueNet.parameters(), 1.0)
                self.optimizer.step()

        avg_loss = total_loss / len(processed) if processed else 0.0
        cur_val = np.exp(-avg_loss) if avg_loss < 20 else 0.0

        save_path = os.path.join(self.check_path, f"imitation_{now_str()}_step{self.count}_prob{cur_val:.3f}.pth")
        torch.save({
            "coach": "imitation",
            "mode": "imitation",
            "model_state_dict": self.ValueNet.state_dict(),
            "model_class": ActionValueNet
        }, save_path)
        print(Back.GREEN + f"模仿学习模型已保存: {save_path}" + Style.RESET_ALL)

        self.ValueNet.eval()


# ===================== Reinforcement 模式 =====================
class ReinforcementClient(BaseClient):
    def __init__(self, url, args, check_path):
        super().__init__(url, args.render)
        self.args = args
        self.check_path = check_path
        self.action = ReinforcementAction(args)
        self.episode = 0
        self.log_file = os.path.join(check_path, "value.log")

    def write_log(self, message):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def get_reward(self, order):
        myPos = self.state._myPos
        friendPos = (myPos + 2) % 4
        myRank = order.index(myPos)
        friendRank = order.index(friendPos)
        myRank, friendRank = sorted((myRank, friendRank))
        score = (myRank, friendRank)
        reward_map = {(0,1):5, (0,2):3, (0,3):1, (1,2):-1, (1,3):-3, (2,3):-5}
        return reward_map.get(score, 0)

    def received_message(self, message):
        msg = json.loads(str(message))
        self.state.parse(msg)
        msg.setdefault("myPos", self.state._myPos)

        if msg["stage"] == "beginning":
            self.action.reset_episode()
            self.played_cards = torch.zeros(4, 15, dtype=torch.long)
            self.episode += 1

        elif msg["stage"] == "episodeOver":
            final_reward = self.get_reward(msg["order"])
            self.action.apply_final_reward(final_reward)

            # 训练
            if len(self.action.replay_memory) >= self.args.batch_size:
                all_losses = []
                last_batch_reward = 0.0
                for _ in range(self.args.update_steps):
                    batch = self.action.replay_memory.sample(self.args.batch_size)
                    losses = self.action.replay_memory.learn_batch(
                        batch,
                        self.action.ValueNet,
                        None,  # [纯MC] target_net 已废弃
                        self.action.optimizer,
                        self.args.gamma,
                        self.args.device
                    )
                    all_losses.extend(losses)
                    batch_rewards = [t[3] for t in batch]
                    last_batch_reward = sum(batch_rewards) / len(batch_rewards)

                if self.episode % self.args.log_interval == 0:
                    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
                    self.write_log(
                        f"[episode={self.episode}] avg_loss={avg_loss:.4f} "
                        f"reward_sample={last_batch_reward:.1f}"
                    )

            self.action.decay_epsilon()

            if self.episode % self.args.save_interval == 0:
                save_path = os.path.join(
                    self.check_path,
                    f"rl_{now_str()}_ep{self.episode}.pth"
                )
                torch.save({
                    "coach": "reinforcement",
                    "mode": "reinforcement",
                    "model_state_dict": self.action.ValueNet.state_dict(),
                    "model_class": ActionValueNet
                }, save_path)
                print(Back.GREEN + f"强化学习模型已保存: {save_path}" + Style.RESET_ALL)

            # [纯MC] target_net 同步已废弃
            # if self.episode % self.args.target_update_freq == 0:
            #     self.action.sync_target_network()

        if msg.get("type") == "notify" and msg.get("stage") == "play":
            cur_action = msg.get("curAction")
            if cur_action:
                cards = process_card_list(cur_action)
                self.played_cards = self.played_cards + encode_card(cards)

        if "actionList" in msg:
            msg["playedCards"] = getattr(self, "played_cards", torch.zeros(4, 15, dtype=torch.long))
            act_idx = self.action.parse(msg, self.render, state=self.state)
            if msg.get("stage") == "play":
                chosen_action = msg["actionList"][act_idx]
                cards = process_card_list(chosen_action)
                if cards != ('PASS', 'PASS', 'PASS'):
                    self.played_cards = self.played_cards + encode_card(cards)
            self.send(json.dumps({"actIndex": act_idx}))


class ReinforcementAction:
    def __init__(self, args):
        self.args = args
        self.action = []
        self.act_range = -1
        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.replay_memory = MemoryBuffer(capacity=args.replay_capacity)

        # TOP 专家（辅助决策）
        self.top_expert = TOPAction(render=False)

        # 本局暂存
        self.episode_transitions = []
        self.last_obs = None
        self.last_history = None
        self.last_act = None

        # 加载或创建网络
        if args.model and args.model != "None":
            check_path(args.model)
            state_dict = torch.load(args.model)
        else:
            state_dict = {}
        model_class = state_dict.get("model_class", ActionValueNet)
        self.ValueNet = model_class().to(args.device)
        if "model_state_dict" in state_dict:
            self.ValueNet.load_state_dict(state_dict["model_state_dict"])

        # [纯MC] target_net 已废弃 — 纯 MC 不需要目标网络
        # self.target_net = model_class().to(args.device)
        # self.target_net.load_state_dict(self.ValueNet.state_dict())
        # self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.ValueNet.parameters(), lr=args.lr)
        self.step_count = 0

    def MapHistoryToLSTM(self):
        return torch.stack(
            [encode_card(a).flatten() for a in self.history_action], dim=0
        ).unsqueeze(0)

    def reset_episode(self):
        self.episode_transitions.clear()
        self.last_obs = None
        self.last_history = None
        self.last_act = None
        self.history_action = [['PASS', 'PASS', 'PASS']]

    PASS_PENALTY = 0.05

    def apply_final_reward(self, final_reward):
        """纯 MC：整局所有步的目标统一为终局奖励。"""
        if not self.episode_transitions:
            return
        for i in range(len(self.episode_transitions)):
            t = list(self.episode_transitions[i])
            target = final_reward
            if t[2][0] == 'PASS':
                target -= self.PASS_PENALTY
            t[3] = target
            t[7] = True
            self.replay_memory.append(tuple(t))

    # [纯MC] sync_target_network 已废弃 — 纯 MC 不需要目标网络
    # def sync_target_network(self):
    #     self.target_net.load_state_dict(self.ValueNet.state_dict())

    def decay_epsilon(self):
        """每局结束后衰减 epsilon"""
        self.args.epsilon = max(0.1, self.args.epsilon * self.args.epsilon_decay)

    def parse(self, msg, render=True, state=None):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]

        pass_idx = None
        for i, act in enumerate(self.action):
            if act[0] == 'PASS':
                pass_idx = i
                break

        # 1. TOP 专家意见
        try:
            top_idx = self.top_expert.parse_AI(msg, msg.get('myPos', 0), state)
            top_idx = min(top_idx, self.act_range)
        except Exception:
            top_idx = -1

        # 2. V14.0 候选过滤（TOP VIP 通道 + 全局 state）
        try:
            candidates = compute_candidate_list(
                msg, self.action, msg['myPos'], state=state, top_idx=top_idx
            )
        except Exception:
            candidates = list(range(self.act_range + 1))

        # 3. Batch 并行计算候选 Q 值
        state = StateCatEmbedding(msg)
        history = self.MapHistoryToLSTM().float().to(self.args.device)
        q_vals = [float('-inf')] * (self.act_range + 1)
        if candidates:
            state_flat = state.flatten().to(self.args.device)
            act_embs = torch.stack([ActionEmbedding(msg, i).to(self.args.device) for i in candidates])
            inp_batch = torch.cat((
                state_flat.unsqueeze(0).expand(len(candidates), -1),
                act_embs
            ), dim=1)
            hist_batch = history.expand(len(candidates), -1, -1)
            with torch.no_grad():
                q_batch = self.ValueNet(inp_batch, hist_batch).sum(dim=1)
            for idx, i in enumerate(candidates):
                q_vals[i] = q_batch[idx].item()

        # 4. 决策
        if random.random() > self.args.epsilon:
            best_score = -float('inf')
            action_idx = candidates[0]
            for i in candidates:
                if i == pass_idx and len(candidates) > 1:
                    continue
                score = q_vals[i] + (0.3 if i == top_idx else 0.0)
                if score > best_score:
                    best_score = score
                    action_idx = i
            if best_score == -float('inf'):
                action_idx = int(np.argmax([q_vals[i] if q_vals[i] != float('-inf') else -float('inf') for i in range(self.act_range + 1)]))
        else:
            non_pass = [i for i in candidates if i != pass_idx]
            action_idx = random.choice(non_pass) if non_pass else 0

        act = process_card_list(self.action[action_idx])

        # 记录经验（reward 由 apply_final_reward 复盘后填入）
        if self.last_obs is not None and self.last_history is not None:
            transition = (
                self.last_obs.cpu(),
                self.last_history.cpu(),
                self.last_act,
                0.0,
                state.cpu(),
                self.action,
                history.cpu(),
                False
            )
            self.episode_transitions.append(transition)

        self.last_obs = state
        self.last_history = history
        self.last_act = act
        self.history_action.append(process_card_list(msg["actionList"][action_idx]))
        return action_idx


# ===================== Test 模式 =====================
class TestClient(BaseClient):
    def __init__(self, url, args):
        super().__init__(url, args.render)
        self.args = args
        self.action = TestAction(args)
        self.rewards = []
        self.wins = 0
        self.episode = 0

    def get_reward(self, order):
        myPos = self.state._myPos
        friendPos = (myPos + 2) % 4
        myRank = order.index(myPos)
        friendRank = order.index(friendPos)
        myRank, friendRank = sorted((myRank, friendRank))
        score = (myRank, friendRank)
        reward_map = {(0,1):5, (0,2):3, (0,3):1, (1,2):-1, (1,3):-3, (2,3):-5}
        return reward_map.get(score, 0)

    def received_message(self, message):
        msg = json.loads(str(message))
        self.state.parse(msg)
        msg.setdefault("myPos", self.state._myPos)

        if msg["stage"] == "beginning":
            self.action.history_action = [['PASS', 'PASS', 'PASS']]
            self.played_cards = torch.zeros(4, 15, dtype=torch.long)
            self.episode += 1

        elif msg["stage"] == "episodeOver":
            reward = self.get_reward(msg["order"])
            self.rewards.append(reward)
            if reward > 0:
                self.wins += 1

        elif msg["stage"] == "gameResult":
            print(f"累计奖励: {self.rewards}, 胜率: {self.wins/len(self.rewards):.2%}")

        if msg.get("type") == "notify" and msg.get("stage") == "play":
            cur_action = msg.get("curAction")
            if cur_action:
                cards = process_card_list(cur_action)
                self.played_cards = self.played_cards + encode_card(cards)

        if "actionList" in msg:
            msg["playedCards"] = getattr(self, "played_cards", torch.zeros(4, 15, dtype=torch.long))
            act_idx = self.action.parse(msg, self.render, state=self.state)
            if msg.get("stage") == "play":
                chosen_action = msg["actionList"][act_idx]
                cards = process_card_list(chosen_action)
                if cards != ('PASS', 'PASS', 'PASS'):
                    self.played_cards = self.played_cards + encode_card(cards)
            self.send(json.dumps({"actIndex": act_idx}))


class TestAction:
    def __init__(self, args):
        self.args = args
        self.no_coach = getattr(args, 'no_coach', False)
        self.action = []
        self.act_range = -1
        self.history_action = [['PASS', 'PASS', 'PASS']]
        check_path(args.model)
        state_dict = torch.load(args.model)
        model_class = state_dict.get("model_class", ActionValueNet)
        self.ValueNet = model_class().to(args.device)
        self.ValueNet.load_state_dict(state_dict["model_state_dict"])
        self.ValueNet.eval()
        coach_status = "纯模型（无教练）" if self.no_coach else "含 TOP 候选过滤"
        print(Back.GREEN, f"成功载入测试模型: {args.model} | {coach_status}", Style.RESET_ALL)

    def MapHistoryToLSTM(self):
        ret = torch.stack([encode_card(a).flatten() for a in self.history_action], dim=0).unsqueeze(0)
        return ret

    def parse(self, msg, render=True, state=None):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]

        if self.no_coach:
            action_indices = list(range(self.act_range + 1))
        else:
            try:
                action_indices = compute_candidate_list(
                    msg, self.action, msg['myPos'], state=state, top_idx=-1)
            except Exception:
                action_indices = list(range(self.act_range + 1))

        # Batch 并行计算 Q 值
        state = StateCatEmbedding(msg)
        history = self.MapHistoryToLSTM().float().to(self.args.device)
        q_vals = [float('-inf')] * (self.act_range + 1)
        if action_indices:
            state_flat = state.flatten().to(self.args.device)
            act_embs = torch.stack([ActionEmbedding(msg, i).to(self.args.device) for i in action_indices])
            inp_batch = torch.cat((
                state_flat.unsqueeze(0).expand(len(action_indices), -1),
                act_embs
            ), dim=1)
            hist_batch = history.expand(len(action_indices), -1, -1)
            with torch.no_grad():
                q_batch = self.ValueNet(inp_batch, hist_batch).sum(dim=1)
            for idx, i in enumerate(action_indices):
                q_vals[i] = q_batch[idx].item()

        best_score = -float('inf')
        index = action_indices[0]
        for i in action_indices:
            if q_vals[i] > best_score:
                best_score = q_vals[i]
                index = i
        if best_score == -float('inf'):
            index = int(np.argmax([q_vals[i] if q_vals[i] != float('-inf') else -float('inf') for i in range(self.act_range + 1)]))

        self.history_action.append(process_card_list(msg["actionList"][index]))
        return index


# ===================== 主入口 =====================
def main():
    parser = argparse.ArgumentParser(description="统一游戏客户端")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    # ---------- demo ----------
    demo_parser = subparsers.add_parser("rule", help="规则/自定义教练")
    demo_parser.add_argument("pos", type=int, help="座位号")
    demo_parser.add_argument("-c", "--client", default="Demo", help="教练名称")
    demo_parser.add_argument("-r", "--render", default=False, action="store_true")
    demo_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    demo_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")

    # ---------- imitation ----------
    im_parser = subparsers.add_parser("imitation", help="模仿学习 (DAgger)")
    im_parser.add_argument("pos", type=int, help="座位号")
    im_parser.add_argument("-r", "--render", default=False, action="store_true")
    im_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    im_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    im_parser.add_argument("--model", default=None, help="预训练模型路径")
    im_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    im_parser.add_argument("--device", default="cpu")
    im_parser.add_argument("--dagger_epochs", type=int, default=3)
    im_parser.add_argument("--dagger_interval", type=int, default=5)
    im_parser.add_argument("--expert_decay", type=float, default=0.995)
    im_parser.add_argument("--expert_init", type=float, default=1.0)
    im_parser.add_argument("--min_expert_prob", type=float, default=0.1)
    im_parser.add_argument("--batch_size", type=int, default=16)
    im_parser.add_argument("--min_dataset_size", type=int, default=32)
    im_parser.add_argument("--log_interval", type=int, default=50, help="日志记录频率（步）")

    # ---------- reinforcement ----------
    rl_parser = subparsers.add_parser("reinforcement", help="强化学习 (DQN / DouZero-MC)")
    rl_parser.add_argument("pos", type=int, help="座位号")
    rl_parser.add_argument("-r", "--render", default=False, action="store_true")
    rl_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    rl_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    rl_parser.add_argument("--model", default=None)
    rl_parser.add_argument("--lr", type=float, default=1e-5, help="学习率（MC建议更低）")
    rl_parser.add_argument("--device", default="cpu")
    rl_parser.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    rl_parser.add_argument("--epsilon_decay", type=float, default=0.9999, help="每局 epsilon 衰减系数")
    rl_parser.add_argument("--gamma", type=float, default=0.98)
    rl_parser.add_argument("--save_interval", type=int, default=1000)
    rl_parser.add_argument("--log_interval", type=int, default=100, help="日志记录频率（局）")
    rl_parser.add_argument("--replay_capacity", type=int, default=100000, help="经验回放池最大容量")
    rl_parser.add_argument("--batch_size", type=int, default=128, help="训练时的批量大小")
    rl_parser.add_argument("--target_update_freq", type=int, default=10, help="每多少局同步一次目标网络")
    rl_parser.add_argument("--update_steps", type=int, default=100, help="每局训练多少个 batch")

    # ---------- test ----------
    test_parser = subparsers.add_parser("test", help="测试已训练模型")
    test_parser.add_argument("pos", type=int, help="座位号")
    test_parser.add_argument("-r", "--render", default=False, action="store_true")
    test_parser.add_argument("--host", default="127.0.0.1", help="游戏服务器 IP")
    test_parser.add_argument("--port", type=int, default=23456, help="游戏服务器端口")
    test_parser.add_argument("--model", required=True)
    test_parser.add_argument("--device", default="cpu")
    test_parser.add_argument("--epsilon", type=float, default=0.1)
    test_parser.add_argument("--no_coach", action="store_true", default=False,
                           help="禁用 TOP 候选过滤")

    args = parser.parse_args()

    # ----- 创建专属输出目录并写入参数日志 -----
    if args.mode in ("imitation", "reinforcement"):
        check_path_dir = f"model/checkpoints_{now_str()}_{args.mode}"
        os.makedirs(check_path_dir, exist_ok=True)
        print(f"输出目录已创建: {check_path_dir}")

        log_path = os.path.join(check_path_dir, "value.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"* LEARNING_RATE : {args.lr}\n")
            f.write(f"* SAVE_INTERVAL : {args.save_interval if args.mode == 'reinforcement' else 'N/A'}\n")
            f.write(f"* LOG_INTERVAL  : {args.log_interval}\n")
            f.write(f"* DEVICE : {args.device}\n")
            if args.mode == "imitation":
                f.write(f"* MODE : Imitation Learning (DAgger)\n")
                f.write(f"* DAGGER_EPOCHS : {args.dagger_epochs}\n")
                f.write(f"* DAGGER_INTERVAL : {args.dagger_interval}\n")
                f.write(f"* EXPERT_DECAY : {args.expert_decay}\n")
                f.write(f"* MIN_EXPERT_PROB : {args.min_expert_prob}\n")
                f.write(f"* BATCH_SIZE : {args.batch_size}\n")
                f.write(f"* MIN_DATASET_SIZE : {args.min_dataset_size}\n")
            else:
                f.write(f"* MODE : Reinforcement Learning (MC / DQN)\n")
                f.write(f"* EPSILON : {args.epsilon}\n")
                f.write(f"* EPSILON_DECAY : {args.epsilon_decay}\n")
                f.write(f"* GAMMA : {args.gamma}\n")
                f.write(f"* BATCH_SIZE : {args.batch_size}\n")
                f.write(f"* UPDATE_STEPS : {args.update_steps}\n")
    else:
        check_path_dir = None

    url = f"ws://{args.host}:{args.port}/game/client{args.pos}"

    if args.mode == "rule":
        run_demo(args)
    else:
        if args.mode == "imitation":
            client = ImitationClient(url, args, check_path_dir)
        elif args.mode == "reinforcement":
            client = ReinforcementClient(url, args, check_path_dir)
        elif args.mode == "test":
            client = TestClient(url, args)
        else:
            sys.exit("未知模式")

        try:
            client.connect()
            client.run_forever()
        except KeyboardInterrupt:
            client.close()


if __name__ == "__main__":
    main()