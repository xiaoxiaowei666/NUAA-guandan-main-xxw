# # -*- coding: utf-8 -*-
# # @Time       : 2022/2/23
# # @Author     : Zhelong Huang
# # @File       : imitation_client.py
# # @Description: imitation learning module
#
# import sys, os
# sys.path.append(os.path.abspath("."))
# import torch.nn.functional as F
# import json
# from ws4py.client.threadedclient import WebSocketClient
# from state import State
# import argparse
# import numpy as np
# from util import *
# from model import ActionValueNet
# from colorama import Back, Style
# import torch
# import time
#
# from coach.EggPan.action import Action
#
# arg = argparse.ArgumentParser()
# arg.add_argument('-r', '--render', default=False, type=bool)
# arg.add_argument('-c', '--client', default="Demo")
# arg.add_argument('-d', '--device', default="cpu")
# arg.add_argument('--model',        default=None, type=str,  help="path of the model, is None")
# arg.add_argument('--lr',           default=1e-3, type=float,help="learning rate of training model")
# arg.add_argument('--save_interval',default=1000, type=int,  help="frequency of saving the model")
# arg.add_argument('--log_interval', default=100,  type=int,  help="frequency of logging the situation of training")
# args = vars(arg.parse_args())
#
# MODE          = "Imitation Learning"
# RENDER        = args["render"]
# COACH         = args["client"]
# DEVICE        = args["device"]
# MODEL         = args["model"]
# LEARNING_RATE = args["lr"]
# SAVE_INTERVAL = args["save_interval"]
# LOG_INTERVAL  = args["log_interval" ]
#
#
# CHECK_PATH = "model/checkpoints_{}".format(now_str())
# if not os.path.exists(CHECK_PATH):
#     os.makedirs(CHECK_PATH)
#
# if MODEL is None or MODEL != "None":
#     check_path(MODEL)
#     STATE_DICT = torch.load(MODEL)
# else:
#     STATE_DICT = {}
#
# with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
#     fp.write("* LEARNING_RATE : {}\n".format(LEARNING_RATE))
#     fp.write("* SAVE_INTERVAL : {}\n".format(SAVE_INTERVAL))
#     fp.write("* LOG_INTERVAL  : {}\n".format(LEARNING_RATE))
#     fp.write("* DEVICE : {}\n".format(DEVICE))
#     fp.write("* MODE : {}\n".format(MODE))
#
#
# class ExampleClient(WebSocketClient):
#     def __init__(self, url, render=False):
#         super().__init__(url)
#         self.state = State(render)
#         self.action = MLPAction()
#         self.render = render
#
#     def opened(self):
#         pass
#
#     def closed(self, code, reason=None):
#         print("Closed down", code, reason)
#
#     def received_message(self, message):
#         message = json.loads(str(message))
#         self.state.parse(message)
#         if "actionList" in message:
#             act_index = self.action.parse(message, self.render)
#             self.send(json.dumps({"actIndex": act_index}))
#         #新加的选择
#         # 放在这里，不依赖 actionList
#         if message.get("stage") == "episodeOver" or message.get("stage") == "gameOver":
#             self.action.history_action = [['PASS', 'PASS', 'PASS']]
#         #     self.history_action = [['PASS', 'PASS', 'PASS']]
#
#
# class MLPAction(object):
#     def __init__(self):
#         self.action         = []
#         self.act_range      = -1
#         self.history_action = [['PASS', 'PASS', 'PASS']]
#         self.count  = 0
#
#         save_model_class = STATE_DICT.get("model_class", None)
#         if save_model_class:
#             self.ValueNet = save_model_class().to(DEVICE)
#         else:
#             self.ValueNet = ActionValueNet().to(DEVICE)
#
#
#         save_state_dict = STATE_DICT.get("model_state_dict", None)
#         if save_state_dict:
#             self.ValueNet.load_state_dict(save_state_dict)
#             print(Back.GREEN, "成功载入模型 : ", MODEL, " 3秒后开始训练", Style.RESET_ALL)
#             time.sleep(3)
#
#        # self.optimizer = torch.optim.SGD(self.ValueNet.parameters(), lr=LEARNING_RATE)
#         #修改的方案！！！
#         self.optimizer = torch.optim.Adam(self.ValueNet.parameters(), lr=LEARNING_RATE)
#         # TODO : 未来这个名字应该就只叫做self.action
#         self.eggpan_action = Action(render=RENDER)
#
#     def MapHistoryToLSTM(self):
#         ret = torch.stack([encode_card(action).flatten() for action in self.history_action], dim=0)
#         ret = ret.unsqueeze(0)
#         return ret
#
#     def update(self, loss : torch.Tensor):
#         # loss : 需要被最小化的值
#         self.optimizer.zero_grad()
#         loss.backward()
#
#         self.optimizer.step()
#
#     # def parse(self, msg, render=True):
#     #     self.action = msg["actionList"]
#     #     self.act_range = msg["indexRange"]
#     #     if render:
#     #         # print(self.action)
#     #         print(Back.BLUE, "可选动作范围为: 0至{}".format(self.act_range), Style.RESET_ALL)
#     #
#     #     index = self.eggpan_action.parse_AI(msg, msg.get("myPos", 0))
#     #     index = np.clip(index, 0, self.act_range).tolist()
#     #
#     #     state       = StateCatEmbedding(msg)
#     #     demo_action = ActionEmbedding(msg, index)
#     #     history     = self.MapHistoryToLSTM().float().to(DEVICE)
#     #
#     #     state_input = torch.cat((state.flatten(), demo_action.flatten()), dim=0).float().to(DEVICE)
#     #     state_input = state_input.unsqueeze(0).float().to(DEVICE)
#     #
#     #     # @debug
#     #     # print(state_input.shape)
#     #     # print(RENDER)
#     #     # print(type(RENDER))
#     #     # time.sleep(3)
#     #
#     #     # state_input shape torch.Size([1, 493])
#     #     value = self.ValueNet(state_input, history).sum()
#     #     loss = -value
#     #     self.update(loss)
#     #
#     #     cur_val = value.detach().cpu().numpy().tolist()
#     #
#     #     if self.count % SAVE_INTERVAL == 0:
#     #         torch.save({
#     #             "coach" : COACH,
#     #             "mode"  : MODE,
#     #             "model_state_dict" : self.ValueNet.state_dict(),
#     #             "model_class"  : ActionValueNet
#     #         }, CHECK_PATH + "/{}_value_{}.pth".format(now_str(), round(cur_val, 3)))
#     #     # 计算 diff（专家动作与随机其他动作的价值差）
#     #
#     #     if self.count % LOG_INTERVAL == 0:
#     #         with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
#     #             fp.write("[count={}] value = {}\n".format(self.count, - loss.item()))
#     #
#     #     self.history_action.append(process_card_list(msg["actionList"][index]))
#     #     self.count += 1
#     #     return index
#
#     #正在修改的代码:
#     def parse(self, msg, render=True):
#         self.action = msg["actionList"]
#         self.act_range = msg["indexRange"]
#         if render:
#             print(Back.BLUE, "可选动作范围为: 0至{}".format(self.act_range), Style.RESET_ALL)
#
#         index = self.eggpan_action.parse_AI(msg, msg.get("myPos", 0))
#         index = np.clip(index, 0, self.act_range).tolist()
#
#         state = StateCatEmbedding(msg)
#         history = self.MapHistoryToLSTM().float().to(DEVICE)
#
#         # --- 方案B：计算所有动作的 Q 值，softmax + 交叉熵 ---
#         q_values = []
#         for i in range(self.act_range + 1):
#             action_i = ActionEmbedding(msg, i)
#             inp = torch.cat((state.flatten(), action_i.flatten()), dim=0).to(DEVICE)
#             inp = inp.unsqueeze(0)  # [1, 493]
#             q = self.ValueNet(inp, history).sum()
#             q_values.append(q)
#
#         q_tensor = torch.stack(q_values)  # (N_actions,)
#         probs = torch.softmax(q_tensor, dim=0)
#         loss = -torch.log(probs[index] + 1e-8)
#         self.update(loss)
#
#         cur_val = probs[index].detach().cpu().item()
#         # --- 方案B结束 ---
#
#         if self.count % SAVE_INTERVAL == 0:
#             torch.save({
#                 "coach": COACH,
#                 "mode": MODE,
#                 "model_state_dict": self.ValueNet.state_dict(),
#                 "model_class": ActionValueNet
#             }, CHECK_PATH + "/{}_value_{}.pth".format(now_str(), round(cur_val, 3)))
#
#         if self.count % LOG_INTERVAL == 0:
#             with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
#                 fp.write("[count={}] prob = {}\n".format(self.count, cur_val))
#
#         self.history_action.append(process_card_list(msg["actionList"][index]))
#         self.count += 1
#         return index
# if __name__ == '__main__':
#     try:
#         ws = ExampleClient('ws://127.0.0.1:23456/game/client1', render=bool(RENDER))
#         ws.connect()
#         ws.run_forever()
#     except KeyboardInterrupt:
#         ws.close()


# # -*- coding: utf-8 -*-
# # imitation_client.py with DAgger (Online Behavioral Cloning)
#
# import sys, os
#
# sys.path.append(os.path.abspath("."))
# import torch.nn.functional as F
# import json
# from ws4py.client.threadedclient import WebSocketClient
# from state import State
# import argparse
# import numpy as np
# from util import *
# from model import ActionValueNet
# from colorama import Back, Style
# import torch
# import time
# import random
# from torch.utils.data import DataLoader, Dataset
#
# from coach.EggPan.action import Action
#
# arg = argparse.ArgumentParser()
# arg.add_argument('-r', '--render', default=False, type=bool)
# arg.add_argument('-c', '--client', default="Demo")
# arg.add_argument('-d', '--device', default="cpu")
# arg.add_argument('--model', default=None, type=str, help="path of the model, is None")
# arg.add_argument('--lr', default=1e-3, type=float, help="learning rate of training model")
# arg.add_argument('--save_interval', default=1000, type=int, help="frequency of saving the model")
# arg.add_argument('--log_interval', default=100, type=int, help="frequency of logging the situation of training")
# arg.add_argument('--dagger_epochs', default=5, type=int, help="epochs of retraining after each episode")
# arg.add_argument('--expert_decay', default=0.98, type=float, help="decay factor for expert probability")
# arg.add_argument('--expert_init', default=1.0, type=float, help="initial expert probability")
# args = vars(arg.parse_args())
#
# MODE = "Imitation Learning (DAgger)"
# RENDER = args["render"]
# COACH = args["client"]
# DEVICE = args["device"]
# MODEL = args["model"]
# LEARNING_RATE = args["lr"]
# SAVE_INTERVAL = args["save_interval"]
# LOG_INTERVAL = args["log_interval"]
# DAGGER_EPOCHS = args["dagger_epochs"]
# EXPERT_DECAY = args["expert_decay"]
# EXPERT_INIT = args["expert_init"]
#
# CHECK_PATH = "model/checkpoints_{}".format(now_str())
# if not os.path.exists(CHECK_PATH):
#     os.makedirs(CHECK_PATH)
#
# if MODEL is None or MODEL != "None":
#     check_path(MODEL)
#     STATE_DICT = torch.load(MODEL)
# else:
#     STATE_DICT = {}
#
# with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
#     fp.write("* LEARNING_RATE : {}\n".format(LEARNING_RATE))
#     fp.write("* SAVE_INTERVAL : {}\n".format(SAVE_INTERVAL))
#     fp.write("* LOG_INTERVAL  : {}\n".format(LOG_INTERVAL))
#     fp.write("* DEVICE : {}\n".format(DEVICE))
#     fp.write("* MODE : {}\n".format(MODE))
#     fp.write("* DAGGER_EPOCHS : {}\n".format(DAGGER_EPOCHS))
#     fp.write("* EXPERT_DECAY : {}\n".format(EXPERT_DECAY))
#
#
# # ----- 自定义数据集类，用于批量训练 -----
# class DaggerDataset(Dataset):
#     def __init__(self, data):
#         # data: list of (state_tensor, action_embeddings_list, history_actions_list, expert_idx)
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         state, action_embs, hist_actions, expert_idx = self.data[idx]
#         return state, action_embs, hist_actions, expert_idx
#
#
# class ExampleClient(WebSocketClient):
#     def __init__(self, url, render=False):
#         super().__init__(url)
#         self.state = State(render)
#         self.action = MLPAction()
#         self.render = render
#         self.episode = 0
#
#     def opened(self):
#         pass
#
#     def closed(self, code, reason=None):
#         print("Closed down", code, reason)
#
#     def received_message(self, message):
#         message = json.loads(str(message))
#         self.state.parse(message)
#
#         if message.get("stage") == "beginning":
#             # 新一局开始，重置历史
#             self.action.history_action = [['PASS', 'PASS', 'PASS']]
#             self.episode += 1
#
#         elif message.get("stage") == "episodeOver" or message.get("stage") == "gameOver":
#             # 一局结束，进行 DAgger 重训练
#             print(
#                 Back.YELLOW + f"Episode {self.episode} over. Retraining on {len(self.action.dataset)} samples..." + Style.RESET_ALL)
#             self.action.train_from_dataset(epochs=DAGGER_EPOCHS)
#             # 衰减专家概率
#             self.action.use_expert_prob = max(0.1, self.action.use_expert_prob * EXPERT_DECAY)
#             print(f"Expert prob = {self.action.use_expert_prob:.4f}")
#
#         if "actionList" in message:
#             act_index = self.action.parse(message, self.render)
#             self.send(json.dumps({"actIndex": act_index}))
#
#
# class MLPAction(object):
#     def __init__(self):
#         self.action = []
#         self.act_range = -1
#         self.history_action = [['PASS', 'PASS', 'PASS']]
#         self.count = 0
#
#         # 模型
#         save_model_class = STATE_DICT.get("model_class", None)
#         if save_model_class:
#             self.ValueNet = save_model_class().to(DEVICE)
#         else:
#             self.ValueNet = ActionValueNet().to(DEVICE)
#
#         save_state_dict = STATE_DICT.get("model_state_dict", None)
#         if save_state_dict:
#             self.ValueNet.load_state_dict(save_state_dict)
#             print(Back.GREEN, "成功载入模型 : ", MODEL, " 3秒后开始训练", Style.RESET_ALL)
#             time.sleep(3)
#
#         self.optimizer = torch.optim.Adam(self.ValueNet.parameters(), lr=LEARNING_RATE)
#         self.eggpan_action = Action(render=RENDER)
#
#         # DAgger 相关
#         self.dataset = []  # 存储 (state, action_embs, hist_actions, expert_idx)
#         self.max_dataset_size = 100000  # 内存上限
#         self.use_expert_prob = EXPERT_INIT  # 当前使用专家动作的概率
#
#     def MapHistoryToLSTM(self):
#         ret = torch.stack([encode_card(action).flatten() for action in self.history_action], dim=0)
#         ret = ret.unsqueeze(0)
#         return ret
#
#     def update(self, loss: torch.Tensor):
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#     # 记录一条新样本
#     def add_to_dataset(self, msg, expert_index):
#         state_tensor = StateCatEmbedding(msg).cpu()
#         action_embeddings = [
#             encode_card(process_card_list(msg["actionList"][i])).flatten().cpu()
#             for i in range(len(msg["actionList"]))
#         ]
#         # 用当前历史动作（出牌前状态）作为历史特征
#         hist_actions = [list(a) for a in self.history_action]  # 深拷贝
#         self.dataset.append((state_tensor, action_embeddings, hist_actions, expert_index))
#         if len(self.dataset) > self.max_dataset_size:
#             self.dataset.pop(0)  # 丢弃最旧的
#
#     # 使用当前模型选择动作（softmax 概率采样）
#     def select_action_by_model(self, msg):
#         state = StateCatEmbedding(msg).to(DEVICE)
#         history = self.MapHistoryToLSTM().float().to(DEVICE)
#
#         q_values = []
#         for i in range(len(msg["actionList"])):
#             action_emb = encode_card(process_card_list(msg["actionList"][i])).flatten().to(DEVICE)
#             inp = torch.cat((state.flatten(), action_emb), dim=0).unsqueeze(0)
#             q = self.ValueNet(inp, history).sum()
#             q_values.append(q)
#         q_tensor = torch.stack(q_values)
#         probs = torch.softmax(q_tensor, dim=0).detach().cpu().numpy()
#         # 依概率采样，避免总是选最大（增加探索）
#         index = np.random.choice(len(q_values), p=probs)
#         return index
#
#     # 解析消息，返回动作索引
#     def parse(self, msg, render=True):
#         self.action = msg["actionList"]
#         self.act_range = msg["indexRange"]
#         if render:
#             print(Back.BLUE, "可选动作范围为: 0至{}".format(self.act_range), Style.RESET_ALL)
#
#         # 1. 专家动作
#         expert_index = self.eggpan_action.parse_AI(msg, msg.get("myPos", 0))
#         expert_index = np.clip(expert_index, 0, self.act_range).tolist()
#
#         # 2. 记录状态-专家对到数据集
#         self.add_to_dataset(msg, expert_index)
#
#         # 3. 动作选择：以一定概率跟随专家，否则用模型
#         if random.random() < self.use_expert_prob:
#             index = expert_index
#         else:
#             index = self.select_action_by_model(msg)
#             # 防止选择超出范围的索引
#             index = min(index, self.act_range)
#
#         # 更新历史
#         self.history_action.append(process_card_list(msg["actionList"][index]))
#
#         # 可选：记录日志（如果满足间隔）
#         if self.count % LOG_INTERVAL == 0:
#             with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
#                 fp.write(
#                     f"[count={self.count}] expert_prob={self.use_expert_prob:.4f}, dataset_size={len(self.dataset)}\n")
#
#         self.count += 1
#         return index
#
#     def train_from_dataset(self, epochs=5):
#         if len(self.dataset) < 4:
#             return
#         self.ValueNet.train()
#         dataset = DaggerDataset(self.dataset)
#         dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: batch)
#
#         total_loss = 0.0
#         total_samples = 0  # <--- 新增这一行
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             for batch in dataloader:
#                 for state_t, action_embs, hist_actions, expert_idx in batch:
#                     state_t = state_t.to(DEVICE)
#                     hist_tensor = torch.stack([encode_card(a).flatten() for a in hist_actions], dim=0).unsqueeze(
#                         0).float().to(DEVICE)
#                     q_vals = []
#                     for emb in action_embs:
#                         emb = emb.to(DEVICE)
#                         inp = torch.cat((state_t.flatten(), emb), dim=0).unsqueeze(0)
#                         q = self.ValueNet(inp, hist_tensor).sum()
#                         q_vals.append(q)
#                     q_tensor = torch.stack(q_vals)
#                     probs = torch.softmax(q_tensor, dim=0)
#                     loss = -torch.log(probs[expert_idx] + 1e-8)
#                     self.update(loss)
#                     epoch_loss += loss.item()
#                     total_loss += loss.item()
#                     total_samples += 1  # <--- 新增计数
#             avg_epoch_loss = epoch_loss / len(dataset)
#             print(f" Epoch {epoch + 1}/{epochs} - avg loss: {avg_epoch_loss:.4f}")
#
#         # 计算 cur_val（平均专家概率）
#         avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
#         cur_val = np.exp(-avg_loss) if avg_loss < 20 else 0.0
#
#         # 保存模型
#         torch.save({
#             "coach": COACH,
#             "mode": MODE,
#             "model_state_dict": self.ValueNet.state_dict(),
#             "model_class": ActionValueNet
#         }, CHECK_PATH + "/{}_value_{}.pth".format(now_str(), round(cur_val, 3)))
#
#         print(f"Model saved. cur_val (avg expert prob): {cur_val:.4f}")
# if __name__ == '__main__':
#     try:
#         ws = ExampleClient('ws://127.0.0.1:23456/game/client1', render=bool(RENDER))
#         ws.connect()
#         ws.run_forever()
#     except KeyboardInterrupt:
#         ws.close()
# -*- coding: utf-8 -*-
# imitation_client.py - DAgger with Correct Mini-Batch Training & Gradient Clipping

import sys, os
sys.path.append(os.path.abspath("."))
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ws4py.client.threadedclient import WebSocketClient
from state import State
from util import *
from model import ActionValueNet
from colorama import Back, Style
from coach.EggPan.action import Action

# ======================= 参数解析 =======================
arg = argparse.ArgumentParser()
arg.add_argument('-r', '--render', default=False, type=bool)
arg.add_argument('-c', '--client', default="Demo")
arg.add_argument('-d', '--device', default="cpu")
arg.add_argument('--model', default=None, type=str, help="预训练模型路径")
arg.add_argument('--lr', default=1e-4, type=float, help="学习率 (建议 1e-4)")
arg.add_argument('--save_interval', default=250, type=int, help="每多少步保存一次模型")
arg.add_argument('--log_interval', default=50, type=int, help="每多少步记录日志")
arg.add_argument('--dagger_epochs', default=3, type=int, help="每次全量训练的 epoch 数")
arg.add_argument('--dagger_interval', default=5, type=int, help="每 N 局进行一次全量训练")
arg.add_argument('--expert_decay', default=0.995, type=float, help="专家概率衰减因子 (建议先保守)")
arg.add_argument('--expert_init', default=1.0, type=float, help="初始专家概率")
arg.add_argument('--min_expert_prob', default=0.1, type=float, help="最低专家概率")
arg.add_argument('--batch_size', default=16, type=int, help="训练时的 mini-batch 大小")
arg.add_argument('--min_dataset_size', default=32, type=int, help="至少多少条数据才开始全量训练")
args = vars(arg.parse_args())

MODE = "Imitation Learning (DAgger - Corrected)"
RENDER = args["render"]
COACH = args["client"]
DEVICE = args["device"]
MODEL = args["model"]
LEARNING_RATE = args["lr"]
SAVE_INTERVAL = args["save_interval"]
LOG_INTERVAL = args["log_interval"]
DAGGER_EPOCHS = args["dagger_epochs"]
DAGGER_INTERVAL = args["dagger_interval"]
EXPERT_DECAY = args["expert_decay"]
EXPERT_INIT = args["expert_init"]
MIN_EXPERT_PROB = args["min_expert_prob"]
BATCH_SIZE = args["batch_size"]
MIN_DATASET_SIZE = args["min_dataset_size"]

# ======================= 路径与模型加载 =======================
CHECK_PATH = "model/checkpoints_{}".format(now_str())
if not os.path.exists(CHECK_PATH):
    os.makedirs(CHECK_PATH)

if MODEL is None or MODEL != "None":
    check_path(MODEL)
    STATE_DICT = torch.load(MODEL)
else:
    STATE_DICT = {}

with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
    fp.write("* LEARNING_RATE : {}\n".format(LEARNING_RATE))
    fp.write("* SAVE_INTERVAL : {}\n".format(SAVE_INTERVAL))
    fp.write("* LOG_INTERVAL  : {}\n".format(LOG_INTERVAL))
    fp.write("* DEVICE : {}\n".format(DEVICE))
    fp.write("* MODE : {}\n".format(MODE))
    fp.write("* DAGGER_EPOCHS : {}\n".format(DAGGER_EPOCHS))
    fp.write("* DAGGER_INTERVAL : {}\n".format(DAGGER_INTERVAL))
    fp.write("* EXPERT_DECAY : {}\n".format(EXPERT_DECAY))
    fp.write("* MIN_EXPERT_PROB : {}\n".format(MIN_EXPERT_PROB))
    fp.write("* BATCH_SIZE : {}\n".format(BATCH_SIZE))
    fp.write("* MIN_DATASET_SIZE : {}\n".format(MIN_DATASET_SIZE))

# ======================= 自定义数据集 =======================
class DaggerDataset(Dataset):
    def __init__(self, data):
        # data: list of (state_tensor, action_embeddings_list, history_actions_list, expert_idx)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action_embs, hist_actions, expert_idx = self.data[idx]
        return state, action_embs, hist_actions, expert_idx

# ======================= WebSocket 客户端 =======================
class ExampleClient(WebSocketClient):
    def __init__(self, url, render=False):
        super().__init__(url)
        self.state = State(render)
        self.action = MLPAction()
        self.render = render
        self.episode = 0

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        message = json.loads(str(message))
        self.state.parse(message)

        if message.get("stage") == "beginning":
            self.action.history_action = [['PASS', 'PASS', 'PASS']]
            self.episode += 1

        elif message.get("stage") in ("episodeOver", "gameOver"):
            # 定期全量训练
            if self.episode % DAGGER_INTERVAL == 0:
                print(Back.YELLOW +
                      f"[Episode {self.episode}] 全量训练, 数据集: {len(self.action.dataset)}" +
                      Style.RESET_ALL)
                self.action.train_from_dataset(epochs=DAGGER_EPOCHS)
            # 衰减专家概率
            self.action.use_expert_prob = max(MIN_EXPERT_PROB,
                                              self.action.use_expert_prob * EXPERT_DECAY)
            print(f"Expert prob = {self.action.use_expert_prob:.4f}")

        if "actionList" in message:
            act_index = self.action.parse(message, self.render)
            self.send(json.dumps({"actIndex": act_index}))

# ======================= 核心动作/训练类 =======================
class MLPAction(object):
    def __init__(self):
        self.action = []
        self.act_range = -1
        self.history_action = [['PASS', 'PASS', 'PASS']]
        self.count = 0

        # 模型
        save_model_class = STATE_DICT.get("model_class", None)
        if save_model_class:
            self.ValueNet = save_model_class().to(DEVICE)
        else:
            self.ValueNet = ActionValueNet().to(DEVICE)

        save_state_dict = STATE_DICT.get("model_state_dict", None)
        if save_state_dict:
            self.ValueNet.load_state_dict(save_state_dict)
            print(Back.GREEN, "成功载入模型 : ", MODEL, " 3秒后开始训练", Style.RESET_ALL)
            time.sleep(3)

        self.optimizer = torch.optim.Adam(self.ValueNet.parameters(), lr=LEARNING_RATE)
        self.eggpan_action = Action(render=RENDER)

        # DAgger 数据
        self.dataset = []                     # (state, action_embs, hist_actions, expert_idx)
        self.max_dataset_size = 100000        # 内存上限
        self.use_expert_prob = EXPERT_INIT    # 当前专家概率

    def MapHistoryToLSTM(self):
        ret = torch.stack([encode_card(action).flatten() for action in self.history_action], dim=0)
        ret = ret.unsqueeze(0)
        return ret

    # ---------- 数据收集 ----------
    def add_to_dataset(self, msg, expert_index):
        state_tensor = StateCatEmbedding(msg).cpu()
        action_embeddings = [
            encode_card(process_card_list(msg["actionList"][i])).flatten().cpu()
            for i in range(len(msg["actionList"]))
        ]
        hist_actions = [list(a) for a in self.history_action]  # 深拷贝
        self.dataset.append((state_tensor, action_embeddings, hist_actions, expert_index))
        if len(self.dataset) > self.max_dataset_size:
            self.dataset.pop(0)

    # ---------- 动作选择（模型） ----------
    def select_action_by_model(self, msg):
        state = StateCatEmbedding(msg).to(DEVICE)
        history = self.MapHistoryToLSTM().float().to(DEVICE)

        q_values = []
        for i in range(len(msg["actionList"])):
            action_emb = encode_card(process_card_list(msg["actionList"][i])).flatten().to(DEVICE)
            inp = torch.cat((state.flatten(), action_emb), dim=0).unsqueeze(0)
            q = self.ValueNet(inp, history).sum()
            q_values.append(q)
        q_tensor = torch.stack(q_values)
        probs = torch.softmax(q_tensor, dim=0).detach().cpu().numpy()
        index = np.random.choice(len(q_values), p=probs)
        return index

    # ---------- 每一步的解析 ----------
    def parse(self, msg, render=True):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        if render:
            print(Back.BLUE, "可选动作范围为: 0至{}".format(self.act_range), Style.RESET_ALL)

        # 专家动作
        expert_index = self.eggpan_action.parse_AI(msg, msg.get("myPos", 0))
        expert_index = np.clip(expert_index, 0, self.act_range).tolist()

        # 记录到数据集（供后续全量训练）
        self.add_to_dataset(msg, expert_index)

        # 混合策略选择动作
        if random.random() < self.use_expert_prob:
            index = expert_index
        else:
            index = self.select_action_by_model(msg)
            index = min(index, self.act_range)

        # 更新历史
        self.history_action.append(process_card_list(msg["actionList"][index]))

        # 日志
        if self.count % LOG_INTERVAL == 0:
            with open(CHECK_PATH + "/value.log", "a", encoding="utf-8") as fp:
                fp.write(
                    f"[count={self.count}] expert_prob={self.use_expert_prob:.4f}, dataset_size={len(self.dataset)}\n")

        self.count += 1
        return index

    # ---------- 全量训练（修正版） ----------
    def train_from_dataset(self, epochs=3):
        if len(self.dataset) < MIN_DATASET_SIZE:
            print(f"数据量不足 ({len(self.dataset)}/{MIN_DATASET_SIZE})，跳过训练")
            return

        self.ValueNet.train()

        # 预处理所有样本，避免重复编码
        processed = []
        for state_t, action_embs, hist_actions, expert_idx in self.dataset:
            state_t = state_t.to(DEVICE)
            hist_tensor = torch.stack(
                [encode_card(a).flatten() for a in hist_actions], dim=0
            ).unsqueeze(0).float().to(DEVICE)
            action_embs_dev = [emb.to(DEVICE) for emb in action_embs]
            processed.append((state_t, action_embs_dev, hist_tensor, expert_idx))

        total_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            random.shuffle(processed)

            # mini-batch 循环
            for start in range(0, len(processed), BATCH_SIZE):
                batch = processed[start:start + BATCH_SIZE]

                self.optimizer.zero_grad()          # 一个 batch 清零一次梯度
                batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

                for state_t, action_embs, hist_tensor, expert_idx in batch:
                    # 计算该样本的所有动作 Q 值
                    q_vals = []
                    for emb in action_embs:
                        inp = torch.cat((state_t.flatten(), emb), dim=0).unsqueeze(0)
                        q = self.ValueNet(inp, hist_tensor).sum()
                        q_vals.append(q)
                    q_tensor = torch.stack(q_vals)
                    probs = torch.softmax(q_tensor, dim=0)
                    loss = -torch.log(probs[expert_idx] + 1e-8)
                    batch_loss = batch_loss + loss
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    total_samples += 1

                # 取平均后反向传播
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()

                # 梯度裁剪，防止爆炸
                torch.nn.utils.clip_grad_norm_(self.ValueNet.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / len(processed) if processed else 0
            print(f"  Epoch {epoch + 1}/{epochs} - avg loss: {avg_epoch_loss:.4f}")

        # 计算 cur_val（平均专家概率）并保存模型
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        cur_val = np.exp(-avg_loss) if avg_loss < 20 else 0.0

        torch.save({
            "coach": COACH,
            "mode": MODE,
            "model_state_dict": self.ValueNet.state_dict(),
            "model_class": ActionValueNet
        }, CHECK_PATH + "/{}_value_{}.pth".format(now_str(), round(cur_val, 3)))

        print(f"模型已保存. cur_val (avg expert prob): {cur_val:.4f}")
        self.ValueNet.eval()

# ======================= 主入口 =======================
if __name__ == '__main__':
    try:
        ws = ExampleClient('ws://127.0.0.1:23456/game/client1', render=bool(RENDER))
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()