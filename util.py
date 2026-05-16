import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from colorama import Back, Style
from datetime import datetime
from collections import deque
import random

card_color = ['S', 'H', 'C', 'D']
card_score = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
color2index = {v: i for i, v in enumerate(card_color)}
score2index = {v: i for i, v in enumerate(card_score)}
rank2index = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
    '7': 5, '8': 6, '9': 7,
    't': 8, 'T': 8,
    'j': 9, 'J': 9,
    'q': 10, 'Q': 10,
    'k': 11, 'K': 11,
    'a': 12, 'A': 12
}


def encode_card(card_list):
    embedding_matrix = np.zeros((4, 15))
    embedding_matrix = torch.LongTensor(embedding_matrix)
    if card_list is None:
        return embedding_matrix
    for card in card_list:
        if card == "PASS":
            return embedding_matrix
        if card == "SB":
            embedding_matrix[3, 13] += 1
        elif card == "HR":
            embedding_matrix[3, 14] += 1
        else:
            color_index = color2index[card[0]]
            score_index = score2index[card[1]]
            embedding_matrix[color_index, score_index] += 1
    return embedding_matrix


def process_card_list(card):
    if card is None:
        return ('PASS', 'PASS', 'PASS')
    if card[0] == 'PASS':
        return card
    else:
        return card[-1]


def encode_message(message):
    handcards_tensor = encode_card(message["handCards"])
    playArea_tensor = [encode_card(process_card_list(card["playArea"])) for card in message["publicInfo"]]
    rest_tensor = [card["rest"] for card in message["publicInfo"]]
    try:
        rest_tensor = F.one_hot(torch.tensor(rest_tensor), num_classes=30)
    except:
        print(Back.RED, rest_tensor, Style.RESET_ALL)
    rank_tensor = F.one_hot(torch.tensor([rank2index[message["curRank"]]]), num_classes=13)
    return dict(
        handcards=handcards_tensor,
        playArea=playArea_tensor,
        rest_num=rest_tensor,
        rank_num=rank_tensor
    )


def StateCatEmbedding(message):
    encode_info = encode_message(message)
    state_tensor = torch.cat((
        torch.flatten(encode_info["handcards"]),
        torch.flatten(encode_info["playArea"][0]),
        torch.flatten(encode_info["playArea"][1]),
        torch.flatten(encode_info["playArea"][2]),
        torch.flatten(encode_info["playArea"][3]),
        torch.flatten(encode_info["rest_num"]),
        torch.flatten(encode_info["rank_num"])
    ), dim=0)
    return state_tensor


def ActionEmbedding(message, action_index):
    action = message["actionList"][action_index]
    action_tensor = encode_card(process_card_list(action))
    return torch.flatten(action_tensor)


def StateAndActionCatEmbedding(message, action_index):
    state_tensor = StateCatEmbedding(message)
    action_tensor = ActionEmbedding(message, action_index)
    return torch.cat((torch.flatten(state_tensor), torch.flatten(action_tensor)), dim=0)


def check_path(path: str, fail_text=""):
    if not os.path.exists(path):
        print(Back.RED, "文件/文件夹 {} 不存在!{}".format(path, fail_text), Style.RESET_ALL)
        exit(-1)
    else:
        return True


def now_str():
    now = datetime.now()
    UUD = "{}_{}_{}_{}_{}_{}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    return UUD


class MemoryBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def learn_batch(self, batch, ValueNet, target_net, optimizer, gamma, device):
        """
        TD(0) 训练：target = r + gamma * max_a' Q(s', a') * (1 - done)
        target_net 参数保留以兼容调用方。
        """
        losses = []
        optimizer.zero_grad()
        total_loss = 0.0

        for obs, history, act, reward, obs_next, actionListNext, history_next, done in batch:
            # 当前 Q 值
            act_emb = encode_card(act).flatten().to(device)
            state_curr = torch.cat((obs.flatten().to(device), act_emb)).unsqueeze(0)
            q_curr = ValueNet(state_curr, history.float().to(device)).sum()

            # TD target
            with torch.no_grad():
                if done or not actionListNext:
                    td_target = reward
                else:
                    obs_next = obs_next.float().to(device)
                    history_next = history_next.float().to(device)
                    inps = []
                    for act_entry in actionListNext:
                        act_emb_next = encode_card(process_card_list(act_entry)).flatten().to(device)
                        inps.append(torch.cat((obs_next.flatten(), act_emb_next)))
                    batched_inp = torch.stack(inps, dim=0)
                    batched_hist = history_next.expand(len(inps), -1, -1)
                    # Double Q：online 选动作，target 打分
                    online_qs = ValueNet(batched_inp, batched_hist).squeeze(-1)
                    best_idx = online_qs.argmax().item()
                    target_qs = target_net(batched_inp, batched_hist).squeeze(-1)
                    td_target = reward + gamma * target_qs[best_idx].item()

            target = torch.FloatTensor([td_target]).to(device).squeeze()
            loss = F.mse_loss(q_curr, target)
            total_loss += loss
            losses.append(loss.item())

        avg_loss = total_loss / len(batch)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(ValueNet.parameters(), max_norm=1.0)
        optimizer.step()

        return losses


def lock_model_path(value, model_root="./model", keyword="value", reverse=True):
    iter_obj = os.listdir(model_root)
    if reverse:
        iter_obj.reverse()
    for checkpoints in iter_obj:
        check_path = os.path.join(model_root, checkpoints)
        for pth in os.listdir(check_path):
            if pth.endswith("pth"):
                params = pth[:-4].split("_")
                if params[-2] == keyword and float(params[-1]) == value:
                    return os.path.join(check_path, pth)
    return ""


def debugout(text, color: str = "BLUE"):
    pre_obj = getattr(Back, color.upper())
    print(pre_obj, text, Style.RESET_ALL)


if __name__ == "__main__":
    actionList = [['PASS', 'PASS', 'PASS'], ['Straight', 'T', ['ST', 'SJ', 'SQ', 'SK', 'HA']]]
    for card_list in actionList:
        if card_list[0] == 'PASS':
            encode_card(card_list)
        else:
            encode_card(card_list[-1])