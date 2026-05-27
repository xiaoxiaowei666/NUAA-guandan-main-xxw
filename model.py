# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional

class CrossUnit(nn.Module):
    def __init__(self, input_dim, inner_dim, out_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, inner_dim)
        self.fc_2 = nn.Linear(inner_dim, out_dim)
        self.align = (input_dim == out_dim)
        if not self.align:
            self.fc_3 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        z = self.fc_1(x).relu()
        z = self.fc_2(z)
        if not self.align:
            x = self.fc_3(x)
        return functional.relu(x + z)


class ActionValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM 处理历史出牌序列
        self.lstm = nn.LSTM(60, 512, batch_first=True)
        # 4层 CrossUnit，逐级收窄
        self.total_cross = nn.Sequential(
            CrossUnit(685 + 512, 1024, 1024),
            CrossUnit(1024, 1024, 512),
            CrossUnit(512, 512, 512),
            CrossUnit(512, 512, 256),
        )
        self.value_head = nn.Linear(256, 1)  # 无激活函数，Q 值可正可负

    def forward(self, state, history):
        # state: [B, 685]  (625 维状态 + 60 维动作编码)
        # history: [B, T, 60]
        out, (h_n, _) = self.lstm(history)
        # 取最后一个时间步的LSTM输出
        x = torch.cat((out[:, -1, :], state), dim=1)   # [B, 512+685]
        value = self.value_head(self.total_cross(x))   # [B, 1]
        return value