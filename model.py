# # -*- coding: utf-8 -*-
# # @Time       : 2022/2/20
# # @Author     : Zhelong Huang
# # @File       : model.py
# # @Description: forward model
#
import torch
from torch import nn
from torch.nn import functional

#残差交叉单元
class CrossUnit(nn.Module):
    def __init__(self, input_dim, inner_dim, out_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, inner_dim)
        self.fc_2 = nn.Linear(inner_dim, out_dim)
        self.align = (input_dim == out_dim)   # 判断输入输出维度是否一样
        if not self.align:
            self.fc_3 = nn.Linear(input_dim, out_dim)  # 不一样就做维度对齐

    def forward(self, x):
        z = self.fc_1(x).relu()   # 第一层 + 激活函数（增加非线性）
        z = self.fc_2(z)    # 第二层
        if not self.align:
            x = self.fc_3(x)  # 维度对齐
        return functional.relu(x + z)  # 残差连接：原始输入 + 加工结果，再激活
#
#
# # 计算当前输入message中某一个action的值  这是整个模型的本体，输入游戏信息，输出一个价值分数。
class ActionValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ① LSTM 处理历史出牌序列
        self.lstm = nn.LSTM(60, 256, batch_first=True)
        # ② 5层 CrossUnit 堆叠，最后一层输出 1 维（标量价值）
        self.total_cross = nn.Sequential(
            CrossUnit(493 + 256, 512, 512),
            CrossUnit(512      , 512, 512),
            CrossUnit(512      , 512, 512),
            CrossUnit(512      , 512, 512),
            CrossUnit(512      , 512, 1  )# 最后输出 1 个值：价值分数
        )

    def forward(self, state, history):
        # state : [B, 492]
        # history : [B, T, 60]

        out, (h_n, _) = self.lstm(history)  # 用 LSTM 处理历史出牌
        # out : [1, T, 256]
        # h_n : [1, 1, 256]
        state = torch.cat((out[:, -1, :], state), dim=1)
        value = self.total_cross(state)
        return value

