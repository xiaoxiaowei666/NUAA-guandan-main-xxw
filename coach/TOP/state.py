# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 19:21
# @Author     : Duofeng Wu
# @File       : gd_handle.py
# @Description: 自动解析掼蛋所发送来的JSON数据

import json
import ast   # 新增：用于字符串转列表

class State(object):

    def __init__(self, render=True):
        self.render = render
        # ... 保持原样，不变 ...
        self.tribute_result = None
        self.history = {
            '0': {'send': [], 'remain': 27},
            '1': {'send': [], 'remain': 27},
            '2': {'send': [], 'remain': 27},
            '3': {'send': [], 'remain': 27},
        }
        self.remain_cards = {
            "S": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "H": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "C": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            "D": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        }
        self.play_cards = {'0': [], '1': [], '2': [], '3': []}
        self.remain_cards_classbynum = [8] * 13
        self.remain_cards_classbynum.append(2)
        self.remain_cards_classbynum.append(2)

        self._type = None
        self._stage = None
        self._myPos = None
        self._publicInfo = None
        self._actionList = None
        self._curAction = None
        self._curPos = None
        self._greaterPos = None
        self._greaterAction = None
        self._handCards = None
        self._oppoRank = None
        self._curRank = None
        self._selfRank = None
        self._antiNum = None
        self._antiPos = None
        self._result = None
        self._order = None
        self._curTimes = None
        self._settingTimes = None
        self._victoryNum = None
        self._draws = None
        self._restCards = None
        self.pass_num = 0
        self.my_pass_num = 0

        self.__parse_func = {
            ("beginning", "notify"): self.notify_begin,
            ("play", "notify"): self.notify_play,
            ("tribute", "notify"): self.notify_tribute,
            ("anti-tribute", "notify"): self.notify_anti,
            ("back", "notify"): self.notify_back,
            ("gameOver", "notify"): self.notify_game_over,
            ("episodeOver", "notify"): self.notify_episode_over,
            ("gameResult", "notify"): self.notify_game_result,
            ("play", "act"): self.act_play,
            ("tribute", "act"): self.act_tribute,
            ("back", "act"): self.act_back,
        }

    def parse(self, msg):
        assert type(msg) == dict
        for key, value in msg.items():
            setattr(self, "_{}".format(key), value)

        try:
            self.__parse_func[(self._stage, self._type)]()
            self._stage = None
            self._type = None
        except KeyError:
            print(f"未处理的消息类型: stage={self._stage}, type={self._type}")
            raise KeyError

    # 以下方法保持你原来已经写好的防御逻辑，无需改动
    def notify_begin(self):
        pass

    def notify_play(self):
        if self._curAction and len(self._curAction) >= 3 and self._curAction[2] not in ("PASS", None):
            for card in self._curAction[2]:
                if not isinstance(card, str) or len(card) < 2:

                    continue
                card_type = card[0]
                card_rank = card[1]
                if card_type not in self.remain_cards:

                    continue
                self.history[str(self._curPos)]["send"].append(card)
                self.history[str(self._curPos)]["remain"] -= 1
                card_value = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "T": 9, "J": 10,
                              "Q": 11, "K": 12, "R": 13, "B": 13}
                if card_rank not in card_value:
                    continue
                x = card_value[card_rank]
                self.remain_cards[card_type][x] -= 1
        if self._curPos == (self._myPos + 2) % 4 or self._curPos == self._myPos:
            if self._curAction[0] == "PASS":
                self.pass_num += 1
            else:
                self.pass_num = 0
        if self._curPos == self._myPos:
            if self._curAction[0] == "PASS":
                self.my_pass_num += 1
            else:
                self.my_pass_num = 0

    def notify_tribute(self):
        self.tribute_result = self._result
        for tribute_result in self._result:
            tribute_pos, receive_tribute_pos, card = tribute_result

    def notify_anti(self):
        pass

    def notify_back(self):
        for back_result in self._result:
            back_pos, receive_back_pos, card = back_result

    def notify_episode_over(self):
        self.history = {str(i): {'send': [], 'remain': 27} for i in range(4)}
        self.remain_cards = {
            "S": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "H": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "C": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            "D": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        }
        self.play_cards = {'0': [], '1': [], '2': [], '3': []}
        self.remain_cards_classbynum = [8] * 13 + [2, 2]
        self.pass_num = 0
        self.my_pass_num = 0

    def notify_game_over(self):
        self.history = {str(i): {'send': [], 'remain': 27} for i in range(4)}
        self.remain_cards = {
            "S": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "H": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "C": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            "D": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        }
        self.play_cards = {'0': [], '1': [], '2': [], '3': []}
        self.remain_cards_classbynum = [8] * 13 + [2, 2]

    def notify_game_result(self):
        print("达到设定场次, 其中0号位胜利{}次，1号位胜利{}次，2号位胜利{}次，3号位胜利{}次".format(*self._victoryNum))
        print("其中平局次数：0号位平局{}次，1号位平局{}次，2号位平局{}次，3号位平局{}次".format(*self._draws))
        pass

    def act_play(self):
        for i in range(4):
            public_info = self._publicInfo[i] if self._publicInfo and i < len(self._publicInfo) else None
            if not public_info:
                self.play_cards[str(i)] = []
                continue
            play_area = public_info.get("playArea") if isinstance(public_info, dict) else None
            if play_area is None:
                self.play_cards[str(i)] = []
            elif isinstance(play_area, dict):
                self.play_cards[str(i)] = []
            elif isinstance(play_area, list) and len(play_area) >= 3:
                self.play_cards[str(i)] = play_area[2] if isinstance(play_area[2], list) else []
            else:
                self.play_cards[str(i)] = []

    def act_tribute(self):
        pass

    def act_back(self):
        pass