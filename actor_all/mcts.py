# -*- coding: utf-8 -*-
"""
MCTS for Guandan — determinization + Q-network guided search.

在推理时用 MCTS 替代单步贪心 Q 值选择，提供前向规划能力。
核心思路：
  1. 多次采样对手手牌（determinization）
  2. 每轮采样的世界状态下运行 MCTS，用 Q 网络做先验和估值
  3. 汇总多轮采样结果的访问次数，选最佳动作
"""

import random
import math
import itertools
from collections import defaultdict

import torch
import numpy as np

from util import encode_card, process_card_list, rank2index


# ============================================================
# 常量
# ============================================================

SUITS = ['S', 'H', 'C', 'D']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

# 牌值映射：A=1, 2=2, ..., K=13, B(SB)=14, R(HR)=15
CARD_VAL = {r: i + 1 for i, r in enumerate(RANKS)}
CARD_VAL['B'] = 14
CARD_VAL['R'] = 15

# 牌型层级（越大越强，Bomb 克制一切非炸弹）
TYPE_LEVEL = {
    'Single': 1, 'Pair': 2, 'Trips': 3,
    'ThreeWithTwo': 4, 'Straight': 5,
    'ThreePair': 6, 'TwoTrips': 7,
    'StraightFlush': 8, 'Bomb': 9,
}

# 王炸（4 张鬼牌）= 最大炸弹
MAX_BOMB_LEVEL = 100


# ============================================================
# 牌 / 动作工具
# ============================================================

def card_val(c):
    """返回单张牌的数值，用于比较大小"""
    return CARD_VAL.get(c[-1], 0)


def rank_char(c):
    """返回牌的 rank 字符"""
    return c[-1]


def group_by_rank(hand):
    """将手牌按 rank 分组，返回 {rank_char: [cards]}"""
    groups = defaultdict(list)
    for c in hand:
        groups[rank_char(c)].append(c)
    return dict(groups)


def classify_action(cards):
    """
    判断一组牌的牌型。cards 是 card 字符串列表，如 ['S3', 'H3']。
    返回 (type_name, primary_value, extra_info)
      type_name: 牌型字符串
      primary_value: 主牌值（同类型比较用）
      extra_info: 额外信息（如炸弹张数）
    """
    if not cards or cards == ('PASS', 'PASS', 'PASS') or cards[0] == 'PASS':
        return ('PASS', 0, 0)

    n = len(cards)
    groups = group_by_rank(cards)
    counts = sorted(groups.values(), key=len, reverse=True)

    same_suit = all(c[0] == cards[0][0] for c in cards)

    # 炸弹判断：4 张及以上同 rank
    if n >= 4 and len(groups) == 1:
        rank = list(groups.keys())[0]
        return ('Bomb', CARD_VAL.get(rank, 0), n)

    # 王炸（全鬼牌）
    if all(c[-1] in ('B', 'R') for c in cards):
        return ('Bomb', MAX_BOMB_LEVEL, n)

    # 同花顺：5 张同花色连续
    if n == 5 and same_suit:
        vals = sorted(card_val(c) for c in cards)
        if vals == list(range(vals[0], vals[0] + 5)):
            return ('StraightFlush', vals[0], 0)

    # 顺子：5 张连续
    if n == 5 and len(groups) == 5:
        vals = sorted(card_val(c) for c in cards)
        if vals == list(range(vals[0], vals[0] + 5)):
            return ('Straight', vals[0], 0)

    # 三带二
    if n == 5 and len(counts) == 2 and len(counts[0]) == 3:
        primary_rank = rank_char(counts[0][0])
        return ('ThreeWithTwo', CARD_VAL[primary_rank], 0)

    # 三连对
    if n == 6 and len(counts) == 3 and all(len(g) == 2 for g in counts):
        vals = sorted(CARD_VAL[rank_char(g[0])] for g in counts)
        if vals == list(range(vals[0], vals[0] + 3)):
            return ('ThreePair', vals[0], 0)

    # 钢板
    if n == 6 and len(counts) == 2 and all(len(g) == 3 for g in counts):
        vals = sorted(CARD_VAL[rank_char(g[0])] for g in counts)
        if abs(vals[0] - vals[1]) == 1:
            return ('TwoTrips', min(vals), 0)

    # 三张
    if n == 3 and len(groups) == 1:
        rank = list(groups.keys())[0]
        return ('Trips', CARD_VAL.get(rank, 0), 0)

    # 对子
    if n == 2 and len(groups) == 1:
        rank = list(groups.keys())[0]
        return ('Pair', CARD_VAL.get(rank, 0), 0)

    # 单张
    if n == 1:
        return ('Single', card_val(cards[0]), 0)

    return ('Single', card_val(cards[0]), 0)


def can_beat(action_type, action_val, action_extra,
             target_type, target_val, target_extra):
    """
    判断 action 能否击败 target。
    extra: 对于 Bomb 是张数 (4/5/6...)
    """
    if action_type == 'PASS':
        return False

    # 炸弹克一切
    if action_type in ('Bomb', 'StraightFlush'):
        if target_type in ('Bomb', 'StraightFlush'):
            # 同是炸弹/同花顺：比较大小
            a_bomb_level = action_extra if action_type == 'Bomb' else 4.5
            t_bomb_level = target_extra if target_type == 'Bomb' else 4.5
            if a_bomb_level != t_bomb_level:
                return a_bomb_level > t_bomb_level
            return action_val > target_val
        return True

    # 非炸弹只能接同类型且值更大
    if action_type == target_type:
        return action_val > target_val

    return False


def action_to_cards(action_entry):
    """把服务器 actionList 条目转为卡牌字符串列表（兼容真实服务器和模拟器）"""
    if action_entry is None:
        return []
    if isinstance(action_entry, (list, tuple)):
        if len(action_entry) >= 3 and isinstance(action_entry[2], list):
            return action_entry[2]
        if len(action_entry) == 1 and isinstance(action_entry[0], str):
            return [action_entry[0]]
    return []


# ============================================================
# 简化游戏状态（用于 MCTS 模拟）
# ============================================================

class SimGameState:
    """
    MCTS 内部使用的轻量游戏状态。
    不模拟完整牌局，只追踪一轮 trick 内的出牌序列。
    """
    def __init__(self, my_hand, my_pos, current_rank,
                 cur_trick_type, cur_trick_val, cur_trick_extra,
                 cur_leader, trick_starter,
                 known_played, total_unseen, opp_hands):
        """
        my_hand: list[str] — 我的手牌
        my_pos: int 0-3
        current_rank: str — 当前级牌
        cur_trick_*: 当前最大牌型信息
        cur_leader: 当前领先玩家
        trick_starter: 本轮出牌起始者
        known_played: 4×15 tensor — 已出牌统计（用于构造状态）
        total_unseen: int — 未见的牌总数
        opp_hands: list[set] — 采样出的对手/队友手牌（各为 set of card str）
        """
        self.my_hand = list(my_hand)
        self.my_pos = my_pos
        self.current_rank = current_rank

        self.trick_type = cur_trick_type
        self.trick_val = cur_trick_val
        self.trick_extra = cur_trick_extra
        self.trick_leader = cur_leader
        self.trick_starter = trick_starter

        self.known_played = known_played
        self.total_unseen = total_unseen
        self.opp_hands = opp_hands  # 4 个 set

        self.pass_count = 0
        self.trick_done = False

    def copy(self):
        s = SimGameState.__new__(SimGameState)
        s.my_hand = list(self.my_hand)
        s.my_pos = self.my_pos
        s.current_rank = self.current_rank
        s.trick_type = self.trick_type
        s.trick_val = self.trick_val
        s.trick_extra = self.trick_extra
        s.trick_leader = self.trick_leader
        s.trick_starter = self.trick_starter
        s.known_played = self.known_played
        s.total_unseen = self.total_unseen
        s.opp_hands = [set(h) for h in self.opp_hands]
        s.pass_count = self.pass_count
        s.trick_done = self.trick_done
        return s

    def current_player(self):
        """返回当前应出牌的玩家 (0-3)，基于 trick 状态"""
        if self.trick_done:
            return self.trick_starter
        if self.trick_type is None:
            return self.trick_starter
        return (self.trick_leader + 1) % 4

    def hand_of(self, player):
        if player == self.my_pos:
            return self.my_hand
        return list(self.opp_hands[player])

    def apply_action(self, player, action_cards):
        """
        action_cards: list[str] 或 ('PASS','PASS','PASS')
        更新游戏状态并返回新的 current_player
        """
        is_pass = (not action_cards or
                   (isinstance(action_cards, tuple) and action_cards[0] == 'PASS') or
                   (isinstance(action_cards, list) and len(action_cards) > 0
                    and isinstance(action_cards[0], str) and action_cards[0] == 'PASS'))

        if is_pass:
            self.pass_count += 1
            if self.pass_count >= 3:
                self.trick_done = True
                self.trick_type = None
                self.trick_val = 0
                self.trick_extra = 0
                self.trick_leader = self.trick_starter
                self.pass_count = 0
            return

        # 出牌
        action_cards_list = list(action_cards)
        a_type, a_val, a_extra = classify_action(action_cards_list)

        # 从手牌移除
        if player == self.my_pos:
            for c in action_cards_list:
                if c in self.my_hand:
                    self.my_hand.remove(c)
        else:
            for c in action_cards_list:
                if c in self.opp_hands[player]:
                    self.opp_hands[player].discard(c)

        self.trick_type = a_type
        self.trick_val = a_val
        self.trick_extra = a_extra
        self.trick_leader = player
        self.trick_starter = player
        self.pass_count = 0

    def is_terminal(self):
        """简化判断：任一手牌为空则终局"""
        if len(self.my_hand) == 0:
            return True
        for h in self.opp_hands:
            if len(h) == 0:
                return True
        return False


# ============================================================
# 动作枚举器（用于 MCTS 模拟中的对手回合）
# ============================================================

def enumerate_actions(hand, trick_type, trick_val, trick_extra):
    """
    从手牌中生成可行动作列表（用于模拟）。
    hand: list[str] 卡牌字符串
    trick_type/val/extra: 当前牌型信息，None 表示轮到自己出牌（无限制）

    返回 [(action_cards, type, val, extra), ...]
    """
    if not hand:
        return [(('PASS', 'PASS', 'PASS'), 'PASS', 0, 0)]

    groups = group_by_rank(hand)

    actions = []
    # PASS 总是可选
    actions.append((('PASS', 'PASS', 'PASS'), 'PASS', 0, 0))

    if trick_type is None:
        # 自由出牌：枚举所有可能
        # 单张
        for c in hand:
            actions.append(([c], 'Single', card_val(c), 0))
        # 对子
        for rank, cards in groups.items():
            if len(cards) >= 2:
                for combo in itertools.combinations(cards, 2):
                    actions.append((list(combo), 'Pair', CARD_VAL.get(rank, 0), 0))
            # 三张
            if len(cards) >= 3:
                for combo in itertools.combinations(cards, 3):
                    actions.append((list(combo), 'Trips', CARD_VAL.get(rank, 0), 0))
            # 炸弹
            if len(cards) >= 4:
                for combo in itertools.combinations(cards, 4):
                    actions.append((list(combo), 'Bomb', CARD_VAL.get(rank, 0), 4))
        # 顺子（简化：5连）
        _add_straight_actions(hand, groups, actions)
        # 三带二
        _add_three_with_two_actions(groups, actions)
    else:
        # 跟牌：只需生成能击败当前牌的候选
        # 同类型的更大牌
        _add_same_type_beaters(hand, groups, trick_type, trick_val, trick_extra, actions)
        # 炸弹
        _add_bomb_beaters(hand, groups, trick_type, trick_val, trick_extra, actions)

    return actions


def _add_straight_actions(hand, groups, actions):
    """添加顺子候选（5连）"""
    ranks_present = defaultdict(list)
    for c in hand:
        ranks_present[rank_char(c)].append(c)
    # 查找连续 5 个 rank
    normal_ranks = [r for r in RANKS if r in ranks_present]
    for i in range(len(normal_ranks) - 4):
        window = normal_ranks[i:i + 5]
        window_vals = [CARD_VAL[r] for r in window]
        expected = list(range(window_vals[0], window_vals[0] + 5))
        if window_vals == expected:
            cards = []
            for r in window:
                cards.append(ranks_present[r][0])
            actions.append((cards, 'Straight', window_vals[0], 0))


def _add_three_with_two_actions(groups, actions):
    """添加三带二候选"""
    trips_ranks = []
    pair_ranks = []
    for rank, cards in groups.items():
        if len(cards) >= 3:
            trips_ranks.append((rank, cards[:3]))
        if len(cards) >= 2:
            pair_ranks.append((rank, cards[:2]))
    for trip_rank, trip_cards in trips_ranks:
        for pair_rank, pair_cards in pair_ranks:
            if trip_rank != pair_rank:
                acts = trip_cards + pair_cards
                actions.append((acts, 'ThreeWithTwo', CARD_VAL[trip_rank], 0))


def _add_same_type_beaters(hand, groups, trick_type, trick_val, trick_extra, actions):
    """生成同类型但更大的候选动作"""
    if trick_type == 'Single':
        for c in hand:
            if card_val(c) > trick_val:
                actions.append(([c], 'Single', card_val(c), 0))

    elif trick_type == 'Pair':
        for rank, cards in groups.items():
            if len(cards) >= 2 and CARD_VAL.get(rank, 0) > trick_val:
                actions.append((list(cards[:2]), 'Pair', CARD_VAL[rank], 0))

    elif trick_type == 'Trips':
        for rank, cards in groups.items():
            if len(cards) >= 3 and CARD_VAL.get(rank, 0) > trick_val:
                actions.append((list(cards[:3]), 'Trips', CARD_VAL[rank], 0))

    elif trick_type == 'ThreeWithTwo':
        trips_ranks = [(r, cs[:3]) for r, cs in groups.items()
                       if len(cs) >= 3 and CARD_VAL.get(r, 0) > trick_val]
        pair_ranks = [(r, cs[:2]) for r, cs in groups.items() if len(cs) >= 2]
        for tr, tc in trips_ranks:
            for pr, pc in pair_ranks:
                if tr != pr:
                    actions.append((tc + pc, 'ThreeWithTwo', CARD_VAL[tr], 0))

    elif trick_type == 'Straight':
        # 简化：只找起始值更大的顺子
        ranks_present = defaultdict(list)
        for c in hand:
            ranks_present[rank_char(c)].append(c)
        normal_ranks = [r for r in RANKS if r in ranks_present]
        for i in range(len(normal_ranks) - 4):
            window = normal_ranks[i:i + 5]
            wv = [CARD_VAL[r] for r in window]
            expected = list(range(wv[0], wv[0] + 5))
            if wv == expected and wv[0] > trick_val:
                cards = [ranks_present[r][0] for r in window]
                actions.append((cards, 'Straight', wv[0], 0))


def _add_bomb_beaters(hand, groups, trick_type, trick_val, trick_extra, actions):
    """添加能击败当前牌的炸弹候选"""
    # 如果当前已经是炸弹/同花顺
    if trick_type in ('Bomb', 'StraightFlush'):
        t_level = trick_extra if trick_type == 'Bomb' else 4.5
        for rank, cards in groups.items():
            if len(cards) >= 4:
                cv = CARD_VAL.get(rank, 0)
                for k in range(4, len(cards) + 1):
                    b_level = k
                    if b_level > t_level or (b_level == t_level and cv > trick_val):
                        actions.append((list(cards[:k]), 'Bomb', cv, k))
                        break
    else:
        # 任何炸弹都能击败非炸弹
        for rank, cards in groups.items():
            if len(cards) >= 4:
                cv = CARD_VAL.get(rank, 0)
                actions.append((list(cards[:4]), 'Bomb', cv, 4))


def pick_best_actions(actions, max_actions=20):
    """
    从候选动作中选出最有代表性的（按牌型优先、值最大优先）。
    用于限制模拟分支因子。
    """
    if len(actions) <= max_actions:
        return actions

    # 分类：PASS + 各类型 top-K
    pass_acts = [a for a in actions if a[1] == 'PASS']
    typed = defaultdict(list)
    for a in actions:
        if a[1] != 'PASS':
            typed[a[1]].append(a)

    result = list(pass_acts)
    for t, acts in typed.items():
        # 每种类型保留值最大的几个
        acts_sorted = sorted(acts, key=lambda x: x[2], reverse=True)
        kept = acts_sorted[:max(1, max_actions // (len(typed) + 1))]
        result.extend(kept)

    return result[:max_actions]


# ============================================================
# 状态编码（为 Q 网络构建输入张量）
# ============================================================

def _one_hot(val, num_classes):
    t = torch.zeros(num_classes)
    if 0 <= val < num_classes:
        t[val] = 1
    return t


def build_sim_state_tensor(sim_state, perspective):
    """
    从 SimGameState 构造 StateCatEmbedding 风格的 625 维张量。
    perspective: 模拟视角的玩家位置 (0-3)
    """
    my_pos = perspective
    teammate_pos = (my_pos + 2) % 4

    # 手牌
    hand = sim_state.hand_of(my_pos)
    hand_tensor = encode_card(hand).flatten()

    # 出牌区
    play_area_tensors = []
    for p in range(4):
        last_play = []
        if p == sim_state.trick_leader and sim_state.trick_type is not None:
            last_play = _last_played_cards(sim_state, p)
        play_area_tensors.append(encode_card(last_play).flatten())

    # 余牌数
    rest_tensors = []
    for p in range(4):
        rest = len(sim_state.hand_of(p))
        rest_tensors.append(_one_hot(min(rest, 29), 30))

    # 级牌
    rank_idx = rank2index.get(sim_state.current_rank, 0)
    rank_tensor = _one_hot(rank_idx, 13)

    # 位置
    my_pos_t = _one_hot(my_pos, 4)
    teammate_pos_t = _one_hot(teammate_pos, 4)

    # 队友出牌区
    teammate_play = encode_card(_last_played_cards(sim_state, teammate_pos)).flatten()

    # 当前最大牌
    greater_cards = _last_played_cards(sim_state, sim_state.trick_leader) if sim_state.trick_type else []
    greater_t = encode_card(greater_cards).flatten()

    # 当前最大牌玩家
    leader = sim_state.trick_leader if sim_state.trick_type else sim_state.trick_starter
    leader_t = _one_hot(leader, 4)

    # 累计已出牌
    round_played = sim_state.known_played.clone().detach()

    return torch.cat((
        hand_tensor.flatten(),               # 60
        play_area_tensors[0].flatten(),      # 60
        play_area_tensors[1].flatten(),      # 60
        play_area_tensors[2].flatten(),      # 60
        play_area_tensors[3].flatten(),      # 60
        torch.cat(rest_tensors).flatten(),   # 120
        rank_tensor.flatten(),               # 13
        my_pos_t.flatten(),                  # 4
        teammate_pos_t.flatten(),            # 4
        teammate_play.flatten(),             # 60
        greater_t.flatten(),                 # 60
        leader_t.flatten(),                  # 4
        round_played.flatten(),              # 60
    ), dim=0)


def _last_played_cards(sim_state, player):
    """获取某玩家最近一次出的牌（简化：在模拟中我只追踪 trick_leader 的牌）"""
    # MCTS 模拟中不追踪完整历史，返回空（对短模拟影响小）
    return []


# ============================================================
# MCTS 节点 & 搜索
# ============================================================

class MCTSNode:
    __slots__ = ('action_idx', 'parent', 'children',
                 'visit_count', 'total_value', 'prior')

    def __init__(self, action_idx=-1, parent=None, prior=0.0):
        self.action_idx = action_idx  # 到达此节点的动作索引
        self.parent = parent
        self.children = {}            # action_idx → MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


def _ucb_score(node, parent_visits, c_puct=1.4):
    """UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)"""
    q = node.value
    p = node.prior if node.prior > 0 else 1e-6
    u = c_puct * p * math.sqrt(parent_visits) / (1 + node.visit_count)
    return q + u


class MCTS:
    """
    MCTS 搜索器（批量计算优化版）。
    尽可能将所有动作的 Q 值计算合并为单次前向传播，大幅提速。
    默认参数已针对 CPU 推理调优（每步决策约 1-3 秒）。
    """

    def __init__(self, model, device='cpu',
                 num_simulations=30, max_depth=3,
                 c_puct=1.4, temperature=0.5,
                 num_determinizations=2,
                 epsilon=0.1):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_determinizations = num_determinizations
        self.epsilon = epsilon

    # ----------------------------------------------------------
    # 核心：批量 Q 值计算（一次前向传播）
    # ----------------------------------------------------------
    def _batch_q(self, state_tensor, action_list, history_tensor):
        """
        批量计算所有动作的 Q 值。
        state_tensor: 625 维状态张量
        action_list: [(action_cards, type, val, extra), ...] 或服务器 actionList 格式
        history_tensor: LSTM 历史 [1, T, 60]

        返回: torch.Tensor [N] Q 值
        """
        if not action_list:
            return torch.tensor([])

        n = len(action_list)
        hist_batch = history_tensor.expand(n, -1, -1).float().to(self.device)

        inps = []
        for item in action_list:
            if isinstance(item, tuple) and len(item) == 4:
                # 模拟器格式: (action_cards, type, val, extra)
                act_cards = list(item[0])
            else:
                # 服务器格式: ['Single', '3', ['S3']] 等
                act_cards = process_card_list(item)
            act_t = encode_card(act_cards).flatten().to(self.device)
            inps.append(torch.cat((state_tensor.flatten().to(self.device), act_t)))

        batched_inp = torch.stack(inps, dim=0)  # [N, 685]

        with torch.no_grad():
            qs = self.model(batched_inp, hist_batch).squeeze(-1)  # [N]
        return qs

    # ----------------------------------------------------------
    # 公共入口
    # ----------------------------------------------------------
    def search(self, state_msg, action_list, act_range,
               history_tensor, played_cards_tensor, hand_cards):
        my_pos = state_msg.get('myPos', 0)
        current_rank = state_msg.get('curRank', '2')

        cur_trick_type, cur_trick_val, cur_trick_extra = \
            _parse_current_trick(state_msg)
        trick_starter = _get_trick_starter(state_msg, my_pos)
        unseen = _get_unseen_cards(hand_cards, played_cards_tensor)

        # 汇总多轮 determinization
        total_visits = [0] * (act_range + 1)

        for _ in range(self.num_determinizations):
            opp_hands = _sample_opp_hands(unseen, my_pos, hand_cards)
            sim_state = SimGameState(
                my_hand=hand_cards, my_pos=my_pos, current_rank=current_rank,
                cur_trick_type=cur_trick_type, cur_trick_val=cur_trick_val,
                cur_trick_extra=cur_trick_extra,
                cur_leader=_get_cur_leader(state_msg, my_pos),
                trick_starter=trick_starter,
                known_played=played_cards_tensor,
                total_unseen=len(unseen), opp_hands=opp_hands,
            )
            visits, _ = self._run_mcts(
                sim_state, action_list, act_range,
                history_tensor
            )
            for i in range(act_range + 1):
                total_visits[i] += visits[i]

        best_idx = self._select_best(total_visits, action_list, act_range)
        total_v = sum(total_visits) or 1
        probs = {i: total_visits[i] / total_v for i in range(act_range + 1)}
        return best_idx, probs

    # ----------------------------------------------------------
    # MCTS 核心
    # ----------------------------------------------------------
    def _run_mcts(self, sim_state, action_list, act_range, history_tensor):
        """单轮 determinization 下的 MCTS"""
        # 批量计算根节点所有动作的先验 Q 值 → 一次前向传播
        root_state_t = build_sim_state_tensor(sim_state, sim_state.my_pos)
        server_actions = list(action_list[:act_range + 1])
        root_qs = self._batch_q(root_state_t, server_actions, history_tensor)
        prior = torch.softmax(root_qs / self.temperature, dim=0).tolist()

        root = MCTSNode(action_idx=-1, prior=1.0)
        for i in range(act_range + 1):
            root.children[i] = MCTSNode(action_idx=i, parent=root, prior=prior[i])

        total_sims = self.num_simulations
        for _ in range(total_sims):
            # 选择
            node = root
            while not node.is_leaf():
                best_child_idx = max(
                    node.children.keys(),
                    key=lambda k: _ucb_score(node.children[k], node.visit_count, self.c_puct)
                )
                node = node.children[best_child_idx]

            # 评估
            if node is root:
                value = root_qs[node.children[list(node.children.keys())[0]].action_idx].item()
            else:
                value = self._evaluate_leaf(node, sim_state, action_list, history_tensor)

            # 反向传播
            self._backprop_to_root(node, value)

        visits = [0] * (act_range + 1)
        for i, child in root.children.items():
            visits[i] = child.visit_count
        values = [child.value if child.visit_count > 0 else 0.0
                  for i, child in root.children.items()]
        return visits, values

    def _evaluate_leaf(self, node, sim_state, action_list, history_tensor):
        """叶节点估值：用批量 Q + 浅 rollout"""
        state = sim_state.copy()
        act_idx = node.action_idx
        if act_idx < 0:
            return 0.0

        act_entry = action_list[act_idx]
        act_cards = process_card_list(act_entry)
        state.apply_action(state.my_pos, act_cards)

        if state.is_terminal():
            return 10.0 if len(state.my_hand) == 0 else -10.0

        return self._quick_rollout(state, history_tensor, depth=1)

    def _quick_rollout(self, sim_state, history_tensor, depth):
        """快速 rollout：所有玩家都用 Q 网络选牌（批量计算），对手/队友 softmax 采样"""
        if depth >= self.max_depth or sim_state.is_terminal():
            if sim_state.is_terminal():
                return 10.0 if len(sim_state.my_hand) == 0 else -10.0
            return self._max_q_batched(sim_state, history_tensor)

        cur_player = sim_state.current_player()
        hand = sim_state.hand_of(cur_player)
        actions = enumerate_actions(
            hand, sim_state.trick_type,
            sim_state.trick_val, sim_state.trick_extra
        )
        pruned = pick_best_actions(actions, max_actions=12)
        non_pass = [a for a in pruned if a[1] != 'PASS']

        if not non_pass:
            chosen = pruned[0]  # 只能 PASS
        elif cur_player == sim_state.my_pos:
            # 我: epsilon-greedy
            if random.random() < self.epsilon:
                chosen = random.choice(non_pass)
            else:
                state_t = build_sim_state_tensor(sim_state, cur_player)
                qs = self._batch_q(state_t, non_pass, history_tensor)
                chosen = non_pass[qs.argmax().item()]
        elif cur_player == (sim_state.my_pos + 2) % 4:
            # 队友: argmax Q（同目标）
            state_t = build_sim_state_tensor(sim_state, cur_player)
            qs = self._batch_q(state_t, non_pass, history_tensor)
            chosen = non_pass[qs.argmax().item()]
        else:
            # 对手: softmax(Q) 采样（合理的出牌 + 随机性模拟不确定性）
            state_t = build_sim_state_tensor(sim_state, cur_player)
            qs = self._batch_q(state_t, non_pass, history_tensor)
            probs = torch.softmax(qs / 1.0, dim=0).tolist()
            chosen = _weighted_choice(non_pass, probs)

        act_cards = list(chosen[0])
        sim_state.apply_action(cur_player, act_cards)
        return self._quick_rollout(sim_state, history_tensor, depth + 1)

    def _max_q_batched(self, sim_state, history_tensor):
        """批量计算状态下所有可行动作的最大 Q 值"""
        hand = sim_state.hand_of(sim_state.my_pos)
        actions = enumerate_actions(
            hand, sim_state.trick_type,
            sim_state.trick_val, sim_state.trick_extra
        )
        pruned = pick_best_actions(actions, max_actions=15)

        non_pass = [a for a in pruned if a[1] != 'PASS']
        if not non_pass:
            return 0.0

        state_t = build_sim_state_tensor(sim_state, sim_state.my_pos)
        qs = self._batch_q(state_t, non_pass, history_tensor)
        return qs.max().item()

    def _select_best(self, visits, action_list, act_range):
        pass_idx = None
        for i in range(act_range + 1):
            if i < len(action_list) and action_list[i][0] == 'PASS':
                pass_idx = i
                break
        if pass_idx is not None and act_range > 0:
            masked = visits.copy()
            masked[pass_idx] = -1
            return int(np.argmax(masked))
        return int(np.argmax(visits))

    def _backprop_to_root(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent


# ============================================================
# 辅助函数
# ============================================================

def _parse_current_trick(state_msg):
    """从服务器消息中解析当前牌型信息"""
    ga = state_msg.get('greaterAction', None)
    gp = state_msg.get('greaterPos', None)

    # 判断哪个是牌型数据（list/dict），哪个是位置（int）
    trick_action = None
    if isinstance(ga, (list, dict)) and ga:
        trick_action = ga
    elif isinstance(gp, (list, dict)) and gp:
        trick_action = gp

    if trick_action is None:
        return None, 0, 0

    cards = action_to_cards(trick_action)
    if not cards:
        return None, 0, 0

    return classify_action(cards)


def _get_cur_leader(state_msg, my_pos):
    """获取当前领先玩家"""
    ga = state_msg.get('greaterAction', None)
    gp = state_msg.get('greaterPos', None)
    if isinstance(gp, int) and 0 <= gp <= 3:
        return gp
    if isinstance(ga, int) and 0 <= ga <= 3:
        return ga
    return my_pos


def _get_trick_starter(state_msg, my_pos):
    """推测本轮起始者"""
    leader = _get_cur_leader(state_msg, my_pos)
    return leader  # 简化：假设领先者 = 起始者


def _get_unseen_cards(hand, played_tensor):
    """
    返回未见牌列表（2副牌 = 108张：52×2 + SB×2 + HR×2）。
    牌用 (suit, rank) 元组表示，如 ('S', '3')，joker 用 ('J', 0)/( 'J', 1)。
    """
    from collections import Counter

    # 完整牌库：每种牌 2 张
    full_counts = Counter()
    for suit in range(4):
        for rank in range(13):
            full_counts[(suit, rank)] = 2
    full_counts[('J', 0)] = 2  # SB
    full_counts[('J', 1)] = 2  # HR

    # 扣除手牌
    for c in hand:
        key = _card_to_key(c)
        if full_counts.get(key, 0) > 0:
            full_counts[key] -= 1

    # 扣除已出牌 (4×15 tensor)
    if played_tensor is not None:
        pt = played_tensor
        for s in range(4):
            for r in range(15):
                cnt = int(pt[s, r].item())
                if cnt > 0:
                    if r < 13:
                        key = (s, r)
                    elif r == 13:
                        key = ('J', 0)
                    else:
                        key = ('J', 1)
                    full_counts[key] = max(0, full_counts.get(key, 0) - cnt)

    # 展开为卡牌字符串列表
    unseen = []
    for key, count in full_counts.items():
        for _ in range(count):
            unseen.append(_key_to_card(key))
    return unseen


def _card_to_key(c):
    """'S3' → (0,2), 'SB' → ('J',0), 'HR' → ('J',1)"""
    if c == 'SB':
        return ('J', 0)
    if c == 'HR':
        return ('J', 1)
    suit = SUITS.index(c[0])
    rank = RANKS.index(c[1])
    return (suit, rank)


def _key_to_card(key):
    """(0,2) → 'S3', ('J',0) → 'SB', ('J',1) → 'HR'"""
    if key[0] == 'J':
        return 'SB' if key[1] == 0 else 'HR'
    return SUITS[key[0]] + RANKS[key[1]]


def _weighted_choice(items, probs):
    """加权随机选择"""
    r = random.random()
    cum = 0.0
    for item, p in zip(items, probs):
        cum += p
        if r <= cum:
            return item
    return items[-1]


def _sample_opp_hands(unseen, my_pos, my_hand):
    """
    从未见牌中随机分配给对手。
    返回 list[set]，索引为玩家位置。
    """
    opp_hands = [set() for _ in range(4)]
    opp_hands[my_pos] = set(my_hand)

    remaining = list(unseen)
    random.shuffle(remaining)

    # 平均分配
    n = len(remaining)
    per_player = n // 3
    remainder = n % 3

    opp_indices = [p for p in range(4) if p != my_pos]
    start = 0
    for i, pid in enumerate(opp_indices):
        extra = 1 if i < remainder else 0
        end = start + per_player + extra
        opp_hands[pid] = set(remaining[start:end])
        start = end

    return opp_hands
