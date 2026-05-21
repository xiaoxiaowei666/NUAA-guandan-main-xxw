# -*- coding: utf-8 -*-
"""
TOP 结构安全过滤器
复用 TOP 的 combine_handcards 做手牌结构分析，
过滤掉「明显拆家」的动作候选，不替代 RL 决策。
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from coach.TOP.utils import combine_handcards, cal_bomb_num


def _build_card_val(rank):
    """构建与 TOP 一致的牌值映射表"""
    card_val = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
                "B": 16, "R": 17}
    card_val[rank] = 15
    return card_val


def compute_safety_mask(msg, action_list, my_pos):
    """
    用 TOP 的结构分析生成安全掩码。

    参数:
        msg: 服务器消息 dict
        action_list: 候选动作列表
        my_pos: 我方位置 (0-3)

    返回:
        list[bool]: 长度同 action_list。
                    True  = 安全，RL 可自由选择
                    False = 结构不安全，RL 应避开
    """
    n = len(action_list)
    mask = [True] * n  # 默认全部安全

    if n <= 1:
        return mask  # 只有一个候选（只能是 PASS），不过滤

    hand_cards = msg.get('handCards', [])
    if not hand_cards:
        return mask

    rank = msg.get('curRank', '2')
    card_val = _build_card_val(rank)
    rank_card = 'H' + rank

    # ---------- TOP 手牌结构分析 ----------
    sorted_cards, bomb_info = combine_handcards(hand_cards, rank, card_val)

    # 提取结构成员牌
    bomb_members = set()
    for bomb in sorted_cards.get('Bomb', []):
        bomb_members.update(bomb)

    straight_members = set()
    if sorted_cards.get('StraightFlush', []):
        straight_members.update(sorted_cards['StraightFlush'][0])
    elif sorted_cards.get('Straight', []):
        straight_members.update(sorted_cards['Straight'][0])

    pair_members = set()
    for pair in sorted_cards.get('Pair', []):
        pair_members.update(pair)

    # 级牌（百搭牌）
    rank_cards_in_hand = {c for c in hand_cards if c == rank_card}
    has_rank_card = len(rank_cards_in_hand) > 0

    # 队友位置
    teammate_pos = (my_pos + 2) % 4
    greater_pos = _get_greater_pos(msg)
    teammate_leads = (greater_pos == teammate_pos)

    # 下家剩余牌数
    public_info = msg.get('publicInfo', [])
    next_pos = (my_pos + 1) % 4
    next_rest = 27
    if public_info and next_pos < len(public_info):
        next_rest = public_info[next_pos].get('rest', 27)

    # ----------------------------------------------------------
    # 遍历每个候选动作，标记不安全的
    # ----------------------------------------------------------
    for i, act in enumerate(action_list):
        act_type = act[0]

        # PASS 永远安全
        if act_type == 'PASS':
            continue

        act_cards = _extract_cards(act)
        act_set = set(act_cards)

        # ---- 规则 0: 一步清空手牌 → 永远保留 ----
        if len(act_cards) == len(hand_cards):
            mask[i] = True
            continue

        # ---- 规则 1: 拆炸弹 ----
        # 如果动作包含炸弹的部分牌但不是完整炸弹 → 拆弹
        if _breaks_structure(act_set, bomb_members, sorted_cards.get('Bomb', [])):
            mask[i] = False
            continue

        # ---- 规则 2: 拆同花顺/顺子 ----
        if act_set & straight_members:
            if not _fully_contains(act_set, sorted_cards.get('StraightFlush', [])):
                if not _fully_contains(act_set, sorted_cards.get('Straight', [])):
                    mask[i] = False
                    continue

        # ---- 规则 3: 抢队友牌 ----
        if teammate_leads:
            mask[i] = False
            continue

        # ---- 规则 4: 无意义使用级牌 ----
        if has_rank_card and (act_set & rank_cards_in_hand):
            # 检查是否存在不含级牌的同类替代动作
            has_alternative = False
            for j, other in enumerate(action_list):
                if j != i and other[0] == act_type and other[0] != 'Bomb':
                    other_cards = set(_extract_cards(other))
                    if not (other_cards & rank_cards_in_hand):
                        has_alternative = True
                        break
            if has_alternative:
                mask[i] = False
                continue

        # ---- 规则 5: 下家快赢(≤4张)时阻止 PASS ----
        if act_type == 'PASS' and act_range > 0 and next_rest <= 4:
            # 存在非炸弹压制选项时才过滤 PASS
            has_non_bomb = any(
                action_list[j][0] != 'Bomb' and action_list[j][0] != 'StraightFlush'
                for j in range(len(action_list))
                if j != i and j <= act_range
            )
            if has_non_bomb:
                mask[i] = False
                continue

    return mask


# ============================================================
# 辅助函数
# ============================================================

def _get_greater_pos(msg):
    """提取当前最大牌的出牌者位置"""
    gp = msg.get('greaterPos', -1)
    ga = msg.get('greaterAction', -1)
    if isinstance(gp, int) and 0 <= gp <= 3:
        return gp
    if isinstance(ga, int) and 0 <= ga <= 3:
        return ga
    return -1


def _extract_cards(action):
    """从 actionList 条目中提取卡牌字符串列表"""
    if action is None:
        return []
    if len(action) >= 3 and isinstance(action[2], list):
        return action[2]
    return []


def _breaks_structure(act_set, structure_cards, full_structures):
    """
    判断动作是否破坏了某个结构。
    act_set: 动作中的牌集合
    structure_cards: 所有属于该类型结构的牌
    full_structures: 完整结构的列表（每个是一个牌列表）
    """
    if not structure_cards:
        return False
    # 如果动作牌和结构牌有交集
    overlap = act_set & structure_cards
    if not overlap:
        return False
    # 检查是否是完整地出掉某个结构（允许）
    for full_struct in full_structures:
        if act_set.issuperset(set(full_struct)):
            return False
    # 有交集但不是完整结构 → 正在拆结构
    return True


def _fully_contains(act_set, full_structures):
    """动作是否完整包含某个结构（即一次性出掉整个结构）"""
    if not full_structures:
        return False
    for fs in full_structures:
        if act_set.issuperset(set(fs)):
            return True
    return False


# ============================================================
# 策略引导：安全候选里「更好的」加点分
# ============================================================

# 主动出牌时牌型优先级（越高越优先出）
TYPE_PRIORITY = {
    'ThreePair': 0.25, 'TwoTrips': 0.25,
    'Straight': 0.20,
    'ThreeWithTwo': 0.15,
    'Trips': 0.10,
    'Pair': 0.05,
    'Single': 0.00,
    'Bomb': 0.00,       # 炸弹留着防守
    'StraightFlush': 0.00,
}


def compute_strategy_bonus(msg, action_list, my_pos):
    """
    策略引导加分：安全候选里，用 TOP 的棋感给更好的动作轻微加分。
    量级 0.05~0.25，远小于终局奖励 ±5，只做方向引导。

    返回: list[float]，长度同 action_list，每个元素是对应动作的 Q 值加分
    """
    n = len(action_list)
    bonus = [0.0] * n

    if n <= 1:
        return bonus

    rank = msg.get('curRank', '2')
    card_val = _build_card_val(rank)

    greater_pos = _get_greater_pos(msg)
    is_active = (greater_pos == -1 or msg.get('curPos', -1) == -1)

    # 下家信息
    public_info = msg.get('publicInfo', [])
    next_pos = (my_pos + 1) % 4
    next_rest = 27
    if public_info and next_pos < len(public_info):
        next_rest = public_info[next_pos].get('rest', 27)

    # ----------------------------------------------------------
    # 策略 A: 主动出牌 — 牌型优先级
    # ----------------------------------------------------------
    if is_active:
        for i, act in enumerate(action_list):
            act_type = act[0]
            if act_type == 'PASS':
                continue
            if act_type in TYPE_PRIORITY:
                bonus[i] += TYPE_PRIORITY[act_type]

    # ----------------------------------------------------------
    # 策略 B: 剩余牌分析 — 跟牌时评估压制安全性
    # ----------------------------------------------------------
    if not is_active and greater_pos >= 0:
        cur_action = msg.get('curAction', None)
        if cur_action and cur_action[0] != 'PASS':
            cur_val = card_val.get(cur_action[1], 0)

            for i, act in enumerate(action_list):
                act_type = act[0]
                if act_type == 'PASS':
                    continue
                act_val = card_val.get(act[1], 0)

                # 高值牌（≥A=14）压制 → 大概率稳赢这一轮
                if act_val >= 14 and act_val > cur_val:
                    bonus[i] += 0.12
                # 比当前牌大 4 级以上 → 较安全
                elif act_val - cur_val >= 4:
                    bonus[i] += 0.06

    # ----------------------------------------------------------
    # 策略 C: 炸弹选择 — 多个炸弹能压制时，优先小炸弹
    # ----------------------------------------------------------
    if not is_active and greater_pos >= 0:
        bomb_actions = [(i, act) for i, act in enumerate(action_list)
                        if act[0] in ('Bomb', 'StraightFlush')]
        if len(bomb_actions) >= 2:
            # 按炸弹值排序
            bomb_actions.sort(key=lambda x: card_val.get(x[1][1], 0))
            # 最小炸弹 +0.10，其余根据差距递减
            min_val = card_val.get(bomb_actions[0][1][1], 0)
            for i, act in bomb_actions:
                act_val = card_val.get(act[1], 0)
                if act_val == min_val:
                    bonus[i] += 0.10
                else:
                    # 差距越大扣越多（但不过滤，只是降权）
                    penalty = min(0.08, (act_val - min_val) * 0.01)
                    bonus[i] -= penalty

    # ----------------------------------------------------------
    # 策略 D: 下家只剩 1 张时，避免出单张
    # ----------------------------------------------------------
    if next_rest == 1:
        for i, act in enumerate(action_list):
            if act[0] == 'Single':
                bonus[i] -= 0.20
            elif act[0] == 'Pair':
                bonus[i] -= 0.10

    return bonus
