import json
from collections import defaultdict

import pygame
from pygame.locals import *
from ws4py.client.threadedclient import WebSocketClient
import os
import ast

import logging
from pathlib import Path

_SHOWDOWN_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("game.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
pos = 0

def safe_wait(seconds):
    """非阻塞等待（处理退出事件）"""
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < seconds * 1000:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
        pygame.event.pump()
        pygame.time.wait(10)


class Main(WebSocketClient):
    def __init__(self, url, render=True):
        super().__init__(url)
        self.running = True  # <-- 添加这一行
        self.render = render
        self.waiting_for_action = False
        self.current_action_result = None
        self.hint_index = 0  # 提示循环索引
        # 初始化 pygame
        log.info("初始化 pygame...")
        pygame.init()
        pygame.display.set_caption("升级对战")

        # 字体
        font_path = str(_SHOWDOWN_ROOT / 'fonts' / 'HanYiRegular.ttf')
        if not os.path.exists(font_path):
            font_path = None
        self.font = pygame.font.Font(font_path, 20)
        self.font_button = pygame.font.Font(font_path, 24)

        # 路径
        self.cards_image_path = str(_SHOWDOWN_ROOT / 'images' / 'cards') + os.sep
        self.background_image_path = str(_SHOWDOWN_ROOT / 'images' / 'background.png')
        self.partbackground_image_path = str(_SHOWDOWN_ROOT / 'images' / 'partback.bmp')
        self.chup_image_path = str(_SHOWDOWN_ROOT / 'images' / 'chup.png')
        self.buc_image_path = str(_SHOWDOWN_ROOT / 'images' / 'buc.png')

        # 加载基础图片
        try:
            self.chupImage = pygame.image.load(self.chup_image_path)
            self.bucImage = pygame.image.load(self.buc_image_path)
            self.backgroundImage = pygame.image.load(self.background_image_path)
            self.partback = pygame.image.load(self.partbackground_image_path)
            self.cardbackImage = pygame.image.load(self.cards_image_path + 'back.bmp')
            log.info("基础图片加载成功")
        except pygame.error as e:
            log.error(f"图片加载失败: {e}")
            raise

        self.height = 900
        self.width = 1200

        # 预缩放背景
        self.bg_scaled = pygame.transform.scale(self.backgroundImage, (self.width, self.height))
        self.partback_scaled = pygame.transform.scale(self.partback, (self.width, self.height // 3))

        # 等待画面
        waiting_image_path = str(_SHOWDOWN_ROOT / 'images' / 'waiting.jpg')
        self.screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        waiting_img = pygame.image.load(waiting_image_path).convert()
        self.screen.blit(pygame.transform.scale(waiting_img, (self.width, self.height)), (0, 0))
        pygame.display.update()
        log.info("窗口创建完成")

        # 游戏状态
        self.myFixedSeat = 3
        self.msgMyPos = None
        self.my_handCards = []
        self.playerRests = [27, 27, 27, 27]
        self.playerPlayAreas = [None, None, None, None]

        self.curRank = '2'
        self.selfRank = '2'
        self.oppRank = '2'
        self.stage = ''
        self.legalActions = []
        self.legalActions_set = []

        self.tributeflag = 0
        self.backflag = 0
        self.antiflag = 0
        self.tributeinfo = None
        self.backinfo = None
        self.antiPosList = []

        # 座位坐标
        self.relative_positions = [
            (self.width // 2 - 100, self.height - 500),
            (self.width - 400, self.height // 2 - 100),
            (self.width // 2 - 100, 200),
            (150, self.height // 2 - 100)
        ]
        self.text_offsets = [(0, -30), (0, -30), (0, -30), (0, -30)]

        # 图片缓存
        self.image_cache = {}

        # UI 交互状态
        self.selected_indices = set()
        self.button_rects = {}
        self.hover_button = None
        self.message_text = None
        self.message_until = 0
        self.message_color = (255, 0, 0)

        self.highlight_cards = None
        self.base_width = 0


    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        try:
            # 改成 ↓
            msg = json.loads(message.data)  # 正确写法
            if msg['type'] == 'notify':
                self.handle_notify(msg)
            elif msg['type'] == 'act':
                self.handle_act(msg)
        except Exception as e:
            print(f"处理消息出错: {e}")

    def handle_notify(self, msg):
        stage = msg.get('stage')
        if stage == 'beginning':
            self.myFixedSeat = msg['myPos']
            self.initInfo(msg)
            print(f"玩家实际座位: {self.myFixedSeat}")
        elif stage == 'play':
            self.handle_notify_play(msg)
        elif stage == 'tribute':
            self.recordTribute(msg)
        elif stage == 'back':
            self.recordBack(msg)
        elif stage == 'anti-tribute':
            self.recordAntiTribute(msg)
        elif stage == 'episodeOver':
            self.showOver(msg)
        elif stage == 'gameOver':
            self.showGameOver(msg)
        elif stage == 'gameResult':
            self.showGameResult(msg)

    def handle_act(self, msg):
        if not self.running:
            return
        self.sync_from_act(msg)
        action_index = self.waitInput()
        if self.running:
            self.send(json.dumps({"actIndex": action_index}))

    def draw_table(self):
        """全屏重绘（每帧调用）"""
        self.screen.blit(self.bg_scaled, (0, 0))
        self.screen.blit(self.partback_scaled, (0, self.height * 2 // 3))

        self.screen.blit(self.font.render(f"当前级数 {self.curRank}", True, (0, 0, 0)), (0, 0))
        self.screen.blit(self.font.render(f"我方级数 {self.selfRank}", True, (0, 0, 0)), (0, 20))
        self.screen.blit(self.font.render(f"敌方级数 {self.oppRank}", True, (0, 0, 0)), (0, 40))

        for seat in range(4):
            pos = self.get_screen_pos(seat)
            text_pos = self.get_text_pos(seat)

            rest_surf = self.font.render(f"{self.playerRests[seat]}张", True, (0, 0, 0))
            self.screen.blit(rest_surf, text_pos)

            if seat != self.myFixedSeat:
                self.screen.blit(self.cardbackImage, pos)

            cards = self.playerPlayAreas[seat]
            if cards and isinstance(cards, list):
                for j, card in enumerate(cards):
                    if card is None or card == 'PASS':
                        continue
                    self.draw_card_with_shadow(card, pos[0] + j * 30, pos[1] + 50)

        self.base_width = (self.width - len(self.my_handCards) * 30) // 2 - 50
        for i, card in enumerate(self.my_handCards):
            y_offset = -15 if i in self.selected_indices else 0
            x = self.base_width + i * 30
            y = self.height - 200
            self.draw_card_with_shadow(card, x, y, y_offset)

            if self.stage in ('tribute', 'back') and self.highlight_cards and card in self.highlight_cards:
                img = self.get_card_image(card)
                if img:
                    rect = pygame.Rect(x, y + y_offset, img.get_width(), img.get_height())
                    pygame.draw.rect(self.screen, (255, 215, 0), rect, 4)
                    glow_rect = rect.inflate(8, 8)
                    pygame.draw.rect(self.screen, (255, 215, 0, 100), glow_rect, 2)

        self._draw_buttons()

        if self.message_text and pygame.time.get_ticks() < self.message_until:
            msg_surf = self.font.render(self.message_text, True, self.message_color)
            msg_rect = msg_surf.get_rect(center=(self.width // 2, self.height - 300))
            bg_rect = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
            self.screen.blit(msg_surf, msg_rect)
        else:
            self.message_text = None

        if self.tributeflag and self.tributeinfo:
            from_seat, to_seat, card = self.tributeinfo
            pos = self.get_screen_pos(to_seat)
            self.draw_card_with_shadow(card, pos[0], pos[1])
            self.tributeflag = 0
            pygame.display.flip()
            #safe_wait(2)
        if self.backflag and self.backinfo:
            from_seat, to_seat, card = self.backinfo
            pos = self.get_screen_pos(to_seat)
            self.draw_card_with_shadow(card, pos[0], pos[1])
            self.backflag = 0
            pygame.display.flip()
           # safe_wait(2)
        if self.antiflag:
            y = self.height // 2 - 50
            for pos_val in self.antiPosList:
                text = self.font.render(f"{pos_val} 号玩家抗贡", True, (0, 0, 0))
                self.screen.blit(text, (self.width // 2 - 50, y))
                y += 30
            self.antiflag = 0
            pygame.display.flip()
            #safe_wait(3)

        pygame.display.flip()

    def _draw_buttons(self):
        """根据 legalActions 绘制出牌/PASS/提示按钮，并存储区域用于交互"""
        if self.stage in ('tribute', 'back'):
            return

        self.button_rects.clear()
        if not self.legalActions:
            return

        has_pass = any(act[0] == 'PASS' for act in self.legalActions)
        has_play = any(act[0] != 'PASS' for act in self.legalActions)

        btn_width, btn_height = 100, 44
        btn_radius = 12
        btn_y = self.height - 260

        if has_pass and not has_play:
            rect = pygame.Rect(self.width // 2 - btn_width // 2, btn_y, btn_width, btn_height)
            color = (70, 130, 70) if self.hover_button == 'pass' else (50, 100, 50)
            pygame.draw.rect(self.screen, color, rect, border_radius=btn_radius)
            text = self.font_button.render("PASS", True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=rect.center))
            self.button_rects['pass'] = rect

        elif has_play and not has_pass:
            rect = pygame.Rect(self.width // 2 - btn_width // 2, btn_y, btn_width, btn_height)
            color = (70, 100, 200) if self.hover_button == 'play' else (50, 70, 160)
            pygame.draw.rect(self.screen, color, rect, border_radius=btn_radius)
            text = self.font_button.render("出牌", True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=rect.center))
            self.button_rects['play'] = rect

        else:
            pass_x = self.width * 3 // 8 - btn_width // 2
            hint_x = self.width // 2 - btn_width // 2
            play_x = self.width * 5 // 8 - btn_width // 2

            hint_rect = pygame.Rect(hint_x, btn_y, btn_width, btn_height)
            hint_color = (200, 120, 50) if self.hover_button == 'hint' else (180, 100, 30)
            pygame.draw.rect(self.screen, hint_color, hint_rect, border_radius=btn_radius)
            hint_text = self.font_button.render("提示", True, (255, 255, 255))
            self.screen.blit(hint_text, hint_text.get_rect(center=hint_rect.center))
            self.button_rects['hint'] = hint_rect

            if has_pass:
                pass_rect = pygame.Rect(pass_x, btn_y, btn_width, btn_height)
                color_pass = (70, 130, 70) if self.hover_button == 'pass' else (50, 100, 50)
                pygame.draw.rect(self.screen, color_pass, pass_rect, border_radius=btn_radius)
                text_pass = self.font_button.render("PASS", True, (255, 255, 255))
                self.screen.blit(text_pass, text_pass.get_rect(center=pass_rect.center))
                self.button_rects['pass'] = pass_rect

            if has_play:
                play_rect = pygame.Rect(play_x, btn_y, btn_width, btn_height)
                color_play = (70, 100, 200) if self.hover_button == 'play' else (50, 70, 160)
                pygame.draw.rect(self.screen, color_play, play_rect, border_radius=btn_radius)
                text_play = self.font_button.render("出牌", True, (255, 255, 255))
                self.screen.blit(text_play, text_play.get_rect(center=play_rect.center))
                self.button_rects['play'] = play_rect

    # ---------- 辅助方法 ----------
    def get_card_image(self, card_name):
        """带缓存的牌面图片加载"""
        if card_name not in self.image_cache:
            try:
                img = pygame.image.load(self.cards_image_path + card_name + '.jpg').convert_alpha()
                shadow = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                shadow.fill((0, 0, 0, 40))
                self.image_cache[card_name] = img
                self.image_cache[card_name + "_shadow"] = shadow
            except pygame.error as e:
                log.error(f"加载牌面失败: {card_name}.jpg - {e}")
                return None
        return self.image_cache[card_name]

    def draw_card_with_shadow(self, card_name, x, y, y_offset=0):
        """绘制一张牌，带阴影，可指定Y轴偏移"""
        img = self.get_card_image(card_name)
        if not img:
            return
        shadow = self.image_cache.get(card_name + "_shadow")
        if shadow:
            self.screen.blit(shadow, (x + 3, y + 3 + y_offset))
        self.screen.blit(img, (x, y + y_offset))

    def get_screen_pos(self, seat):
        offset = (seat - self.myFixedSeat) % 4
        return self.relative_positions[offset]

    def get_text_pos(self, seat):
        pos = self.get_screen_pos(seat)
        off = self.text_offsets[seat % 4]
        return (pos[0] + off[0], pos[1] + off[1])

    def show_message(self, text, color=(255, 0, 0), duration=1.5):
        """显示临时消息"""
        self.message_text = text
        self.message_color = color
        self.message_until = pygame.time.get_ticks() + int(duration * 1000)

    def initInfo(self, message):
        log.info(f"收到 beginning, 座位: {message['myPos']}")
        self.msgMyPos = message['myPos']
        if self.msgMyPos != self.myFixedSeat:
            log.warning(f"服务端座位 {self.msgMyPos} 与固定座位 {self.myFixedSeat} 不一致")
        self.my_handCards = message['handCards']
        self.playerRests = [27, 27, 27, 27]
        self.playerPlayAreas = [None, None, None, None]
        self.selected_indices.clear()
        self.highlight_cards = None
        self.draw_table()

    def handle_notify_play(self, message):
        curPos = message['curPos']
        curAction = message['curAction']

        if isinstance(curAction, str):
            try:
                curAction = ast.literal_eval(curAction)
            except:
                log.error(f"无法解析 curAction: {curAction}")
                return

        if curAction == 'PASS' or (isinstance(curAction, list) and curAction and curAction[0] == 'PASS'):
            self.playerPlayAreas[curPos] = None
            self.draw_table()
            return

        if isinstance(curAction, list) and len(curAction) >= 3:
            cards = curAction[2]
        else:
            log.error(f"未知 curAction 格式: {curAction}")
            return

        if isinstance(cards, str):
            cards = [cards]

        if self.playerRests[curPos] >= len(cards):
            self.playerRests[curPos] -= len(cards)
        self.playerPlayAreas[curPos] = cards
        self.draw_table()

    def _handle_hint(self):
        """处理提示按钮：循环选中下一个合法出牌组合"""
        play_actions = [act for act in self.legalActions if act[0] != 'PASS']
        if not play_actions:
            self.show_message("没有可出的牌，请 PASS", (200, 200, 0))
            return

        self.selected_indices.clear()
        idx = self.hint_index % len(play_actions)
        action = play_actions[idx]
        cards = action[2]
        if isinstance(cards, str):
            cards = [cards]

        hand_cards = self.my_handCards[:]
        for card in cards:
            try:
                pos = hand_cards.index(card)
                self.selected_indices.add(pos)
                hand_cards[pos] = None
            except ValueError:
                pass

        self.draw_table()
        self.hint_index = (self.hint_index + 1) % len(play_actions)


    def sync_from_act(self, message):
        self.stage = message['stage']
        self.my_handCards = message['handCards']
        self.curRank = message['curRank']
        self.selfRank = message['selfRank']
        self.oppRank = message['oppoRank']
        self.legalActions = message['actionList']

        self.legalActions_set = []
        for act in self.legalActions:
            if act[0] == 'PASS':
                self.legalActions_set.append(None)
            else:
                cnt = defaultdict(int)
                for c in act[2]:
                    cnt[c] += 1
                self.legalActions_set.append(cnt)

        self.selected_indices.clear()
        self.highlight_cards = None
        self.draw_table()

    def recordTribute(self, message):
        self.tributeflag = 1
        self.tributeinfo = message['result'][0]
        self.draw_table()

    def recordBack(self, message):
        self.backflag = 1
        self.backinfo = message['result'][0]
        self.draw_table()

    def recordAntiTribute(self, message):
        self.antiflag = 1
        self.antiPosList = message['antiPos']
        self.draw_table()

    def showOver(self, message):
        self.draw_table()
        order = message['order']
        curRank = message['curRank']
        self.screen.blit(self.font.render(f"完牌顺序 {order}", True, (0, 0, 0)),
                         (self.width // 2 - 50, self.height // 2 - 50))

        my_team = [self.myFixedSeat, (self.myFixedSeat + 2) % 4]
        opp_team = [(self.myFixedSeat + 1) % 4, (self.myFixedSeat + 3) % 4]
        my_best = min(order.index(p) for p in my_team)
        opp_best = min(order.index(p) for p in opp_team)

        if my_best < opp_best:
            opp_worst = max(order.index(p) for p in opp_team)
            level_up = 3 if opp_worst == 3 else (2 if opp_worst == 2 else 1)
            self.screen.blit(self.font.render(f"我方 将要升 {level_up} 级", True, (0, 0, 0)),
                             (self.width // 2 - 50, self.height // 2))
        else:
            my_worst = max(order.index(p) for p in my_team)
            level_up = 3 if my_worst == 3 else (2 if my_worst == 2 else 1)
            self.screen.blit(self.font.render(f"敌方 将要升 {level_up} 级", True, (0, 0, 0)),
                             (self.width // 2 - 50, self.height // 2))

        self.screen.blit(self.font.render(f"当前级数 {curRank}", True, (0, 0, 0)), (0, 0))
        self.screen.blit(self.font.render(f"我方级数 {self.selfRank}", True, (0, 0, 0)), (0, 20))
        self.screen.blit(self.font.render(f"敌方级数 {self.oppRank}", True, (0, 0, 0)), (0, 40))
        pygame.display.flip()
        #safe_wait(5)

    def showGameOver(self, message):
        self.draw_table()
        text = f"游戏结束：第 {message['curTimes']} 次 / 共 {message['settingTimes']} 次"
        self.screen.blit(self.font.render(text, True, (0, 0, 0)),
                         (self.width // 2 - 150, self.height // 2))
        pygame.display.flip()
        #safe_wait(5)

    def showGameResult(self, message):
        self.draw_table()
        victory = message['victoryNum']
        draws = message['draws']
        text = f"胜场: {victory}  平局: {draws}"
        self.screen.blit(self.font.render(text, True, (0, 0, 0)),
                         (self.width // 2 - 150, self.height // 2))
        pygame.display.flip()
        #safe_wait(5)

    def waitInput(self):
        """覆盖父类 waitInput，使用 self.running 控制退出，并加入提示功能"""
        # 贡/还贡阶段特殊处理
        if self.stage in ('tribute', 'back'):
            allowed_cards = set()
            card_to_action = {}
            for act_idx, act in enumerate(self.legalActions):
                if act[0] != 'PASS':
                    card_info = act[2]
                    if isinstance(card_info, list) and len(card_info) > 0:
                        card = card_info[0]
                    elif isinstance(card_info, str):
                        card = card_info
                    else:
                        continue
                    allowed_cards.add(card)
                    card_to_action[card] = act_idx
            self.highlight_cards = allowed_cards

            selected_action = None
            while selected_action is None and self.running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                        return 0
                    if event.type == MOUSEBUTTONUP:
                        x, y = event.pos
                        if self.height - 230 <= y <= self.height - 50:
                            idx = int((x - self.base_width) // 30)
                            if 0 <= idx < len(self.my_handCards):
                                card = self.my_handCards[idx]
                                if card in allowed_cards:
                                    selected_action = card_to_action[card]
                                else:
                                    self.show_message("只能选择高亮的牌进行进贡/还贡", (255, 100, 100))
                                    self.draw_table()
                self.draw_table()
                pygame.time.wait(10)
            self.highlight_cards = None
            return selected_action if selected_action is not None else 0

        # 正常出牌阶段
        self.hint_index = 0           # 重置提示索引
        self.selected_indices.clear()
        self.hover_button = None
        action_index = None

        while action_index is None and self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    return 0
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    self.running = False
                    return 0

                if event.type == MOUSEMOTION:
                    old_hover = self.hover_button
                    self.hover_button = None
                    for name, rect in self.button_rects.items():
                        if rect.collidepoint(event.pos):
                            self.hover_button = name
                            break
                    if old_hover != self.hover_button:
                        self.draw_table()

                if event.type == MOUSEBUTTONUP:
                    x, y = event.pos
                    if self.height - 230 <= y <= self.height - 50:
                        idx = int((x - self.base_width) // 30)
                        if 0 <= idx < len(self.my_handCards):
                            if idx in self.selected_indices:
                                self.selected_indices.remove(idx)
                            else:
                                self.selected_indices.add(idx)
                            self.draw_table()
                            continue

                    for btn_name, rect in self.button_rects.items():
                        if rect.collidepoint(x, y):
                            if btn_name == 'hint':
                                self._handle_hint()
                                break
                            elif btn_name == 'pass':
                                for idx, act in enumerate(self.legalActions):
                                    if act[0] == 'PASS':
                                        action_index = idx
                                        break
                                if action_index is None:
                                    self.show_message("当前不能 PASS", (255, 100, 100))
                                    self.draw_table()
                            elif btn_name == 'play':
                                if not self.selected_indices:
                                    self.show_message("请先点击牌选中", (255, 200, 0))
                                    self.draw_table()
                                    continue
                                selected_cards = [self.my_handCards[i] for i in sorted(self.selected_indices)]
                                cnt = defaultdict(int)
                                for c in selected_cards:
                                    cnt[c] += 1
                                matched = False
                                for idx, legal_cnt in enumerate(self.legalActions_set):
                                    if legal_cnt is not None and cnt == legal_cnt:
                                        action_index = idx
                                        matched = True
                                        break
                                if not matched:
                                    self.show_message("出牌组合不符合规则", (255, 100, 100))
                                    self.draw_table()
                                    continue
                            if action_index is not None:
                                break
                    if action_index is not None:
                        break

            self.draw_table()
            pygame.time.wait(10)

        self.selected_indices.clear()
        return action_index if action_index is not None else 0

    def connect(self):
        super().connect()  # 完成原始连接
        import threading
        threading.Thread(target=super().run_forever, daemon=True).start()

    def run_forever(self):
        import pygame
        clock = pygame.time.Clock()
        while self.running:
            pygame.event.pump()  # 保持窗口响应
            clock.tick(30)
        pygame.quit()