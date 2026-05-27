分布式DAgger系统将传统单进程DAgger的训练与数据收集解耦为两个独立角色——**Imitation Learner**（中心训练节点）与**Imitation Dist Client**（分布式数据收集节点）。Learner通过ZMQ PUB-SUB模式广播模型权重，多个客户端并行连接游戏服务器收集专家样本，每局结束后将样本推送至Learner进行交叉熵训练。该架构使单一Learner可同时服务多桌对局，从根本上解决了单客户端DAgger样本收集效率低下的问题。

## 架构全景：三通道ZMQ与双线程协同

分布式DAgger的核心由三个ZMQ通道和两个并发线程构成。以下是完整的拓扑关系：

```mermaid
graph TB
    subgraph Learner["Imitation Learner（中心训练节点）"]
        GUI["Tkinter GUI<br/>参数配置与控制"]
        PUB["PUB Socket<br/>tcp://*:10003"]
        PULL_S["PULL Socket<br/>tcp://*:5557（样本）"]
        PULL_R["PULL Socket<br/>tcp://*:5558（就绪）"]
        TRAIN["训练线程<br/>_train_loop()"]
        RECV["接收线程<br/>_receive_loop()"]
        READY["就绪监听线程<br/>_ready_receiver()"]
        DATASET["数据集 (deque)<br/>容量: 30000"]
    end

    subgraph Client1["Imitation Client 桌1座位1"]
        SUB1["SUB Socket"]
        PUSH_S1["PUSH Socket → 5557"]
        PUSH_R1["PUSH Socket → 5558"]
        WSC1["WebSocket 游戏连接"]
        EXPERT1["TOP 专家策略"]
        MODEL1["本地模型副本"]
        BUF1["样本缓冲 buffer[]"]
    end

    subgraph Client2["Imitation Client 桌2座位1"]
        SUB2["SUB Socket"]
        PUSH_S2["PUSH Socket → 5557"]
        PUSH_R2["PUSH Socket → 5558"]
        WSC2["WebSocket 游戏连接"]
        EXPERT2["TOP 专家策略"]
        MODEL2["本地模型副本"]
        BUF2["样本缓冲 buffer[]"]
    end

    subgraph ClientN["Imitation Client 桌N 座位1"]
        SUBN["SUB Socket"]
        PUSH_SN["PUSH Socket → 5557"]
        PUSH_RN["PUSH Socket → 5558"]
        WSCN["WebSocket 游戏连接"]
        EXPERTN["TOP 专家策略"]
        MODELN["本地模型副本"]
        BUFN["样本缓冲 buffer[]"]
    end

    PUB -->|"pickle(weights + expert_prob)"| SUB1
    PUB -->|"pickle(weights + expert_prob)"| SUB2
    PUB -->|"pickle(weights + expert_prob)"| SUBN

    PUSH_S1 -->|"pickle(samples[])"| PULL_S
    PUSH_S2 -->|"pickle(samples[])"| PULL_S
    PUSH_SN -->|"pickle(samples[])"| PULL_S

    PUSH_R1 -->|b"ready"| PULL_R
    PUSH_R2 -->|b"ready"| PULL_R
    PUSH_RN -->|b"ready"| PULL_R

    WSC1 --- EXPERT1 --- BUF1
    WSC2 --- EXPERT2 --- BUF2
    WSCN --- EXPERTN --- BUFN

    PULL_S --> RECV --> DATASET
    DATASET --> TRAIN --> PUB
    PULL_R --> READY --> PUB

    style Learner fill:#e1f5fe
    style Client1 fill:#fff3e0
    style Client2 fill:#fff3e0
    style ClientN fill:#fff3e0
```

**关键设计决策**：与强化学习Learner（端口5555/5556）使用不同端口（5557/5558），确保模仿学习与强化学习可同时运行而不冲突。

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L17-L25)

## Imitation Learner：三通道服务端设计

### 端口绑定与生命周期

Learner启动时依次绑定三个端口，任一端口绑定失败则立即终止启动流程。端口绑定成功后，同步启动三个守护线程：

| 通道 | 端口 | 协议 | 方向 | 职责 |
|------|------|------|------|------|
| 权重广播 | 10003（可配置） | ZMQ PUB | Learner → Clients | 广播 `(state_dict, expert_prob)` |
| 样本接收 | 5557（固定） | ZMQ PULL | Clients → Learner | 接收每局专家样本列表 |
| 就绪信号 | 5558（固定） | ZMQ PULL | Clients → Learner | 接收客户端 `b"ready"` 握手 |

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L240-L283)

### 就绪同步机制

`_ready_receiver` 线程以阻塞模式监听5558端口，每收到一条 `b"ready"` 消息将就绪计数加一。当 `ready_count % expected == 0`（即每凑满一轮4个客户端）时触发初始权重广播。该取模设计支持客户端断开重连后重新同步——客户端重连时会再次发送 `b"ready"`，Learner据此重新广播当前权重。

```python
# 就绪计数取模触发广播的设计
if count % self.expected == 0:
    self.log("✅ 全部客户端已就绪，广播初始权重")
    self.broadcast_weights()
```

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L292-L307)

### 权重广播内容

不同于强化学习Learner仅广播 `state_dict`，模仿学习Learner广播一个元组 `(state_dict, current_expert_prob)`。专家概率被编码进广播消息中，使所有客户端同步获得衰减后的DAgger混合策略参数，无需额外通信通道。

```python
def broadcast_weights(self):
    state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
    msg = pickle.dumps((state_dict, self.current_expert_prob))
    self.pub_socket.send(msg)
```

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L309-L316)

### 样本接收与数据集管理

`_receive_loop` 线程使用 `zmq.Poller` 以100ms超时轮询5557端口，非阻塞接收样本包。每个样本包是一个列表，元素为四元组 `(obs_np, history_np, action_embs_np_list, expert_idx)`——全部以numpy数组形式传输以避免PyTorch序列化问题。收到后在Learner侧通过 `torch.from_numpy` 还原为张量并入队到容量为30000的 `deque` 数据集中。

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L318-L340)

## 训练循环：交叉熵模仿与专家概率衰减

### 训练触发与批次调度

`_train_loop` 线程持续检查数据集大小，当 `len(dataset) >= batch_size` 时进入训练模式。每轮唤醒最多执行 `min(10, dataset_len // batch_size)` 次训练步骤，避免数据集不足时空转。数据集不足时以递增休眠时间（0.1s → 最多2.0s）等待新数据。

```python
if dataset_len >= self.batch_size_val:
    batches_per_wake = min(10, dataset_len // self.batch_size_val)
    for _ in range(batches_per_wake):
        batch = random.sample(self.dataset, self.batch_size_val)
        loss, acc = self._imitation_train_step(batch)
```

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L343-L365)

### 向量化训练步骤

`_imitation_train_step` 是训练的核心。对于批次中的每个样本，其处理流程为：

1. 将所有候选动作的嵌入堆叠为 `[num_actions, emb_dim]` 张量
2. 复制状态表示与历史表示以匹配动作数量维度
3. 拼接状态-动作对输入模型，得到 `[num_actions]` 的Q值向量
4. 以专家动作索引为标签，计算交叉熵损失：`loss = CrossEntropy(qs, expert_idx)`
5. 批次平均损失反向传播 + 梯度裁剪（max_norm=1.0）

```python
qs = self.model(inp, history_rep).sum(dim=1)
loss = torch.nn.functional.cross_entropy(
    qs.unsqueeze(0),
    torch.tensor([expert_idx], device=device, dtype=torch.long)
)
```

**为什么用交叉熵而非MSE**：模仿学习的目标是让模型输出与专家动作一致的概率分布，交叉熵直接优化分类准确率，比对各动作Q值做MSE回归更贴合"模仿专家选择"的语义。

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L399-L445)

### 专家概率衰减与周期广播

每完成一步训练后，专家概率按指数衰减：

```
expert_prob = max(min_expert_prob, expert_prob × expert_decay)
```

默认参数：`expert_init=1.0`, `expert_decay=0.995`, `min_expert_prob=0.1`。衰减意味着前期完全跟随专家收集高质量样本，后期逐步让模型自主决策。每5步训练触发一次权重广播（含衰减后的专家概率），确保客户端及时获得最新策略。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| expert_init | 1.0 | 初始完全跟随专家 |
| expert_decay | 0.995 | 每次训练乘以此因子（约1400步后降至0.5） |
| min_expert_prob | 0.1 | 专家概率下限，保证始终有10%的专家引导 |

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L352-L358)

### 模型保存

每 `save_interval`（默认100）步训练后，将模型保存到 `model/imitation_checkpoints_{timestamp}/imitation_train{N}.pth`。检查点包含 `model_state_dict` 和 `model_class`（ActionValueNet），与标准加载约定兼容。

Sources: [imitation_learner.py](actor_all/imitation_learner.py#L360-L364)

## Imitation Dist Client：客户端样本收集流水线

### 初始化：三路ZMQ + 专家加载

每个客户端在 `ImitationDistClient.__init__` 中完成以下初始化：

1. **模型**：随机初始化 `ActionValueNet`，设为 `eval()` 模式（不做本地训练）
2. **SUB Socket**：连接到 `learner_host:learner_port`（默认10003），订阅所有消息
3. **PUSH Socket (5557)**：连接到Learner的样本接收端口
4. **PUSH Socket (5558)**：发送 `b"ready"` 就绪信号
5. **权重监听线程**：独立线程阻塞接收PUB消息，更新本地模型权重和专家概率

```python
# 客户端不加载任何本地模型，完全由Learner广播更新
self.model = ActionValueNet().to(self.device)
self.model.eval()
# SUB 接收权重
self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
self.sub_socket.connect(f"tcp://{args.learner_host}:{args.learner_port}")
self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
```

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L40-L52) [tcli_imitation.py](clients/tcli_imitation.py#L148-L178)

### 专家策略动态加载

`_load_expert` 方法通过 `importlib.import_module` 动态加载指定教练的Action类。支持两种加载路径：硬编码的 `EggPan`（直接从 `coach.EggPan.action` 导入）和通用的 `{expert_name}` 动态导入。专家类必须实现 `parse_AI(msg, myPos, state) -> int` 方法。

```python
def _load_expert(self, expert_name):
    if expert_name == "EggPan":
        from coach.EggPan.action import Action
        return Action(render=False)
    else:
        mod = importlib.import_module(f"coach.{expert_name}.action")
        return mod.Action(render=False)
```

默认专家为 `TOP`（完整的掼蛋规则引擎），可通过 `--expert` 参数切换。

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L56-L73)

### 每步决策：混合策略与样本收集

`ImitationAction.parse` 方法实现DAgger的核心混合策略。每步执行以下操作：

1. **查询专家**：调用 `expert.parse_AI(msg, myPos, state)` 获取专家动作索引
2. **记录样本**：将当前 `(状态, 历史, 动作嵌入列表, 专家索引)` 转为numpy存入buffer
3. **动作选择**：以概率 `use_expert_prob` 跟随专家，否则用模型softmax采样
4. **更新历史**：将选择的动作加入LSTM历史序列

```python
def parse(self, msg, render=False, state=None):
    expert_idx = self.expert.parse_AI(msg, msg.get("myPos", 0), state)
    expert_idx = np.clip(expert_idx, 0, self.act_range).tolist()
    self.add_to_buffer(msg, expert_idx)  # 始终记录专家样本
    
    if random.random() < self.use_expert_prob:
        index = expert_idx
    else:
        index = self.select_action_by_model(msg)
```

**关键设计**：无论最终选择了专家动作还是模型动作，始终以专家动作为标签记录样本。这遵循DAgger的核心原则——"学习专家会怎么做"，而非"学习自己实际做了什么"。

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L113-L139)

### 模型动作采样

`select_action_by_model` 对所有候选动作计算Q值，经softmax转为概率分布后随机采样。这种方式比argmax引入了必要的探索噪声，避免模型过早收敛到次优策略。

```python
q_tensor = torch.stack(q_vals)
probs = torch.softmax(q_tensor, dim=0).detach().cpu().numpy()
return np.random.choice(len(q_vals), p=probs)
```

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L97-L107)

### 样本发送时机

每局结束时（`stage == "episodeOver"`），客户端将本局积累的全部样本通过PUSH Socket发送至Learner的5557端口：

```python
elif msg["stage"] == "episodeOver":
    if self.action.buffer:
        self.send_expert_samples()
    self.action.reset_episode()
```

一局掼蛋通常包含数十步出牌决策，因此每次推送的样本量在数十条量级。新局开始时（`stage == "beginning"`）重置buffer。

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L209-L216) [tcli_imitation.py](clients/tcli_imitation.py#L225-L230)

### 权重热加载

权重监听线程以500ms超时轮询SUB Socket。收到消息后通过 `pickle.loads` 反序列化为 `(state_dict, expert_prob)`，然后调用 `load_weights` 将权重拷贝到本地模型并同步专家概率：

```python
def load_weights(self, state_dict, expert_prob=None):
    with torch.no_grad():
        for k, v in state_dict.items():
            state_dict[k] = v.to(self.device)
        self.model.load_state_dict(state_dict)
    if expert_prob is not None:
        self.use_expert_prob = expert_prob
```

整个过程在 `torch.no_grad()` 下完成，不触发任何梯度计算。

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L141-L147) [tcli_imitation.py](clients/tcli_imitation.py#L195-L201)

## 多桌启动：start_imitation.py 编排

`start_imitation.py` 负责启动完整的分布式模仿学习环境——4张独立游戏桌，每桌1个Imitation客户端（座位1）+ 3个TOP规则客户端（座位2-4）：

| 桌子 | 端口 | 座位1 | 座位2-4 |
|------|------|-------|---------|
| Table 1 | 23456 | Imitation Dist | TOP 规则 |
| Table 2 | 23457 | Imitation Dist | TOP 规则 |
| Table 3 | 23458 | Imitation Dist | TOP 规则 |
| Table 4 | 23459 | Imitation Dist | TOP 规则 |

所有Imitation客户端连接同一个Learner（`127.0.0.1:10003`），所有规则客户端通过 `tcli.py rule` 模式运行。进程以 `CREATE_NO_WINDOW` 标记启动，不弹出控制台窗口。

```python
def start_imitation(port, seat, extra_args=[]):
    cmd = [PYTHON, IMITATION_CLIENT, "imitation_dist", str(seat),
           "--host", "127.0.0.1", "--port", str(port)] + extra_args
    return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW, ...)
```

Sources: [start_imitation.py](actor_all/start_imitation.py#L19-L38)

## 与单机DAgger的架构对比

分布式DAgger是对 `clients/imitation_client.py` 中单机DAgger的根本性重构：

| 维度 | 单机DAgger (imitation_client.py) | 分布式DAgger |
|------|----------------------------------|-------------|
| 训练位置 | 客户端本地（WebSocket线程内） | Learner独立训练线程 |
| 数据集范围 | 单个客户端自己的对局 | 所有客户端聚合 |
| 权重同步 | 无——模型仅本地更新 | ZMQ PUB-SUB实时广播 |
| 专家概率管理 | 客户端各自衰减 | Learner统一管理并广播 |
| 模型保存 | 客户端本地保存 | Learner集中保存 |
| 扩展性 | 1个客户端 = 1个训练单元 | N个客户端 + 1个Learner |
| 训练时机 | 每N局结束后全量训练 | 流式持续训练 |
| 损失函数 | softmax + -log(prob) 逐样本BP | 向量化交叉熵 + 梯度裁剪 |

单机DAgger的 `MLPAction` 类承担了模型持有、优化器持有、数据集管理、训练执行四重职责；分布式版本将这些职责拆分到 `ImitationAction`（纯推理+数据收集）和 `ImitationLearnerGUI`（纯训练+权重分发）。

Sources: [imitation_client.py](clients/imitation_client.py#L631-L728) [tcli_imitation.py](clients/tcli_imitation.py#L40-L147)

## 样本数据结构

每条专家样本包含四个组件，在客户端以numpy数组形式序列化后通过pickle传输：

| 字段 | 来源 | 形状 | 说明 |
|------|------|------|------|
| obs_np | `StateCatEmbedding(msg)` | `(493,)` | 手牌 + 四家出牌区 + 剩余牌数 + 级牌 |
| history_np | `MapHistoryToLSTM()` | `(1, T, 60)` | 出牌历史序列的LSTM输入 |
| action_embs_np_list | 所有候选动作的 `encode_card` | `[N, (60,)]` | N为可选动作数 |
| expert_idx | `expert.parse_AI()` | `int` | 专家策略选择的动作索引 |

`StateCatEmbedding` 将手牌（4×15=60维）、四家出牌区（4×60=240维）、剩余牌数（one-hot 30维 × 4 = 120维）、级牌（one-hot 13维）拼接为493维向量。

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L79-L93) [util.py](util.py#L67-L80)

## 命令行用法

### 启动Learner（GUI模式）

```bash
python actor_all/imitation_learner.py
```

GUI提供完整的参数配置界面：模型路径、学习率、数据集容量、batch大小、专家概率参数、PUB端口等。

### 启动单个客户端

```bash
python clients/tcli_imitation.py imitation_dist 1 \
    --host 127.0.0.1 --port 23456 \
    --learner_host 127.0.0.1 --learner_port 10003 \
    --expert TOP --device cuda
```

参数说明：
- `imitation_dist`：模式标识（必选子命令）
- `pos`：座位号（1-4）
- `--learner_host/port`：Learner的PUB端口地址
- `--expert`：专家教练名称，对应 `coach/{Name}/action.py`
- `--device`：模型推理设备

### 一键启动4桌

```bash
python actor_all/start_imitation.py
```

自动启动4张桌子的全部16个进程（4个Imitation + 12个TOP规则）。

Sources: [tcli_imitation.py](clients/tcli_imitation.py#L238-L267) [start_imitation.py](actor_all/start_imitation.py#L52-L101)

## 阅读路径建议

本文档描述了分布式DAgger的核心训练架构。建议按以下顺序继续阅读：

- **上游原理**：[模仿学习原理：DAgger算法与专家策略混合采样](6-mo-fang-xue-xi-yuan-li-daggersuan-fa-yu-zhuan-jia-ce-lue-hun-he-cai-yang) — 理解DAgger算法的数学基础与混合采样策略
- **状态编码**：[状态编码设计：手牌、出牌区、剩余牌数与级牌的多维嵌入](13-zhuang-tai-bian-ma-she-ji-shou-pai-chu-pai-qu-sheng-yu-pai-shu-yu-ji-pai-de-duo-wei-qian-ru) — 理解493维状态向量的构造细节
- **专家策略**：[TOP专家策略：完整的掼蛋规则引擎与牌型组合搜索](18-topzhuan-jia-ce-lue-wan-zheng-de-guan-dan-gui-ze-yin-qing-yu-pai-xing-zu-he-sou-suo) — 理解默认专家TOP的决策机制
- **分布式对比**：[分布式Learner设计：ZMQ PUB-SUB权重广播与经验收集](21-fen-bu-shi-learnershe-ji-zmq-pub-subquan-zhong-yan-bo-yu-jing-yan-shou-ji) — 与强化学习Learner的架构对比
- **客户端统一入口**：[客户端精简化：tcli.py统一入口与多模式路由](23-ke-hu-duan-jing-jian-hua-tcli-pytong-ru-kou-yu-duo-mo-shi-lu-you) — 理解tcli如何路由到imitation_dist模式