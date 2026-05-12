# -*- coding: utf-8 -*-
"""
分布式模仿学习 Learner — 单端口 PUB + 双 PULL（样本+就绪）
绑定 1 个 PUB 端口 (默认 10003) 广播最新权重
绑定 1 个 PULL 端口 (5557) 接收专家样本（obs, history, action_embs, expert_idx）
绑定 1 个 PULL 端口 (5558) 接收客户端就绪消息
等待所有客户端就绪后广播初始权重
训练在独立线程中进行，使用交叉熵损失模仿专家动作
"""

import os
import sys
import time
import pickle
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from collections import deque
import random

import zmq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ActionValueNet
from util import encode_card, now_str, check_path


class ImitationLearnerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Imitation Learner 广播客户端")
        master.geometry("900x700")

        # UI 变量
        self.model_path = tk.StringVar(value="")
        self.use_none = tk.BooleanVar(value=False)
        self.device = tk.StringVar(value="cuda")
        self.lr = tk.DoubleVar(value=1e-4)
        self.dataset_capacity = tk.IntVar(value=200000)
        self.batch_size = tk.IntVar(value=512)
        self.save_interval = tk.IntVar(value=100)
        self.log_interval = tk.IntVar(value=10)
        self.pub_port = tk.IntVar(value=10003)
        self.expected_clients = tk.IntVar(value=4)

        # 后端变量
        self.context = None
        self.pull_socket = None      # 接收专家样本 (5557)
        self.pub_socket = None
        self.ready_pull_socket = None
        self.running = False

        self.model = None
        self.optimizer = None
        # 数据集：每个元素为 (obs_tensor, history_tensor, action_embs_list, expert_idx)
        self.dataset = deque(maxlen=self.dataset_capacity.get())
        self.dataset_lock = threading.Lock()
        self.train_count = 0
        self.save_dir = ""

        self.ready_count = 0
        self.ready_lock = threading.Lock()
        self.expected = 0

        # 线程安全变量
        self.batch_size_val = None
        self.log_interval_val = None
        self.save_interval_val = None
        self.device_val = None

        self.build_ui()

    def build_ui(self):
        # 模型选择区
        model_frame = ttk.LabelFrame(self.master, text="模型加载", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(model_frame, text="模型文件:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="浏览...", command=self.browse_model).grid(row=0, column=2, padx=5)
        ttk.Checkbutton(model_frame, text="不加载预训练模型 (None)",
                        variable=self.use_none, command=self.toggle_none).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)

        # 训练参数
        param_frame = ttk.LabelFrame(self.master, text="训练参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        row = 0
        ttk.Label(param_frame, text="设备:").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.device, values=["cuda", "cpu"], state="readonly", width=8).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="学习率:").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.lr, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="数据集容量:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.dataset_capacity, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="Batch Size:").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.batch_size, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="保存间隔(训):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.save_interval, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="日志间隔(训):").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.log_interval, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="PUB 端口:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.pub_port, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="期望客户端数:").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.expected_clients, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)

        # 控制按钮
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="启动 Learner", command=self.start_learner)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_learner, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 端口状态
        status_frame = ttk.LabelFrame(self.master, text="端口状态", padding=5)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.status_vars = []
        ports_desc = ["PULL 样本 5557", "PULL 就绪 5558", "PUB 权重"]
        for desc in ports_desc:
            var = tk.StringVar(value=f"{desc}: 未启动")
            ttk.Label(status_frame, textvariable=var, foreground="red").pack(side=tk.LEFT, padx=8)
            self.status_vars.append(var)

        self.ready_label = ttk.Label(status_frame, text="就绪: 0", foreground="blue")
        self.ready_label.pack(side=tk.LEFT, padx=20)

        # 日志
        log_frame = ttk.LabelFrame(self.master, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_model(self):
        initdir = os.path.abspath("./model") if os.path.exists("./model") else os.getcwd()
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            initialdir=initdir,
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
            self.use_none.set(False)

    def toggle_none(self):
        if self.use_none.get():
            self.model_path.set("")

    def log(self, msg):
        def _write():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.master.after(0, _write)

    # ---------- 核心逻辑 ----------
    def start_learner(self):
        if self.running:
            return

        try:
            lr = self.lr.get()
            batch_size = self.batch_size.get()
            dataset_cap = self.dataset_capacity.get()
            save_interval = self.save_interval.get()
            log_interval = self.log_interval.get()
            device = self.device.get()
            pub_port = self.pub_port.get()
            expected = self.expected_clients.get()
        except tk.TclError:
            messagebox.showerror("错误", "参数格式不正确")
            return

        self.batch_size_val = batch_size
        self.log_interval_val = log_interval
        self.save_interval_val = save_interval
        self.device_val = device

        # 加载模型
        if self.use_none.get():
            model_path = None
        else:
            model_path = self.model_path.get().strip()

        try:
            if model_path:
                check_path(model_path)
                ckpt = torch.load(model_path, map_location=device)
                self.model = ckpt.get("model_class", ActionValueNet)().to(device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.log(f"✅ 已加载模型: {model_path}")
            else:
                self.model = ActionValueNet().to(device)
                self.log("✅ 使用随机初始化模型")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")
            return

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.dataset = deque(maxlen=dataset_cap)
        self.train_count = 0
        self.save_dir = f"model/imitation_checkpoints_{now_str()}"
        os.makedirs(self.save_dir, exist_ok=True)

        # ZMQ 初始化
        self.context = zmq.Context()

        try:
            self.pull_socket = self.context.socket(zmq.PULL)
            self.pull_socket.bind("tcp://*:5557")
            self.status_vars[0].set("PULL 样本 5557: 已绑定")
            self.log("📥 PULL 样本 5557 已绑定")
        except Exception as e:
            self.log(f"❌ PULL 样本绑定失败: {e}")
            return

        try:
            self.ready_pull_socket = self.context.socket(zmq.PULL)
            self.ready_pull_socket.bind("tcp://*:5558")
            self.status_vars[1].set("PULL 就绪 5558: 已绑定")
            self.log("🔗 PULL 就绪 5558 已绑定")
        except Exception as e:
            self.log(f"❌ PULL 就绪绑定失败: {e}")
            return

        try:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind(f"tcp://*:{pub_port}")
            self.status_vars[2].set(f"PUB {pub_port}: 已绑定")
            self.log(f"📡 PUB {pub_port} 已绑定")
        except Exception as e:
            self.log(f"❌ PUB {pub_port} 绑定失败: {e}")
            return

        self.ready_count = 0
        self.expected = expected
        self.update_ready_display()

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self._ready_receiver, daemon=True).start()
        threading.Thread(target=self._receive_loop, daemon=True).start()
        threading.Thread(target=self._train_loop, daemon=True).start()
        self.log("🚀 Imitation Learner 已启动（等待客户端就绪）")

    def stop_learner(self):
        self.running = False
        if self.pull_socket:
            self.pull_socket.close()
        if self.ready_pull_socket:
            self.ready_pull_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        if self.context:
            self.context.term()
        for var in self.status_vars:
            var.set(var.get().split(":")[0] + ": 已停止")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("🛑 Imitation Learner 已停止")

    def _ready_receiver(self):
        while self.running:
            try:
                msg = self.ready_pull_socket.recv()
                if msg == b"ready":
                    with self.ready_lock:
                        self.ready_count += 1
                        count = self.ready_count
                    self.update_ready_display()
                    self.log(f"🔌 客户端就绪 ({count}/{self.expected})")
                    if count == self.expected:
                        self.log("✅ 全部客户端已就绪，广播初始权重")
                        self.broadcast_weights()
            except zmq.ZMQError:
                break
            except Exception as e:
                self.log(f"就绪接收异常: {e}")

    def update_ready_display(self):
        def _update():
            self.ready_label.config(text=f"就绪: {self.ready_count}/{self.expected}")
        self.master.after(0, _update)

    def broadcast_weights(self):
        if not self.pub_socket:
            return
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        msg = pickle.dumps(state_dict)
        try:
            self.pub_socket.send(msg)
        except Exception as e:
            self.log(f"广播失败: {e}")

    def _receive_loop(self):
        """接收 (obs_np, history_np, action_embs_np_list, expert_idx)"""
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        while self.running:
            socks = dict(poller.poll(timeout=100))
            if self.pull_socket in socks:
                try:
                    raw = self.pull_socket.recv(flags=zmq.NOBLOCK)
                    samples = pickle.loads(raw)
                    with self.dataset_lock:
                        for (obs_np, history_np, action_embs_np_list, expert_idx) in samples:
                            obs = torch.from_numpy(obs_np)
                            history = torch.from_numpy(history_np)
                            action_embs = [torch.from_numpy(emb) for emb in action_embs_np_list]
                            self.dataset.append((obs, history, action_embs, expert_idx))
                    self.log(f"📥 收到 {len(samples)} 条样本 (数据集大小: {len(self.dataset)})")
                except Exception as e:
                    self.log(f"❌ 接收异常: {e}")

    def _train_loop(self):
        print("Train loop started, batch_size:", self.batch_size_val)  # 加一行
        while self.running:
            time.sleep(1)
            with self.dataset_lock:
                dataset_len = len(self.dataset)
            print(f"Dataset length: {dataset_len}, required: {self.batch_size_val}")  # 加一行
            if dataset_len >= self.batch_size_val:
                try:
                    for _ in range(2):
                        with self.dataset_lock:
                            if len(self.dataset) < self.batch_size_val:
                                break
                            batch = random.sample(self.dataset, self.batch_size_val)
                        loss = self._imitation_train_step(batch)
                        self.train_count += 1
                        print(f"Train step {self.train_count}, Loss: {loss:.4f}")  # 加一行
                        if self.train_count % self.log_interval_val == 0:
                            save_path = os.path.join(self.save_dir, f"learner_train{self.train_count}.pth")
                            torch.save({
                                "model_state_dict": self.model.state_dict(),
                                "model_class": ActionValueNet
                            }, save_path)
                            self.log(f"💾 模型已保存: {save_path}")
                        if self.train_count % 10 == 0:
                            self.broadcast_weights()

                except Exception as e:
                    print(f"Train exception: {e}")  # 加一行
                    self.log(f"❌ 训练异常: {e}")

    def _imitation_train_step(self, batch):
        """向量化加速版"""
        if not batch:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()
        device = self.device_val
        total_loss = 0.0

        for obs, history, action_embs, expert_idx in batch:
            # 1. 堆叠动作嵌入 (num_actions, emb_dim)
            embs = torch.stack([emb.to(device).flatten() for emb in action_embs])
            num_actions = embs.shape[0]

            # 2. 复制 obs (num_actions, obs_dim)
            obs_rep = obs.to(device).flatten().unsqueeze(0).repeat(num_actions, 1)

            # 3. 处理 history —— 强制转为 3D (1, seq_len, feat_dim) 再复制
            h = history.float().to(device)
            while h.dim() > 3 and h.shape[0] == 1:
                h = h.squeeze(0)
            while h.dim() > 3:
                h = h[0]
            if h.dim() == 2:
                h = h.unsqueeze(0)  # (1, seq_len, feat_dim)
            history_rep = h.repeat(num_actions, 1, 1)

            # 4. 拼接一次前向
            inp = torch.cat((obs_rep, embs), dim=1)
            qs = self.model(inp, history_rep).sum(dim=1)  # (num_actions,)

            # 5. 损失
            loss = torch.nn.functional.cross_entropy(
                qs.unsqueeze(0),
                torch.tensor([expert_idx], device=device, dtype=torch.long)
            )
            total_loss += loss

        avg_loss = total_loss / len(batch)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        return avg_loss.item()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImitationLearnerGUI(root)

    def on_close():
        app.stop_learner()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()