# -*- coding: utf-8 -*-
"""
Learner V2 — 自博弈版
在 V1 基础上增加：
  - 固定保存目录 model/selfplay_checkpoints/，方便自博弈对手查找
  - 模型池管理：维护所有已保存 checkpoint 列表
  - REP socket (10004) 供外部查询可用模型
  - 保存间隔默认 25 步，生成更细粒度的对手
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
import numpy as np

sys.path.append(os.path.abspath('.'))
from model import ActionValueNet
from util import encode_card, now_str, check_path


class LearnerV2GUI:
    def __init__(self, master):
        self.master = master
        master.title("Learner V2 — 自博弈版")
        master.geometry("900x750")

        # UI 变量
        self.model_path = tk.StringVar(value="")
        self.use_none = tk.BooleanVar(value=False)
        self.device = tk.StringVar(value="cuda")
        self.lr = tk.DoubleVar(value=25e-4)
        self.gamma = tk.DoubleVar(value=0.98)
        self.n_step = tk.IntVar(value=3)                  # TD(n) 步数
        self.target_update_freq = tk.IntVar(value=100)
        self.replay_capacity = tk.IntVar(value=8000)
        self.batch_size = tk.IntVar(value=512)
        self.save_interval = tk.IntVar(value=25)       # V2: 默认 25 步
        self.log_interval = tk.IntVar(value=1)
        self.pub_port = tk.IntVar(value=10002)
        self.expected_clients = tk.IntVar(value=4)
        self.rep_port = tk.IntVar(value=10004)          # V2: 模型池查询端口

        # 后端变量
        self.context = None
        self.pull_socket = None
        self.pub_socket = None
        self.ready_pull_socket = None
        self.rep_socket = None                          # V2: REP socket
        self.running = False

        self.model = None
        self.target_model = None
        self.optimizer = None
        self.replay_buffer = deque(maxlen=self.replay_capacity.get())
        self.buffer_lock = threading.Lock()
        self.step_count = 0
        self.save_dir = ""

        self.model_pool = []                            # V2: 模型池
        self.model_pool_lock = threading.Lock()         # V2

        self.ready_count = 0
        self.ready_lock = threading.Lock()

        self._next_seq = 1
        self.batch_size_val = None
        self.log_interval_val = None
        self.save_interval_val = None
        self.device_val = None
        self.gamma_val = None
        self.n_step_val = None
        self.target_update_freq_val = None

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

        ttk.Label(param_frame, text="经验池容量:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.replay_capacity, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="Batch Size:").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.batch_size, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="保存间隔(训):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.save_interval, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="日志间隔(训):").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.log_interval, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="Gamma (TD):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.gamma, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="TD步数 (n):").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.n_step, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="Target同步频率:").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.target_update_freq, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        ttk.Label(param_frame, text="PUB 端口:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.pub_port, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="期望客户端数:").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.expected_clients, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)
        row += 1

        # V2: REP 端口
        ttk.Label(param_frame, text="REP 端口 (模型池):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.rep_port, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        # 控制按钮
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="启动 Learner V2", command=self.start_learner)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_learner, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 状态显示
        status_frame = ttk.LabelFrame(self.master, text="端口状态", padding=5)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.status_vars = []
        ports_desc = ["PULL 经验 5555", "PULL 就绪 5556", "PUB 权重", "REP 模型池 10004"]
        for desc in ports_desc:
            var = tk.StringVar(value=f"{desc}: 未启动")
            ttk.Label(status_frame, textvariable=var, foreground="red").pack(side=tk.LEFT, padx=8)
            self.status_vars.append(var)

        self.ready_label = ttk.Label(status_frame, text="就绪: 0", foreground="blue")
        self.ready_label.pack(side=tk.LEFT, padx=10)
        self.pool_label = ttk.Label(status_frame, text="模型池: 0", foreground="green")
        self.pool_label.pack(side=tk.LEFT, padx=10)

        # 日志
        log_frame = ttk.LabelFrame(self.master, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_model(self):
        initdir = os.path.abspath("./model") if os.path.exists("model") else os.getcwd()
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

    def update_pool_display(self):
        def _update():
            with self.model_pool_lock:
                n = len(self.model_pool)
            self.pool_label.config(text=f"模型池: {n}")
        self.master.after(0, _update)

    # ---------- 核心逻辑 ----------
    def start_learner(self):
        if self.running:
            return

        try:
            lr = self.lr.get()
            gamma = self.gamma.get()
            n_step = self.n_step.get()
            target_update_freq = self.target_update_freq.get()
            batch_size = self.batch_size.get()
            replay_capacity = self.replay_capacity.get()
            save_interval = self.save_interval.get()
            log_interval = self.log_interval.get()
            device = self.device.get()
            pub_port = self.pub_port.get()
            expected = self.expected_clients.get()
            rep_port = self.rep_port.get()
        except tk.TclError:
            messagebox.showerror("错误", "参数格式不正确")
            return

        self.batch_size_val = batch_size
        self.log_interval_val = log_interval
        self.save_interval_val = save_interval
        self.device_val = device
        self.gamma_val = gamma
        self.n_step_val = n_step
        self.target_update_freq_val = target_update_freq

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

        self.target_model = ActionValueNet().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.step_count = 0

        # V2: 固定保存目录，扫描已有 checkpoint，续接编号 (1,2,3...)
        self.save_dir = "model/selfplay_checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
        import glob as _glob
        existing = _glob.glob(os.path.join(self.save_dir, "*.pth"))
        self.model_pool = sorted(existing, key=lambda f: int(os.path.basename(f).replace(".pth", "")) if os.path.basename(f).replace(".pth", "").isdigit() else 0)
        if self.model_pool:
            nums = []
            for f in self.model_pool:
                try:
                    nums.append(int(os.path.basename(f).replace(".pth", "")))
                except Exception:
                    pass
            self._next_seq = max(nums) + 1 if nums else 1
            self.log(f"📁 检测到 {len(self.model_pool)} 个已有 checkpoint，编号从 {self._next_seq} 继续")
        else:
            self._next_seq = 1
        self.log(f"📁 模型保存目录: {os.path.abspath(self.save_dir)}")

        # ZMQ 初始化
        self.context = zmq.Context()

        try:
            self.pull_socket = self.context.socket(zmq.PULL)
            self.pull_socket.bind("tcp://*:5555")
            self.status_vars[0].set("PULL 经验 5555: 已绑定")
            self.log("📥 PULL 经验 5555 已绑定")
        except Exception as e:
            self.log(f"❌ PULL 经验 绑定失败: {e}")
            return

        try:
            self.ready_pull_socket = self.context.socket(zmq.PULL)
            self.ready_pull_socket.bind("tcp://*:5556")
            self.status_vars[1].set("PULL 就绪 5556: 已绑定")
            self.log("🔗 PULL 就绪 5556 已绑定")
        except Exception as e:
            self.log(f"❌ PULL 就绪 绑定失败: {e}")
            return

        try:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind(f"tcp://*:{pub_port}")
            self.status_vars[2].set(f"PUB {pub_port}: 已绑定")
            self.log(f"📡 PUB {pub_port} 已绑定")
        except Exception as e:
            self.log(f"❌ PUB {pub_port} 绑定失败: {e}")
            return

        # V2: REP socket
        try:
            self.rep_socket = self.context.socket(zmq.REP)
            self.rep_socket.bind(f"tcp://*:{rep_port}")
            self.status_vars[3].set(f"REP 模型池 {rep_port}: 已绑定")
            self.log(f"🗂️ REP 模型池 {rep_port} 已绑定")
        except Exception as e:
            self.log(f"❌ REP 模型池 {rep_port} 绑定失败: {e}")
            return

        self.ready_count = 0
        self.expected = expected
        self.update_ready_display()
        self.update_pool_display()

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self._ready_receiver, daemon=True).start()
        threading.Thread(target=self._receive_loop, daemon=True).start()
        threading.Thread(target=self._train_loop, daemon=True).start()
        threading.Thread(target=self._rep_handler, daemon=True).start()    # V2
        self.log("🚀 Learner V2 已启动（自博弈模式）")

    def stop_learner(self):
        self.running = False
        if self.pull_socket:
            self.pull_socket.close()
        if self.ready_pull_socket:
            self.ready_pull_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        if self.rep_socket:
            self.rep_socket.close()
        if self.context:
            self.context.term()
        for var in self.status_vars:
            var.set(var.get().split(":")[0] + ": 已停止")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("🛑 Learner V2 已停止")

    # ---------- V2: REP 模型池服务 ----------
    def _rep_handler(self):
        while self.running:
            try:
                msg = self.rep_socket.recv(flags=zmq.NOBLOCK)
                if msg == b"pool":
                    with self.model_pool_lock:
                        reply = pickle.dumps(list(self.model_pool))
                    self.rep_socket.send(reply)
                elif msg == b"latest":
                    with self.model_pool_lock:
                        latest = self.model_pool[-1] if self.model_pool else ""
                    self.rep_socket.send(pickle.dumps(latest))
                else:
                    self.rep_socket.send(pickle.dumps([]))
            except zmq.ZMQError:
                time.sleep(0.1)
            except Exception as e:
                self.log(f"REP 异常: {e}")
                time.sleep(0.1)

    # ---------- 以下与原版 learner.py 一致 ----------
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
                    if count % self.expected == 0:
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

    def train_step(self, batch):
        """纯 MC：Q(s,a) 直接回归到 reward（Client 端已填入终局奖励）。
        所有 transition 的 done=True，不做任何 TD bootstrap。"""
        self.model.train()
        total_loss = 0.0
        total_q = 0.0
        total_reward = 0.0
        self.optimizer.zero_grad()
        device = self.device_val

        for obs, history, act, reward, _obs_next, _actionListNext, _history_next, _done in batch:
            obs = obs.float().to(device)
            history = history.float().to(device)
            act_emb = encode_card(act).flatten().to(device)

            state_curr = torch.cat((obs.flatten(), act_emb)).unsqueeze(0)
            q_curr = self.model(state_curr, history).sum()

            target = torch.tensor(reward, dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(q_curr, target)
            total_loss += loss
            total_q += q_curr.item()
            total_reward += reward

        n = len(batch)
        avg_loss = total_loss / n
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        return avg_loss.item(), total_q / n, total_reward / n, total_reward / n

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
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        while self.running:
            socks = dict(poller.poll(timeout=100))
            if self.pull_socket in socks:
                try:
                    raw = self.pull_socket.recv(flags=zmq.NOBLOCK)
                    experiences = pickle.loads(raw)
                    with self.buffer_lock:
                        for trans in experiences:
                            t = list(trans)
                            t[0] = torch.from_numpy(t[0])
                            t[1] = torch.from_numpy(t[1])
                            t[4] = torch.from_numpy(t[4])
                            t[6] = torch.from_numpy(t[6])
                            self.replay_buffer.append(tuple(t))
                        self.step_count += len(experiences)
                    self.log(f"📥 收到 {len(experiences)} 条经验 (池大小: {len(self.replay_buffer)})")
                except Exception as e:
                    self.log(f"❌ 接收异常: {e}")

    def _train_loop(self):
        self.log(f"🔍 训练线程启动，batch_size_val={self.batch_size_val}")
        train_count = 0
        idle_count = 0
        while self.running:
            with self.buffer_lock:
                buffer_len = len(self.replay_buffer)
            if buffer_len >= self.batch_size_val:
                idle_count = 0
                batches_per_wake = min(10, buffer_len // self.batch_size_val)
                try:
                    for step in range(batches_per_wake):
                        with self.buffer_lock:
                            if len(self.replay_buffer) < self.batch_size_val:
                                break
                            batch = random.sample(self.replay_buffer, self.batch_size_val)
                        loss, avg_q, avg_target, avg_r = self.train_step(batch)
                        train_count += 1
                        if train_count % self.target_update_freq_val == 0:
                            self.target_model.load_state_dict(self.model.state_dict())
                            self.log(f"🎯 Target 网络已同步 (训练#{train_count})")
                        if train_count % self.log_interval_val == 0:
                            self.log(
                                f"🔄 训练 #{train_count} | Loss: {loss:.4f} | "
                                f"Q均值: {avg_q:+.2f} | Target均值: {avg_target:+.2f} | "
                                f"Reward均值: {avg_r:+.2f} | Buffer: {buffer_len} | "
                                f"模型池: {len(self.model_pool)}"
                            )
                        if train_count % 5 == 0:
                            self.broadcast_weights()
                        if train_count % self.save_interval_val == 0:
                            save_path = os.path.join(self.save_dir, f"{self._next_seq}.pth")
                            torch.save({
                                "model_state_dict": self.model.state_dict(),
                                "model_class": ActionValueNet
                            }, save_path)
                            with self.model_pool_lock:
                                self.model_pool.append(save_path)
                            self._next_seq += 1
                            self.update_pool_display()
                            self.log(f"💾 模型已保存: {save_path} (池: {len(self.model_pool)})")
                except Exception as e:
                    self.log(f"❌ 训练异常: {e}")
            else:
                idle_count += 1
                sleep_time = min(0.1 * idle_count, 2.0)
                time.sleep(sleep_time)


if __name__ == "__main__":
    root = tk.Tk()
    app = LearnerV2GUI(root)
    def on_close():
        app.stop_learner()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
