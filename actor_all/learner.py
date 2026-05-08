# -*- coding: utf-8 -*-
"""
Learner GUI 分发客户端 — PUB-SUB 模式
绑定 4 个 PUB 端口 (10000-10003) 广播最新权重
绑定 1 个 PULL 端口 (5555) 接收经验
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ActionValueNet
from util import encode_card, now_str, check_path


class LearnerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Learner 广播客户端")
        master.geometry("900x700")

        # UI 变量
        self.model_path = tk.StringVar(value="")
        self.use_none = tk.BooleanVar(value=False)
        self.device = tk.StringVar(value="cuda")
        self.lr = tk.DoubleVar(value=1e-4)
        self.replay_capacity = tk.IntVar(value=200000)
        self.batch_size = tk.IntVar(value=512)
        self.save_interval = tk.IntVar(value=1000)
        self.log_interval = tk.IntVar(value=100)

        # 后端变量
        self.context = None
        self.pull_socket = None
        self.pub_sockets = []      # 四个 PUB
        self.running = False

        self.model = None
        self.optimizer = None
        self.replay_buffer = deque(maxlen=self.replay_capacity.get())
        self.step_count = 0
        self.save_dir = ""

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

        ttk.Label(param_frame, text="保存间隔(步):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.save_interval, width=10).grid(row=row, column=1, padx=5, sticky=tk.W)

        ttk.Label(param_frame, text="日志间隔(步):").grid(row=row, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.log_interval, width=10).grid(row=row, column=3, padx=5, sticky=tk.W)

        # 控制按钮
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="启动 Learner", command=self.start_learner)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_learner, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 状态
        status_frame = ttk.LabelFrame(self.master, text="端口状态", padding=5)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.status_vars = []
        ports_desc = ["PULL 5555"] + [f"PUB {p}" for p in [10000,10001,10002,10003]]
        for desc in ports_desc:
            var = tk.StringVar(value=f"{desc}: 未启动")
            ttk.Label(status_frame, textvariable=var, foreground="red").pack(side=tk.LEFT, padx=8)
            self.status_vars.append(var)

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
            replay_capacity = self.replay_capacity.get()
            save_interval = self.save_interval.get()
            log_interval = self.log_interval.get()
            device = self.device.get()
        except tk.TclError:
            messagebox.showerror("错误", "参数格式不正确")
            return

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
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.step_count = 0
        self.save_dir = f"model/checkpoints_{now_str()}_learner"
        os.makedirs(self.save_dir, exist_ok=True)

        # ZMQ 初始化
        self.context = zmq.Context()
        # PULL
        try:
            self.pull_socket = self.context.socket(zmq.PULL)
            self.pull_socket.bind("tcp://*:5555")
            self.status_vars[0].set("PULL 5555: 已绑定")
            self.log("PULL 5555 已绑定")
        except Exception as e:
            self.log(f"❌ PULL 绑定失败: {e}")
            self.status_vars[0].set("PULL 5555: 失败")

        # 四个 PUB 端口
        self.pub_sockets = []
        ports = [10000, 10001, 10002, 10003]
        for i, port in enumerate(ports):
            try:
                pub = self.context.socket(zmq.PUB)
                pub.bind(f"tcp://*:{port}")
                self.pub_sockets.append(pub)
                self.status_vars[i+1].set(f"PUB {port}: 已绑定")
                self.log(f"PUB {port} 已绑定")
            except Exception as e:
                self.log(f"❌ PUB {port} 绑定失败: {e}")
                self.status_vars[i+1].set(f"PUB {port}: 失败")

        # 启动后立刻广播一次初始权重
        self.broadcast_weights()

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self._main_loop, daemon=True).start()
        self.log("🚀 Learner 已启动")

    def stop_learner(self):
        self.running = False
        if self.pull_socket:
            self.pull_socket.close()
        for pub in self.pub_sockets:
            pub.close()
        if self.context:
            self.context.term()
        for var in self.status_vars:
            var.set(var.get().split(":")[0] + ": 已停止")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("🛑 Learner 已停止")

    def train_step(self, batch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        device = self.device.get()

        for obs, history, act, reward, _, _, _, _ in batch:
            obs = obs.to(device)
            history = history.float().to(device)
            act_emb = encode_card(act).flatten().to(device)
            target = torch.tensor([reward], dtype=torch.float32, device=device)

            state_curr = torch.cat((obs.flatten(), act_emb)).unsqueeze(0)
            q_curr = self.model(state_curr, history).sum()
            loss = torch.nn.functional.mse_loss(q_curr, target)
            total_loss += loss

        avg_loss = total_loss / len(batch)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        return avg_loss.item()

    def broadcast_weights(self):
        """将当前模型权重通过所有 PUB socket 广播"""
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        msg = pickle.dumps(state_dict)
        for pub in self.pub_sockets:
            try:
                pub.send(msg)
            except Exception as e:
                self.log(f"广播失败: {e}")

    def _main_loop(self):
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        while self.running:
            socks = dict(poller.poll(timeout=500))  # 500ms
            if self.pull_socket in socks:
                try:
                    raw = self.pull_socket.recv(flags=zmq.NOBLOCK)
                    experiences = pickle.loads(raw)
                    for trans in experiences:
                        t = list(trans)
                        t[0] = torch.from_numpy(t[0])      # obs
                        t[1] = torch.from_numpy(t[1])      # history
                        t[4] = torch.from_numpy(t[4])      # next_obs
                        t[6] = torch.from_numpy(t[6])      # next_history
                        self.replay_buffer.append(tuple(t))

                    self.step_count += len(experiences)
                    self.log(f"📥 收到 {len(experiences)} 条经验 (池大小: {len(self.replay_buffer)})")

                    if len(self.replay_buffer) >= self.batch_size.get():
                        batch = random.sample(self.replay_buffer, self.batch_size.get())
                        loss = self.train_step(batch)
                        if self.step_count % self.log_interval.get() == 0:
                            self.log(f"🔄 Step {self.step_count} | Loss: {loss:.4f} | Buffer: {len(self.replay_buffer)}")

                        # 训练后立即广播新权重
                        self.broadcast_weights()

                        if self.step_count % self.save_interval.get() == 0 and self.step_count > 0:
                            save_path = os.path.join(self.save_dir, f"learner_step{self.step_count}.pth")
                            torch.save({
                                "model_state_dict": self.model.state_dict(),
                                "model_class": ActionValueNet
                            }, save_path)
                            self.log(f"💾 模型已保存: {save_path}")

                except Exception as e:
                    self.log(f"❌ 经验处理异常: {e}")

            # 即使没有经验，也可以定时广播（避免 Actor 错过更新）
            # 此处简单实现：每 10 秒心跳广播一次
            time.sleep(0.1)   # 避免忙等，外层 while + poll 已有 timeout


# 入口
if __name__ == "__main__":
    root = tk.Tk()
    app = LearnerGUI(root)
    def on_close():
        app.stop_learner()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()