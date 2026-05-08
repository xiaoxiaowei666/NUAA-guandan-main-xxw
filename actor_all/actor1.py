# -*- coding: utf-8 -*-
"""
精简版 GUI 启动器 — 仅 rule / reinforcement，调用 tcli.py
修改 PORT 即可生成不同服务器的启动器
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import sys
import os
import threading

# ========== 配置 —— 修改这里的端口即可 ==========
GAME_PORT = 23456          # 游戏服务器端口
DEFAULT_HOST = "127.0.0.1"
# =================================================

class SimpleLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title(f"客户端启动器 (端口 {GAME_PORT})")
        self.root.geometry("600x500")

        self.script = os.path.join(os.path.dirname(__file__), "tcli.py")
        self.coach_names = self.get_coach_names()
        self.python_exe = self.get_python_exe()

        self.client_process = None

        # 模式变量
        self.mode_var = tk.StringVar(value="rule")
        self.seat_var = tk.StringVar(value="1")

        # 参数控件存储
        self.param_widgets = {}

        self.build_ui()

    # ---------- 工具函数 ----------
    def get_coach_names(self):
        coach_dir = os.path.abspath("../coach")
        if not os.path.isdir(coach_dir):
            return ["Demo"]
        names = [d for d in os.listdir(coach_dir)
                 if os.path.isdir(os.path.join(coach_dir, d)) and not d.startswith('__')]
        return names if names else ["Demo"]

    def get_python_exe(self):
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix and 'egg' in os.path.basename(conda_prefix).lower():
            if sys.platform == 'win32':
                return os.path.join(conda_prefix, 'python.exe')
            else:
                return os.path.join(conda_prefix, 'bin', 'python')
        possible_paths = [
            os.path.expanduser('~/miniconda3/envs/egg/python.exe'),
            os.path.expanduser('~/anaconda3/envs/egg/python.exe'),
            'D:/conda_envs/egg/python.exe',
            os.path.join(sys.prefix, 'python.exe')
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return sys.executable

    # ---------- UI 构建 ----------
    def build_ui(self):
        # 模式 & 座位号
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="模式:").grid(row=0, column=0, sticky=tk.W)
        mode_combo = ttk.Combobox(top_frame, textvariable=self.mode_var,
                                  values=["rule", "reinforcement"], state="readonly", width=15)
        mode_combo.grid(row=0, column=1, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_params())

        ttk.Label(top_frame, text="座位号:").grid(row=0, column=2, sticky=tk.W, padx=(20,5))
        ttk.Entry(top_frame, textvariable=self.seat_var, width=4).grid(row=0, column=3)

        # 参数区
        self.param_frame = ttk.LabelFrame(self.root, text="参数", padding=10)
        self.param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.refresh_params()

        # 控制按钮
        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.pack(fill=tk.X)
        ttk.Button(ctrl_frame, text="启动客户端", command=self.start_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="停止客户端", command=self.stop_client).pack(side=tk.LEFT, padx=5)

        # 日志
        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def refresh_params(self):
        # 清空参数区
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_widgets.clear()

        mode = self.mode_var.get()
        row = 0

        # 公共参数
        ttk.Label(self.param_frame, text="服务器 IP:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        host_var = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(self.param_frame, textvariable=host_var, width=15).grid(row=row, column=1, padx=5)
        self.param_widgets["host"] = host_var
        row += 1

        # rule 模式独有参数
        if mode == "rule":
            ttk.Label(self.param_frame, text="教练名称:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            client_var = tk.StringVar(value="Demo")
            combo = ttk.Combobox(self.param_frame, textvariable=client_var,
                                 values=self.coach_names, state="readonly", width=15)
            combo.grid(row=row, column=1, padx=5)
            self.param_widgets["client"] = client_var
            row += 1

            ttk.Label(self.param_frame, text="渲染:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            render_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.param_frame, variable=render_var).grid(row=row, column=1, padx=5, sticky=tk.W)
            self.param_widgets["render"] = render_var

        # reinforcement 模式独有参数
        elif mode == "reinforcement":
            ttk.Label(self.param_frame, text="渲染:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            render_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.param_frame, variable=render_var).grid(row=row, column=1, padx=5, sticky=tk.W)
            self.param_widgets["render"] = render_var
            row += 1

            ttk.Label(self.param_frame, text="设备:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            device_var = tk.StringVar(value="cuda")
            combo = ttk.Combobox(self.param_frame, textvariable=device_var,
                                 values=["cpu", "cuda"], state="readonly", width=8)
            combo.grid(row=row, column=1, padx=5)
            self.param_widgets["device"] = device_var
            row += 1

            ttk.Label(self.param_frame, text="Epsilon:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            epsilon_var = tk.StringVar(value="0.1")
            ttk.Entry(self.param_frame, textvariable=epsilon_var, width=10).grid(row=row, column=1, padx=5)
            self.param_widgets["epsilon"] = epsilon_var
            row += 1

            ttk.Label(self.param_frame, text="Learner IP:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            learner_host_var = tk.StringVar(value="127.0.0.1")
            ttk.Entry(self.param_frame, textvariable=learner_host_var, width=15).grid(row=row, column=1, padx=5)
            self.param_widgets["learner_host"] = learner_host_var
            row += 1

            ttk.Label(self.param_frame, text="Learner 端口:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            learner_port_var = tk.StringVar(value="10000")  # 可以在此修改默认值
            ttk.Entry(self.param_frame, textvariable=learner_port_var, width=10).grid(row=row, column=1, padx=5)
            self.param_widgets["learner_port"] = learner_port_var

    def build_command(self):
        mode = self.mode_var.get()
        seat = self.seat_var.get().strip()
        if not seat.isdigit():
            messagebox.showerror("错误", "座位号必须为正整数")
            return None

        cmd = [self.python_exe, self.script, mode, seat]
        # 添加通用参数
        cmd += ["--host", self.param_widgets["host"].get(), "--port", str(GAME_PORT)]

        if mode == "rule":
            client = self.param_widgets.get("client", tk.StringVar(value="Demo")).get()
            cmd += ["-c", client]
            if self.param_widgets.get("render", tk.BooleanVar(value=False)).get():
                cmd.append("-r")
        elif mode == "reinforcement":
            if self.param_widgets["render"].get():
                cmd.append("-r")
            device = self.param_widgets["device"].get()
            if device:
                cmd += ["--device", device]
            epsilon = self.param_widgets["epsilon"].get()
            if epsilon:
                cmd += ["--epsilon", epsilon]
            learner_host = self.param_widgets["learner_host"].get()
            learner_port = self.param_widgets["learner_port"].get()
            cmd += ["--learner_host", learner_host, "--learner_port", learner_port]

        return cmd

    def start_client(self):
        if self.client_process and self.client_process.poll() is None:
            self.log("客户端已在运行")
            return

        cmd = self.build_command()
        if cmd is None:
            return

        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            env = os.environ.copy()
            self.client_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace',
                text=True,
                bufsize=1,
                cwd=project_root,
                env=env
            )
            self.log(f"客户端已启动 (PID: {self.client_process.pid})")
            threading.Thread(target=self.read_output, daemon=True).start()
        except Exception as e:
            self.log(f"启动失败: {e}")

    def stop_client(self):
        if self.client_process and self.client_process.poll() is None:
            self.client_process.terminate()
            self.log("客户端已终止")
        else:
            self.log("客户端未运行")

    def read_output(self):
        for line in iter(self.client_process.stdout.readline, ""):
            self.log(line.strip())
        self.client_process.stdout.close()

    def log(self, msg):
        def _write():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _write)


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleLauncher(root)
    root.mainloop()