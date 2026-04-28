# -*- coding: utf-8 -*-
"""
增强版 GUI 启动器 v14 — 端口反查 + os.kill 暴力终止服务器
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import sys
import os
import threading
import time
import socket
import signal                      # ✅ 新增：用于 os.kill


class GameLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("掼蛋游戏多客户端启动器")
        self.root.geometry("900x750")

        self.script = os.path.join(os.path.dirname(__file__), "clients", "gene_client.py")
        self.coach_names = self.get_coach_names()
        self.python_exe = self.get_python_exe()

        self.server_process = None
        self.server_pid = None
        self.client_processes = {}

        self.modes = {
            "rule": {
                "name": "规则 / 自定义教练",
                "params": {
                    "client": {"label": "教练名称", "type": "coach", "default": "Demo"},
                    "render": {"label": "渲染画面", "type": bool, "default": False}
                }
            },
            "imitation": {
                "name": "模仿学习 (DAgger)",
                "params": {
                    "render": {"label": "渲染画面", "type": bool, "default": False},
                    "model": {"label": "模型路径", "type": str, "default": ""},
                    "lr": {"label": "学习率", "type": float, "default": 1e-4},
                    "device": {"label": "设备", "type": str, "default": "cpu", "option": ["cpu", "cuda"]},
                    "dagger_epochs": {"label": "每轮训练 epoch", "type": int, "default": 3},
                    "dagger_interval": {"label": "训练间隔(局)", "type": int, "default": 5},
                    "expert_decay": {"label": "专家衰减系数", "type": float, "default": 0.995},
                    "expert_init": {"label": "初始专家概率", "type": float, "default": 1.0},
                    "min_expert_prob": {"label": "最低专家概率", "type": float, "default": 0.1},
                    "batch_size": {"label": "Batch Size", "type": int, "default": 16},
                    "min_dataset_size": {"label": "最小数据集大小", "type": int, "default": 32},
                    "log_interval": {"label": "日志记录间隔(步)", "type": int, "default": 50}
                }
            },
            "reinforcement": {
                "name": "强化学习 (DQN)",
                "params": {
                    "render": {"label": "渲染画面", "type": bool, "default": False},
                    "model": {"label": "模型路径", "type": str, "default": ""},
                    "lr": {"label": "学习率", "type": float, "default": 1e-3},
                    "device": {"label": "设备", "type": str, "default": "cpu", "option": ["cpu", "cuda"]},
                    "epsilon": {"label": "Epsilon", "type": float, "default": 0.1},
                    "gamma": {"label": "Gamma", "type": float, "default": 0.98},
                    "save_interval": {"label": "保存间隔(局)", "type": int, "default": 25},
                    "log_interval": {"label": "日志记录间隔(局)", "type": int, "default": 50}
                }
            },
            "test": {
                "name": "测试模型",
                "params": {
                    "render": {"label": "渲染画面", "type": bool, "default": True},
                    "model": {"label": "模型路径 (必填)", "type": str, "default": ""},
                    "device": {"label": "设备", "type": str, "default": "cpu", "option": ["cpu", "cuda"]},
                    "epsilon": {"label": "Epsilon", "type": float, "default": 0.1}
                }
            }
        }

        self.seat_mode_vars = [tk.StringVar(value="rule") for _ in range(4)]
        self.seat_widgets = [{} for _ in range(4)]
        self.seat_param_frames = [None] * 4

        self.build_ui()

    # ---------- 工具函数 ----------
    def get_coach_names(self):
        coach_dir = os.path.abspath("./coach")
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
        server_frame = ttk.LabelFrame(self.root, text="服务器控制", padding=10)
        server_frame.pack(fill=tk.X, padx=10, pady=5)

        self.server_status_var = tk.StringVar(value="未启动")
        ttk.Label(server_frame, text="服务器状态：").pack(side=tk.LEFT)
        ttk.Label(server_frame, textvariable=self.server_status_var, foreground="red").pack(side=tk.LEFT, padx=5)

        ttk.Label(server_frame, text="局数:").pack(side=tk.LEFT, padx=(20, 5))
        self.server_rounds_var = tk.StringVar(value="10")
        ttk.Entry(server_frame, textvariable=self.server_rounds_var, width=6).pack(side=tk.LEFT)

        ttk.Button(server_frame, text="启动服务器", command=self.start_server_btn).pack(side=tk.LEFT, padx=20)
        ttk.Label(server_frame, text="路径: ./simulator/windows/server.exe").pack(side=tk.LEFT, padx=10)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for i in range(4):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=f"座位 {i + 1}")
            self.build_seat_tab(tab, i)

        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_notebook = ttk.Notebook(log_frame)
        self.log_notebook.pack(fill=tk.BOTH, expand=True)

        self.log_texts = []
        for i in range(4):
            log_tab = ttk.Frame(self.log_notebook)
            self.log_notebook.add(log_tab, text=f"Client {i + 1}")
            log_text = scrolledtext.ScrolledText(log_tab, height=6, state=tk.DISABLED)
            log_text.pack(fill=tk.BOTH, expand=True)
            self.log_texts.append(log_text)

        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.pack(fill=tk.X)

        ttk.Button(ctrl_frame, text="启动全部客户端", command=self.start_all_clients).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="停止所有进程", command=self.stop_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="清空所有日志", command=self.clear_logs).pack(side=tk.RIGHT, padx=5)

    def build_seat_tab(self, tab, seat_idx):
        mode_frame = ttk.Frame(tab)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(mode_frame, text="模式:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.seat_mode_vars[seat_idx],
                                  values=list(self.modes.keys()), state="readonly", width=15)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", lambda e, idx=seat_idx: self.refresh_seat_params(idx))

        self.seat_param_frames[seat_idx] = ttk.LabelFrame(tab, text="参数", padding=5)
        self.seat_param_frames[seat_idx].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.refresh_seat_params(seat_idx)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="🔗 连接此客户端",
                   command=lambda idx=seat_idx: self.start_single_client(idx)).pack(side=tk.LEFT, padx=5, pady=5)

    def refresh_seat_params(self, seat_idx):
        frame = self.seat_param_frames[seat_idx]
        for widget in frame.winfo_children():
            widget.destroy()

        mode = self.seat_mode_vars[seat_idx].get()
        if mode not in self.modes:
            return

        param_defs = self.modes[mode]["params"]
        self.seat_widgets[seat_idx] = {}
        row = 0
        for pname, info in param_defs.items():
            label = ttk.Label(frame, text=info["label"] + ":")
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            ptype = info["type"]
            default = info.get("default", "")
            widget = None
            if pname == "client":
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(frame, textvariable=var, values=self.coach_names, state="readonly", width=20)
                widget.var = var
            elif ptype == bool:
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(frame, variable=var)
                widget.var = var
            elif "option" in info:
                var = tk.StringVar(value=str(default))
                widget = ttk.Combobox(frame, textvariable=var, values=info["option"], state="readonly", width=10)
                widget.var = var
            elif pname == "model":
                var = tk.StringVar(value=str(default))
                entry = ttk.Entry(frame, textvariable=var, width=25)
                entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                btn = ttk.Button(frame, text="浏览...", command=lambda v=var: self.browse_model(v))
                btn.grid(row=row, column=2, padx=5, pady=2)
                widget = entry
                widget.var = var
            else:
                var = tk.StringVar(value=str(default))
                widget = ttk.Entry(frame, textvariable=var, width=20)
                widget.var = var

            if pname != "model" and pname != "client":
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            elif pname == "client":
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)

            self.seat_widgets[seat_idx][pname] = (widget, ptype if ptype not in ("coach",) else str)
            row += 1

    def browse_model(self, var):
        initial_dir = os.path.abspath("./model")
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            initialdir=initial_dir,
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            var.set(filename)

    def get_seat_params(self, seat_idx):
        params = {}
        for name, (widget, ptype) in self.seat_widgets[seat_idx].items():
            if ptype == bool:
                val = widget.var.get()
            else:
                raw = widget.var.get().strip()
                if raw == "":
                    val = "" if ptype == str else None
                else:
                    try:
                        val = ptype(raw)
                    except ValueError:
                        messagebox.showerror("参数错误", f"座位{seat_idx + 1} 参数 {name} 格式错误")
                        return None
            params[name] = val
        return params

    def build_client_command(self, seat_idx):
        mode = self.seat_mode_vars[seat_idx].get()
        params = self.get_seat_params(seat_idx)
        if params is None:
            return None

        cmd = [self.python_exe, self.script, mode, str(seat_idx + 1)]
        for name, value in params.items():
            if value is None or value == "":
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{name}")
            else:
                cmd.append(f"--{name}")
                cmd.append(str(value))
        return cmd

    # ---------- 端口检测 ----------
    def is_port_in_use(self, port=23456):
        """使用 netstat 检查是否有进程在 LISTENING 指定端口"""
        try:
            cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # 如果找到了包含 LISTENING 的行，说明端口被占用
            for line in result.stdout.split('\n'):
                if 'LISTENING' in line:
                    return True
            return False
        except Exception:
            return False

    # ---------- 服务器控制 ----------
    def start_server_btn(self):
        # 如果端口被占用，直接暴力清理
        if self.is_port_in_use(23456):
            self.log("端口 23456 被占用，暴力清理...")
            self._kill_process_by_port(23456)
            time.sleep(1)
            if self.is_port_in_use(23456):
                messagebox.showerror("错误", "端口仍被占用，请手动关闭占用程序或重启电脑")
                return

        rounds_str = self.server_rounds_var.get().strip()
        if not rounds_str.isdigit() or int(rounds_str) <= 0:
            messagebox.showerror("错误", "请输入有效的游戏局数（正整数）")
            return

        self.start_server()

    def start_server(self):
        server_path = os.path.abspath("./simulator/windows/server.exe")
        if not os.path.exists(server_path):
            messagebox.showerror("错误", f"服务器程序不存在: {server_path}")
            return False

        if self.server_process and self.server_process.poll() is None:
            self.log("服务器已在运行")
            self.server_status_var.set("运行中")
            return True

        rounds = self.server_rounds_var.get().strip()
        cmd = [server_path, rounds]
        try:
            # 简单启动，不保持进程关联（关闭时用端口反查终止）
            self.server_process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(server_path),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            self.server_pid = self.server_process.pid
            self.server_status_var.set("启动中...")
            self.log(f"服务器正在启动（PID: {self.server_pid}，局数: {rounds}）...")

            threading.Thread(target=self._wait_and_update_status, daemon=True).start()
            return True
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self.server_status_var.set("启动失败")
            return False

    def _wait_and_update_status(self):
        start = time.time()
        timeout = 15
        while time.time() - start < timeout:
            if self.server_process and self.server_process.poll() is not None:
                ret = self.server_process.poll()
                self.server_status_var.set("已退出")
                self.log(f"服务器进程意外退出，返回码: {ret}")
                return
            try:
                with socket.create_connection(('127.0.0.1', 23456), timeout=1):
                    self.server_status_var.set("运行中")
                    self.log("服务器端口已就绪。")
                    return
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(0.5)
        self.server_status_var.set("未就绪")
        self.log("警告：服务器未能在预期时间内就绪。")

    def ensure_server_ready(self):
        if not self.server_process or self.server_process.poll() is not None:
            ret = self.server_process.poll() if self.server_process else None
            self.log(f"服务器未运行 (poll={ret})，请先启动服务器。")
            return False
        if not self.wait_for_server(timeout=3):
            self.log("服务器端口未就绪，请稍后重试。")
            return False
        return True

    def wait_for_server(self, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection(('127.0.0.1', 23456), timeout=1):
                    return True
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(0.5)
        return False

    # ----- ✅ 终极暴力关闭：端口反查 + os.kill -----
    def stop_server(self):
        """暴力关闭服务器（端口反查 + os.kill）"""
        if self.server_process is None:
            return
        self.log("正在关闭服务器...")
        self._kill_process_by_port(23456)
        self.server_process = None
        self.server_pid = None
        self.server_status_var.set("已停止")

    def _kill_process_by_port(self, port=23456):
        """查找占用端口的 PID 并用 os.kill 终止"""
        self.log(f"[清理端口 {port}]")
        try:
            cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                self.log("端口空闲")
                return

            pid = None
            for line in result.stdout.strip().split('\n'):
                if 'LISTENING' in line:
                    parts = line.split()
                    if parts[-1].isdigit():
                        pid = int(parts[-1])
                        break
            if not pid:
                self.log("未找到 LISTENING 进程")
                return

            self.log(f"找到 PID: {pid}，发送终止信号")
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            if not self.is_port_in_use(port):
                self.log("端口已释放")
            else:
                self.log("⚠ 未能释放端口，请尝试管理员身份运行")
        except Exception as e:
            self.log(f"清理异常: {e}")

    # ---------- 客户端管理（不变）----------
    def start_single_client(self, seat_idx):
        if not self.ensure_server_ready():
            return
        if seat_idx in self.client_processes and self.client_processes[seat_idx].poll() is None:
            self.log(f"Client {seat_idx + 1} 已在运行", seat_idx)
            return

        cmd = self.build_client_command(seat_idx)
        if cmd is None:
            return
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            env = os.environ.copy()
            proc = subprocess.Popen(
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
            self.client_processes[seat_idx] = proc
            self.log(f"Client {seat_idx + 1} 已启动 (Python: {self.python_exe})", seat_idx)
            threading.Thread(target=self.read_process_output,
                             args=(proc, f"Client{seat_idx + 1}"), daemon=True).start()
        except Exception as e:
            self.log(f"Client {seat_idx + 1} 启动失败: {e}", seat_idx)

    def start_all_clients(self):
        if not self.ensure_server_ready():
            return
        self.log("开始依次启动全部客户端...")
        for i in range(4):
            self.start_single_client(i)
            time.sleep(0.5)

    def stop_all(self):
        for idx, proc in list(self.client_processes.items()):
            if proc and proc.poll() is None:
                proc.terminate()
                self.log(f"Client {idx + 1} 已终止", idx)
        self.client_processes.clear()
        self.stop_server()

    def read_process_output(self, proc, tag):
        for line in iter(proc.stdout.readline, ""):
            if tag.startswith("Client"):
                idx = int(tag[-1]) - 1
                self.log(line.strip(), idx)
            else:
                self.log(f"[{tag}] {line.strip()}")
        proc.stdout.close()

    def log(self, message, client_idx=None):
        def _write(msg=message, idx=client_idx):
            if idx is not None and 0 <= idx < 4:
                text_widget = self.log_texts[idx]
                prefix = ""
            else:
                text_widget = self.log_texts[0] if self.log_texts else None
                prefix = "[全局] " if text_widget else ""
            if text_widget:
                text_widget.configure(state=tk.NORMAL)
                text_widget.insert(tk.END, prefix + msg + "\n")
                text_widget.see(tk.END)
                text_widget.configure(state=tk.DISABLED)
        self.root.after(0, _write)

    def clear_logs(self):
        for text in self.log_texts:
            text.configure(state=tk.NORMAL)
            text.delete(1.0, tk.END)
            text.configure(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = GameLauncher(root)
    root.mainloop()