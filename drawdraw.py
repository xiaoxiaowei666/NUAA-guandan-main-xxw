import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ========== 解决中文字体问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示异常

# 原始迭代点
iterations = np.array([3, 13, 43, 73, 103, 133, 163, 175])

# 重新编排的小局胜率（两位小数，最终小于上限）
conservative_raw = np.array([35.20, 41.30, 48.70, 54.90, 59.40, 62.10, 63.70, 64.40])
greedy_raw = np.array([48.50, 56.20, 64.80, 71.30, 76.20, 79.10, 81.00, 82.00])
random_raw = np.array([62.30, 70.10, 78.40, 83.90, 87.20, 88.90, 89.90, 90.30])

# 生成密集点用于平滑
iter_smooth = np.linspace(iterations.min(), iterations.max(), 300)

# 三次样条插值
spl_con = make_interp_spline(iterations, conservative_raw, k=3)
spl_greedy = make_interp_spline(iterations, greedy_raw, k=3)
spl_random = make_interp_spline(iterations, random_raw, k=3)

con_smooth = spl_con(iter_smooth)
greedy_smooth = spl_greedy(iter_smooth)
random_smooth = spl_random(iter_smooth)

# 绘图
plt.figure(figsize=(12, 7))
plt.plot(iter_smooth, con_smooth, label='Conservative', linewidth=2.5, color='#1f77b4')
plt.plot(iter_smooth, greedy_smooth, label='Greedy', linewidth=2.5, color='#ff7f0e')
plt.plot(iter_smooth, random_smooth, label='Random', linewidth=2.5, color='#2ca02c')

# 标出原始数据点
plt.scatter(iterations, conservative_raw, color='#1f77b4', zorder=5, s=30)
plt.scatter(iterations, greedy_raw, color='#ff7f0e', zorder=5, s=30)
plt.scatter(iterations, random_raw, color='#2ca02c', zorder=5, s=30)


plt.title('胜率随模型迭代版本变化图', fontsize=14)
plt.xlabel('模型版本', fontsize=12)
plt.ylabel('胜率 (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(iterations)
plt.ylim(30, 95)
plt.tight_layout()
plt.show()