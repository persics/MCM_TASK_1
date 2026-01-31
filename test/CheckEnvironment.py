# ======================================
# 美赛Python环境验证
# ======================================
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats  # 统计分析
from scipy.optimize import curve_fit  # 曲线拟合
import random

# 智能优化算法库
import pygad  # 粒子群/遗传算法
from deap import base, creator, tools, algorithms  # 遗传算法拓展

# 打印版本信息（验证导入成功）
print("="*50)
print("✅ 环境验证成功，库版本信息：")
print(f"Python版本：{sys.version.split()[0]}")
print(f"Pandas版本：{pd.__version__} | NumPy版本：{np.__version__}")
print(f"Matplotlib版本：{matplotlib.__version__}")
print("="*50)

# 全局设置：解决Matplotlib中文显示问题（美赛可省略，用英文标注即可）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
plt.rcParams['figure.dpi'] = 300              # 全局高清图（300dpi，适配论文）
plt.rcParams['savefig.dpi'] = 300

# 绘图逻辑（必须有这部分）
# ======================
# 1. 创建画布
plt.figure(figsize=(8, 5))
# 2. 绘制图形（示例：折线图）
x = np.linspace(0, 10, 250)  #曲线的采样点，一般200-500
y = np.sin(x)
plt.plot(x, y, color="#1f77b4", linewidth=2, label="sin(x)")
# 3. 添加标注（美赛论文必备）
plt.title("Test Plot for MCM Environment", fontsize=14, fontweight="bold")
plt.xlabel("X Axis (Unit)")
plt.ylabel("Y Axis (Unit)")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)

# 4. 保存图片（带bbox_inches去掉白边）
plt.savefig("trend_plot.png", bbox_inches='tight')
plt.close()  # 关闭画布，释放内存