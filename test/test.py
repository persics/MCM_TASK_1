# 1. 数据处理（Pandas）
import pandas as pd
data = pd.DataFrame({'x': [1,2,3,4,5], 'y': [2,4,5,4,5]})
print("数据预览：\n", data.head())

# 2. 可视化（Matplotlib）
import matplotlib.pyplot as plt
plt.plot(data['x'], data['y'], 'o-', label='模拟数据')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('test_plot.png') 
plt.close()

# 3. 蒙特卡洛模拟示例
import numpy as np
n = 10000
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
pi_estimate = 4 * sum(x**2 + y**2 < 1) / n
print("蒙特卡洛模拟估算π值：", pi_estimate)