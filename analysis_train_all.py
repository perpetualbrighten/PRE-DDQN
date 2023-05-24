"""
这个文件对生成的训练阶段文件进行数据读取、画图等操作
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator   # 设置刻度

# sns.plotting_context('paper')

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.size'] = 20  # 设置字体大小，全局有效


params = {
    'legend.fontsize': 15,
    'figure.figsize': (12, 8),
}
plt.rcParams.update(params)


PATH_LIST = [
    r'.\analysis\train_ucb.csv',
    r'.\analysis\train_greedy.csv'
]

raw_data = []
for i in range(2):
    train_df = pd.read_csv(PATH_LIST[i])
    raw_data.append(train_df)
    print(train_df.info())


x = raw_data[0].index

y_dt_d = raw_data[0]['number']
y_ql_d = raw_data[1]['number']


# print(y_mr, y_tr)
plt.figure(figsize=(12, 8))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
# 颜色绿色，点形圆形，线性虚线，设置图例显示内容，线条宽度为2

plt.plot(x, y_dt_d, 'r-', linewidth=1, markersize=5, label=r"Adaptive UCB")
plt.plot(x, y_ql_d, 'b-', linewidth=1, markersize=5, label=r"$\epsilon$-greedy")


plt.xlabel("Number of Epoch")  # 横坐标轴的标题
plt.ylabel(r"Data-Energy Ratio $\eta^{total}$ (KB)")  # 纵坐标轴的标题
plt.legend()

plt.savefig(fname=r".\demo\data_process.png", dpi=600)
plt.show()

# plt.ylim([0.50, 1])  # 设置纵坐标轴范围为 -2 到 2
# plt.grid(linestyle="--")  # 设置背景网格线为虚线
# # plt.grid() # 显示网格
# # plt.title('result') # 图形的标题


