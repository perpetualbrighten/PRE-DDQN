"""
这个文件对生成的测试阶段文件进行数据读取、画图等操作
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_png(save_path, xticklabels, xlabel, ylabel, data_list, ymax=None):
    n_groups = 5
    fig, ax = plt.subplots()
    ax.grid(axis="y", linestyle="--")
    index = np.arange(n_groups)
    bar_width = 0.2

    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, data_list[0],
                    bar_width, color='g',
                    error_kw=error_config, zorder=3,
                    label='MDTF')

    rects2 = ax.bar(index + bar_width * 1, data_list[1],
                    bar_width, color='b',
                    error_kw=error_config, zorder=3,
                    label='MDLF')

    rects3 = ax.bar(index + bar_width * 2, data_list[2],
                    bar_width, color='m',
                    error_kw=error_config, zorder=3,
                    label='ACO')

    rects4 = ax.bar(index + bar_width * 3, data_list[3],
                    bar_width, color='r',
                    error_kw=error_config, zorder=3,
                    label='PER-DDQN')

    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(xticklabels)
    if ymax:
        plt.ylim(ymax=ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.rcParams['font.size'] = 15  # 设置字体大小，全局有效
    plt.legend(title="Strategy")

    plt.rcParams['font.size'] = 20  # 设置字体大小，全局有效
    fig.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


# sns.plotting_context('paper')
# sns.set(font = 'fangsong')  # 解决Seaborn中文显示问题

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.size'] = 20  # 设置字体大小，全局有效

params = {
    'legend.fontsize': 15,
    'figure.figsize': (12, 8),
}
plt.rcParams.update(params)

PATH = r'.\analysis\test_all.csv'
test_df = pd.read_csv(PATH)

mdata = test_df.values[0]
mdeath = test_df.values[1]
aco = test_df.values[2]
dqn = test_df.values[3]
# -------------------------------------- #


xticklabels = ('20', '25', '30', '35', '40')
xlabel = "Number of Sensors"
ylabel = r"Data-Energy Ratio $\eta^{total} $(KB)"

lmdata = []
lmdeath = []
laco = []
ldqn = []

start = 0
for i in range(start, start+15, 3):
    lmdata.append(mdata[i])
    lmdeath.append(mdeath[i])
    laco.append(aco[i])
    ldqn.append(dqn[i])

data_list = (lmdata, lmdeath, laco, ldqn)

draw_png(
    r".\demo\p2.png",
    xticklabels, xlabel, ylabel, data_list)


xticklabels = ('20', '25', '30', '35', '40')
xlabel = "Number of Sensors"
ylabel = r"Flight Time(s)"

lmdata = []
lmdeath = []
laco = []
ldqn = []

start = 1
for i in range(start, start+15, 3):
    lmdata.append(mdata[i])
    lmdeath.append(mdeath[i])
    laco.append(aco[i])
    ldqn.append(dqn[i])

data_list = (lmdata, lmdeath, laco, ldqn)

draw_png(
    r".\demo\p3-1.png",
    xticklabels, xlabel, ylabel, data_list)


xticklabels = ('20', '25', '30', '35', '40')
xlabel = "Number of Sensors"
ylabel = r"Execution Time (s)"

lmdata = []
lmdeath = []
laco = []
ldqn = []

start = 2
for i in range(start, start+15, 3):
    lmdata.append(mdata[i])
    lmdeath.append(mdeath[i])
    laco.append(aco[i])
    ldqn.append(dqn[i])

data_list = (lmdata, lmdeath, laco, ldqn)

draw_png(
    r".\demo\p4.png",
    xticklabels, xlabel, ylabel, data_list)


# 数据采集量
xticklabels = ('20', '25', '30', '35', '40')
xlabel = "Number of Sensors"
ylabel = r"Amount of Data Collected (MB)"

lmdata = []
lmdeath = []
laco = []
ldqn = []

start = 15
for i in range(start, start+5, 1):
    lmdata.append(mdata[i])
    lmdeath.append(mdeath[i])
    laco.append(aco[i])
    ldqn.append(dqn[i])

data_list = (lmdata, lmdeath, laco, ldqn)

draw_png(
    r".\demo\p3-2.png",
    xticklabels, xlabel, ylabel, data_list)


