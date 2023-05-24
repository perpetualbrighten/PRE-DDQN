"""
公共参数部分
"""
import numpy as np
import math

# UAV 环境相关
WIDTH = 1000  # 地图的宽度
HEIGHT = 1000  # 地图的高度
UNIT = 1  # 每个方块的大小（像素值）   1000m*1000m的地图
LDA = [4., 8., 15., 20.]  # 假设有4类传感器，即有4个不同的泊松参数,传感器数据生成服从泊松分布
max_LDA = max(LDA)
E = 400     # UAV的能量
C = 10000  # 传感器的容量都假设为10000
P_u = pow(10, -5)  # 传感器的发射功率 0.01mW,-20dbm
P_d = 10  # 无人机下行发射功率 10W,40dBm
H = 15.  # 无人机固定悬停高度
R_d = 30.  # 无人机充电覆盖范围 10m能接受到0.1mW,30m能接收到0.01mW
N_S_ = 30  # 设备个数
V = 1  # 无人机均速飞行速度 1m/s

b_S_pos = np.random.randint(0, 1000, [N_S_, 2])  # 初始化传感器位置
b_S_cache = np.random.randint(200, 800, N_S_)  # 初始化传感器当前数据缓存量
b_S_death = np.random.randint(4000, 8000, N_S_)  # 初始化数据的死亡时间

# 能耗
PF = 0.01
PH = 0.02
PT = 0.25


# 非线性能量接收机模型
Mj = 9.079 * pow(10, -6)
aj = 47083
bj = 2.9 * pow(10, -6)
Oj = 1 / (1 + math.exp(aj * bj))

# 马尔可夫决策相关
STATE_DIM = 4
ACTION_DIM = 1

# 奖励函数相关
GAMMA = 0.99
ALPHA = 0.01
BETA = 0.001
LAMBDA = 0.1
OMEGA = 100


