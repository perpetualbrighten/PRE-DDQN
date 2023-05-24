"""
无人机数据采集任务的路径规划算法的底层实现
"""

import sys
import numpy as np
import time
import math
import copy

from common import *
from tools import *
from tf_model_perddqn import tf_model_perddqn
from gym import spaces
from gym.utils import seeding

np.random.seed(1)

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


# 定义无人机类
class UAV(tk.Tk):
    def __init__(self, is_train=True, R_dc=10., R_eh=30.):
        super(UAV, self).__init__()
        # POI位置
        self.canvas = None
        self.np_random = None
        self.N_POI = N_S_  # 传感器数量
        self.dis = np.zeros(self.N_POI)  # 距离的平方
        self.elevation = np.zeros(self.N_POI)  # 仰角
        self.pro = np.zeros(self.N_POI)  # 视距概率
        self.h = np.zeros(self.N_POI)  # 信道增益
        self.N_UAV = 1
        self.max_speed = V
        self.H = H  # 无人机飞行高度 10m
        self.energy = E
        self.X_min = 0
        self.Y_min = 0
        self.X_max = WIDTH * UNIT
        self.Y_max = HEIGHT * UNIT  # 地图边界
        self.R_dc = R_dc  # 水平覆盖距离 10m
        self.R_eh = R_eh  # 水平覆盖距离 30m
        self.sdc = math.sqrt(pow(self.R_dc, 2) + pow(self.H, 2))  # 最大DC服务距离
        self.seh = math.sqrt(pow(self.R_eh, 2) + pow(self.H, 2))  # 最大EH服务距离
        self.noise = pow(10, -12)  # 噪声功率为-90dbm
        self.AutoUAV = []
        self.Aim = []
        self.N_AIM = 1  # 选择服务的用户数
        self.FX = 0.
        self.time = 0
        self.has_visited = {}
        self.has_collected = 0.
        self.SoPcenter = np.random.randint(0, 1000, size=[self.N_POI, 2])

        self.action_space = spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]),
                                       dtype=np.float32)
        self.state_dim = STATE_DIM  # 状态空间为每一个传感器的相对位置、预期剩余时间、缓存数据量以及当前电量
        self.state = np.zeros([N_S_, self.state_dim])
        self.xy = np.zeros((self.N_UAV, 2))  # 无人机位置

        # 假设有4类传感器，即有4个不同的泊松参数,随机给传感器分配泊松参数
        CoLDA = np.random.randint(0, len(LDA), self.N_POI)
        self.lda = [LDA[CoLDA[i]] for i in range(self.N_POI)]  # 给传感器们指定数据增长速度

        self.b_S_pos = np.random.randint(0, 1000, [N_S_, 2])  # 初始化传感器位置
        self.b_S_cache = np.random.randint(200, 800, N_S_)  # 初始化传感器当前数据缓存量
        self.b_S_death = np.random.randint(4000, 8000, N_S_)  # 初始化数据的死亡时间

        self.Fully_buffer = C
        self.N_Data_death = 0  # 数据死亡计数
        self.Q = np.array(
            [self.lda[i] * self.b_S_cache[i] / self.Fully_buffer for i in range(self.N_POI)])
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S_cache[self.idx_target] / self.Fully_buffer

        # 构建PER-DDQN网络
        self.model = tf_model_perddqn(is_trainable=is_train)

        self.title('MAP')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))  # Tkinter 的几何形状
        self.build_maze()

    # 创建地图
    def build_maze(self):
        # 创建画布 Canvas.白色背景，宽高。
        self.canvas = tk.Canvas(self, bg='white', width=WIDTH * UNIT, height=HEIGHT * UNIT)

        # 创建用户
        for i in range(self.N_POI):
            # 创建椭圆，指定起始位置。填充颜色
            if self.lda[i] == LDA[0]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='pink')
            elif self.lda[i] == LDA[1]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='blue')
            elif self.lda[i] == LDA[2]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='green')
            elif self.lda[i] == LDA[3]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 5, self.SoPcenter[i][1] - 5,
                    self.SoPcenter[i][0] + 5, self.SoPcenter[i][1] + 5,
                    fill='red')

        # 创建无人机
        self.xy = [[0., 0.]]

        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)

        # 用户选择
        pxy = self.SoPcenter[np.argmax(self.Q)]
        L_AIM = self.canvas.create_rectangle(
            pxy[0] - 10, pxy[1] - 10,
            pxy[0] + 10, pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.canvas.pack()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 重置，随机初始化无人机的位置
    def reset(self):
        self.render()
        for i in range(self.N_UAV):
            self.canvas.delete(self.AutoUAV[i])
        self.AutoUAV = []

        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])

        # 初始化无人机位置
        self.xy = [[0., 0.]]
        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - R_d, self.xy[i][1] - R_d,
                self.xy[i][0] + R_d, self.xy[i][1] + R_d,
                fill='yellow')
            self.AutoUAV.append(L_UAV)
        self.FX = 0.

        self.b_S_pos = np.random.randint(0, 1000, [N_S_, 2])  # 初始化传感器位置
        self.b_S_cache = np.random.randint(200, 800, N_S_)  # 初始化传感器当前数据缓存量
        self.b_S_death = np.random.randint(4000, 8000, N_S_)  # 初始化数据的死亡时间

        self.N_Data_death = 0  # 数据死亡计数
        self.Q = np.array([self.lda[i] * self.b_S_cache[i] / self.Fully_buffer for i in range(self.N_POI)])  # 数据收集优先级

        # 初始化状态空间值
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S_cache[self.idx_target] / self.Fully_buffer
        self.pxy = self.SoPcenter[self.idx_target]  # 初始选择优先级最大的

        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 10, self.pxy[1] - 10,
            self.pxy[0] + 10, self.pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.state = np.zeros([N_S_, self.state_dim])
        for i in range(N_S_):
            self.state[i][0] = calc_distance(self.xy, [[self.b_S_pos[i][0], self.b_S_pos[i][1]]])
            self.state[i][1] = b_S_death[i] - self.time
            self.state[i][2] = b_S_cache[i]
            self.state[i][3] = self.energy

        return self.state

    # 传入当前状态和输入动作输出下一个状态和奖励
    def step_move(self, action):

        state_ = np.zeros([N_S_, self.state_dim])
        xy_ = [[self.b_S_pos[action][0], self.b_S_pos[action][1]]]
        D = calc_distance(self.xy, xy_)

        ec = 0 # 能量消耗
        ec += PF * D / self.max_speed
        if self.b_S_death[action] >= self.time + D / self.max_speed:
            ec += (PH + PT) * self.b_S_cache[action] / trans_speed()  # 悬停能耗 + 传输能耗
        else:
            ec += 0

        for i in range(self.N_UAV):
            self.canvas.move(self.AutoUAV[i], xy_[i][0] - self.xy[i][0], xy_[i][1] - self.xy[i][1])

        self.xy = xy_
        for i in range(self.N_POI):  # 数据溢出处理
            if self.b_S_cache[i] >= self.Fully_buffer:
                self.N_Data_death += 1  # 数据死亡用户计数
                self.b_S_cache[i] = self.Fully_buffer
        self.updata = self.b_S_cache[self.idx_target] / self.Fully_buffer

        # 获取下一步状态空间
        for i in range(N_S_):
            state_[i][0] = calc_distance(self.xy, [[self.b_S_pos[i][0], self.b_S_pos[i][1]]])
            state_[i][1] = b_S_death[i] - self.time
            state_[i][2] = b_S_cache[i]
            state_[i][3] = self.energy

        reward = get_reward(D, self.b_S_death[action] - self.time - D / self.max_speed)
        self.Q_dis()

        Done = False
        if self.b_S_death[action] < self.time + D / self.max_speed:
            reward -= LAMBDA * D
        if self.energy <= 0 or len(self.has_visited) == N_S_:
            Done = True
            reward -= 1 - self.has_collected / sum(b_S_cache)
            # 只给目标用户收集数据

        self.model.insert_experience(self.state[action], reward, Done, state_[action])

        # print(self.energy)
        self.state = state_
        self.energy -= ec
        self.time += D / self.max_speed
        self.has_visited[action] = True
        self.has_collected += self.b_S_cache[action]

        return state_, reward, Done

    def step_hover(self, hover_time):
        # 无人机不动，所以是s[:5]不变
        self.N_Data_death = 0  # 记录每时隙数据溢出用户数
        self.b_S_cache += [np.random.poisson(self.lda[i]) * hover_time for i in range(self.N_POI)]  # 传感器数据缓存量更新
        for i in range(self.N_POI):  # 数据溢出处理
            if self.b_S_cache[i] >= self.Fully_buffer:
                self.N_Data_death += 1  # 数据溢出用户计数
                self.b_S_cache[i] = self.Fully_buffer
        self.updata = self.b_S_cache[self.idx_target] / self.Fully_buffer
        self.state[5] = self.N_Data_death / self.N_POI  # 数据死亡占比

    # 每次无人机更新位置后，计算无人机与所有用户的距离与仰角，以及路径增益
    def Q_dis(self):
        for i in range(self.N_POI):
            self.dis[i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[0][0], 2) + pow(self.SoPcenter[i][1] - self.xy[0][1], 2) + pow(
                    self.H, 2))  # 原始距离
            self.elevation[i] = 180 / math.pi * np.arcsin(self.H / self.dis[i])  # 仰角
            self.pro[i] = 1 / (1 + 10 * math.exp(-0.6 * (self.elevation[i] - 10)))  # 视距概率
            self.h[i] = \
                (self.pro[i] + (1 - self.pro[i]) * 0.2) * pow(self.dis[i], -2.3) * pow(10, -30 / 10)  # 参考距离增益为-30db

    # 输入是10-4W~10-5W,输出是0~9.079muW
    def Non_linear_EH(self, Pr):
        if Pr == 0:
            return 0
        P_prac = Mj / (1 + math.exp(-aj * (Pr - bj)))
        Peh = (P_prac - Mj * Oj) / (1 - Oj)  # 以W为单位
        return Peh * pow(10, 6)

    # 输入是10-4W~10-5W,输出是0~9.079muW
    def linear_EH(self, Pr):
        if Pr == 0:
            return 0
        return Pr * pow(10, 6) * 0.2

    # 重选目标用户
    def CHOOSE_AIM(self):
        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])

        # 重选目标用户
        self.Q = np.array([self.lda[i] * self.b_S_cache[i] / C for i in range(self.N_POI)])  # 数据收集优先级
        self.idx_target = np.argmax(self.Q)
        self.updata = self.b_S_cache[self.idx_target] / self.Fully_buffer
        self.pxy = self.SoPcenter[self.idx_target]
        L_AIM = self.canvas.create_rectangle(
            self.pxy[0] - 10, self.pxy[1] - 10,
            self.pxy[0] + 10, self.pxy[1] + 10,
            fill='red')
        self.Aim.append(L_AIM)

        self.state[:2] = (self.pxy - self.xy[0]).flatten() / 400.
        self.render()
        return self.state

    # 调用Tkinter的update方法, 0.01秒去走一步。
    def render(self, epcho_time=0.01):
        time.sleep(epcho_time)
        self.update()

    def sample_action(self):
        action, is_max_Q = self.model.get_decision(self.state)
        return action


def update():
    for epcho in range(30):
        env.reset()
        while True:
            env.render()
            action = env.sample_action()
            s, r, done = env.step_move(action)
            if done:
                break


if __name__ == '__main__':
    env = UAV()
    env.after(10, update)
    env.mainloop()
