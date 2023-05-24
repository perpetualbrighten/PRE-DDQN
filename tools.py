"""
实现各种函数计算相关
"""

import numpy as np
from math import pi, atan2
from common import *

ETX = 50 * (10 ** (-9))  # Energy for transferring of each bit:发射单位报文损耗能量:50nJ/bit
ERX = 50 * (10 ** (-9))  # Energy for receiving of each bit:接收单位报文损耗能量:50nJ/bit
Efs = 10 * (10 ** (-12))  # Energy of free space model:自由空间传播模型:10pJ/bit/m2
K = 2000  # 系数k用于对齐
P = 0.001  # 功率为0.001J/s


def ucb_Q(sum_cnt, single_cnt):
    res = 2 * np.log(sum_cnt + 1)
    res /= single_cnt + 1
    return np.sqrt(res)


def get_reward(D, T, ):
    base_reward = 0
    base_reward -= ALPHA * D
    if T >= 0:
        base_reward -= BETA * T
    return base_reward


def calc_move(pos_start, pos_end):
    """
    计算两个位置间移动向量
    :param pos_start:起始点
    :param pos_end:终点
    :return:向量
    """
    return pos_end[0][0] - pos_start[0][0], pos_end[0][1] - pos_start[0][1]


def calc_distance(pos_1, pos_2):
    """
    计算两个位置间距离
    :param pos_1:位置1
    :param pos_2:位置2
    :return:距离
    """
    d1, d2 = calc_move(pos_1, pos_2)
    return (d1 ** 2 + d2 ** 2) ** 0.5


def calc_region_id(base_pos, neighbor_pos):
    move = calc_move(base_pos, neighbor_pos)
    theta = atan2(move[1], move[0]) / pi * 180
    # 45 degree pre step
    move_pos = round(theta / 45)
    # using right hand
    if move_pos < 0:
        move_pos += 8
    return move_pos


def cross_product(move_1, move_2):
    return move_1[0] * move_2[1] - move_1[1] * move_2[0]


def calc_counterclockwise(move_1, move_2):
    return cross_product(move_1, move_2) > 0


def calc_intersect(line_1_start, line_1_end, line_2_start, line_2_end):
    def get_A_B_C(line_1, line_2):
        x0, y0 = line_1
        x1, y1 = line_2
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    (a0, b0, c0) = get_A_B_C(line_1_start, line_1_end)
    (a1, b1, c1) = get_A_B_C(line_2_start, line_2_end)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D

    # line1:A1,A2
    # line2:B1,B2
    # test if intersect
    B1B2 = calc_move(line_2_start, line_2_end)
    B1A1 = calc_move(line_2_start, line_1_start)
    B1A2 = calc_move(line_2_start, line_1_end)
    c_1 = cross_product(B1B2, B1A1)
    c_2 = cross_product(B1B2, B1A2)
    pass_1_flag = c_2 * c_1 <= 0
    if not pass_1_flag:
        return None

    A1A2 = calc_move(line_1_start, line_1_end)
    A1B1 = calc_move(line_1_start, line_2_start)
    A1B2 = calc_move(line_1_start, line_2_end)
    c_3 = cross_product(A1A2, A1B1)
    c_4 = cross_product(A1A2, A1B2)
    pass_2_flag = c_3 * c_4 <= 0
    if not pass_2_flag:
        return None

    return x, y


def calc_cos_theta(move_1, move_2, is_360=True):
    """
    计算两个向量间cosθ
    :param move_1: 移动向量1
    :param move_2: 移动向量2
    :param is_360: 逆时针360°
    :return: cosθ
    """
    # to avoid move_1 == move_2
    if is_360 and move_1 == move_2:
        return -3
    a_norm = (move_1[0] ** 2 + move_1[1] ** 2) ** 0.5 + 1e-5
    b_norm = (move_2[0] ** 2 + move_2[1] ** 2) ** 0.5 + 1e-5
    cos_theta = (move_1[0] * move_2[0] + move_1[1] * move_2[1]) / (
            a_norm * b_norm)
    # calc_counterclockwise or not
    if is_360 and calc_counterclockwise(move_1, move_2):
        cos_theta = -2 - cos_theta
    return cos_theta


def rotate_right_hand(vector, theta):
    x, y = vector
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    return x1, y1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def trans_speed():
    return np.random.rand() * 450 + 50
