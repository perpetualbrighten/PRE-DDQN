"""
构建PER_DDQN网络
"""

import os
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np

from common import STATE_DIM, ACTION_DIM
from tools import ucb_Q
from numpy import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Hyper Parameters for PER-DDQN
GAMMA = 0.99             # discount factor for target Q
GREEDY_EPSILON = 0.01
REPLAY_SIZE = 3200     # experience replay buffer size
BATCH_SIZE = 32        # size of mini-batch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network

FEATURE_MAX = np.array(
    [1500, 8000, 800, 400],
    dtype=np.float32)
FEATURE_MIN = np.array(
    [0, 0, 0, 0],
    dtype=np.float32)


# 利用SumTree构建PER-DDQN
class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


# 经验回放池
class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), \
                                     np.empty((n,), dtype=object), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class tf_model_perddqn:

    def __init__(self,
                 learning_rate=1e-3,
                 temperature=1e-3,
                 is_trainable=True):

        self.target_replace_op = None
        self.learning_rate = learning_rate
        self.is_trainable = is_trainable
        self.temperature = temperature
        self.model_path_head = r'./models/'
        self.model_path_rear = r'/test_model/test'
        self.cnt = 0
        self.ucb_cnt = np.zeros(50)
        self.graph = tf.Graph()

        self.replay_total = 0
        # init some parameters
        self.time_step = 0
        self.epsilon = GREEDY_EPSILON
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.memory = Memory(capacity=REPLAY_SIZE)

        with self.graph.as_default():
            self.create_Q_network()
            self.create_training_method()

            # Init session
            self.saver = tf.train.Saver()
            self.session = tf.InteractiveSession()
            self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        self.ISWeights = tf.placeholder(tf.float32, [None, 1])
        # network weights
        self.state_input = (self.state_input - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN)
        self.state_input = tf.clip_by_value(self.state_input, 0, 1)
        with tf.variable_scope('current_net'):

            W1 = self.weight_variable([self.state_dim, 16])
            b1 = self.bias_variable([16])
            W2 = self.weight_variable([16, 64])
            b2 = self.bias_variable([64])
            W3 = self.weight_variable([64, self.action_dim])
            b3 = self.bias_variable([self.action_dim])

            # hidden layers
            h_layer1 = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W2) + b2)

            # Q Value layer
            self.Q_value = tf.matmul(h_layer2, W3) + b3

        with tf.variable_scope('target_net'):
            W1t = self.weight_variable([self.state_dim, 16])
            b1t = self.bias_variable([16])
            W2t = self.weight_variable([16, 64])
            b2t = self.bias_variable([64])
            W3t = self.weight_variable([64, self.action_dim])
            b3t = self.bias_variable([self.action_dim])

            # hidden layers
            h_layer1t = tf.nn.relu(tf.matmul(self.state_input, W1t) + b1t)
            h_layer2t = tf.nn.relu(tf.matmul(h_layer1t, W2t) + b2t)

            # Q Value layer
            self.target_Q_value = tf.matmul(h_layer2t, W3) + b3

        with tf.variable_scope('predict_Q'):
            self.output_layer = self.target_Q_value
            self.probability = tf.nn.softmax(
                tf.reshape(self.output_layer, [-1]) / self.temperature)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_training_method(self):
        self.y_input = tf.placeholder("float", [None])
        Q_action = self.Q_value
        Q_action = tf.reshape(Q_action, [-1])
        self.cost = tf.reduce_mean(self.ISWeights * (tf.square(self.y_input - Q_action)))
        self.abs_errors = tf.abs(self.y_input - Q_action)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def store_transition(self, ss, r, done, s_,):
        transition = np.hstack((ss, r, done, s_,))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def perceive(self, sub_state, reward, done, next_state):
        self.store_transition(sub_state, reward, done, next_state)
        self.replay_total += 1
        if self.replay_total > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random mini-batch from replay memory
        tree_idx, minibatch, ISWeights = self.memory.sample(BATCH_SIZE)

        sub_state_batch = [data[0:self.state_dim] for data in minibatch]
        reward_batch = [data[self.state_dim] for data in minibatch]
        next_state_batch = [data[self.state_dim+2:] for data in minibatch]

        # Step 2: calculate y

        target_Q_batch = []
        for next_state in next_state_batch:
            next_state = next_state.reshape(-1, self.state_dim)
            current_Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state})
            current_Q_batch = current_Q_batch.reshape(-1)
            max_action_next = np.argmax(current_Q_batch)
            target_Q = self.target_Q_value.eval(
                feed_dict={self.state_input: next_state[max_action_next][np.newaxis, :]})
            target_Q_batch.append(target_Q[0, 0])

        y_batch = []
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][self.state_dim+1]
            if done:
                y_batch.append(reward_batch[i])
            else:
                target_Q_value = target_Q_batch[i]
                y_batch.append(reward_batch[i] + GAMMA * target_Q_value)

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.state_input: sub_state_batch,
            self.ISWeights: ISWeights
        })
        _, abs_errors, _ = self.session.run([self.optimizer, self.abs_errors, self.cost], feed_dict={
            self.y_input: y_batch,
            self.state_input: sub_state_batch,
            self.ISWeights: ISWeights
        })
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

    def update_target_q_network(self, episode):
        # update target Q netowrk
        if episode % REPLACE_TARGET_FREQ == 0:
            self.session.run(self.target_replace_op)
            # print('episode '+str(episode) +', target Q network params replaced!')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    # softmax最终输出层

    def save_model(self, model_name):
        model_path = self.model_path_head + model_name + self.model_path_rear
        return self.saver.save(self.session, model_path)

    def load_model(self, model_name):
        model_path = self.model_path_head + model_name + self.model_path_rear
        if not os.path.exists(model_path + '.index'):
            return
        tf.reset_default_graph()
        return self.saver.restore(self.session, model_path)

    def learn(self):
        # 计算每次迭代的td_error和损失
        """
        indices = np.random.choice(self.mem_size, size=self.batch_size)  # 随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]  # 从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  # 从bt获得数据a
        br = bt[:, self.s_dim + self.a_dim:-self.s_dim]  # 从bt获得数据r
        bs_ = bt[:, -self.s_dim:]  # 从bt获得数据s_
        with tf.GradientTape() as tape:
            # batch_size*2
            a_ = self.target(bs_)
            # batch_size*4
            q_ = self.target([bs_, a_])
            y = br + self.gamma * q_
            q = self.target([bs, ba])
            td_error = tf.losses.mean_squared_error(tf.reshape(y, [-1]), tf.reshape(q, [-1]))
        c_grads = tape.gradient(td_error, self.target.trainable_weights)
        self.target.apply_gradients(zip(c_grads, self.target.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        self.ema_update()
        self.soft_replace()
        """
        pass

    def predict(self, xs):
        """
        预测
        :param xs:特征
        :return:预测Q值
        """
        y_predict = self.session.run(self.output_layer, feed_dict={self.state_input: xs})
        return y_predict.reshape(-1)

    def get_probability(self, xs):
        """
        得到概率
        :param xs:特征
        :return:softmax概率概率
        """

        return self.session.run(self.probability, feed_dict={self.state_input: xs})

    def get_decision(self, xs):
        """
        做出决策
        :param xs: 特征
        :return:node(softmax策略后选择的转发节点) is_max_Q(是否为最大Q的点)
        """
        prob = self.get_probability(xs)
        prob_list = list(enumerate(prob))
        prob_list = sorted(prob_list, key=lambda x: x[1])
        # 累加和
        accu_threshold = 0
        rand = random.rand()
        selected_next = prob_list[-1][0]
        is_select_max_Q = True
        ucb_max = 0

        # 在训练阶段使用改进UCB策略或者Greedy-epsilon选择最优节点
        if self.is_trainable:
            i = 0
            pos = 0
            sum_cnt = sum(self.ucb_cnt)
            for item in prob_list[::-1]:
                Q = item[1] + ucb_Q(self.ucb_cnt[i], sum_cnt)
                if ucb_max < Q:
                    selected_next = item[0]
                    pos = i
                    ucb_max = Q
                i += 1
                self.ucb_cnt[pos] += 1

        # 在测试阶段使用softmax或者最大值策略
        else:
            rand = random.rand()

            for item in prob_list:
                accu_threshold += item[1]
                if rand <= accu_threshold:
                    selected_next = item[0]
                    break

        if len(prob_list) > 1 and selected_next != prob_list[-1][0]:
            is_select_max_Q = False
        return selected_next, is_select_max_Q

    def insert_experience(self, sub_state, reward, done, next_state):
        """
        插入经验池
        :return:None
        """

        # path = r".\analysis\a.csv"
        # if not os.path.exists(path):
        #
        #     col.to_csv(path, index=False)
        #
        # t = next_state.reshape(-1, self.state_dim)
        # res = np.vstack((sub_state, t))
        # p = pd.DataFrame(res)
        # p.to_csv(path, mode='a', header=False, index=False)

        if self.is_trainable:
            self.perceive(sub_state, reward, done, next_state)
            self.cnt += 1
            self.update_target_q_network(self.cnt)






