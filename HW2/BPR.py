import numpy as np
import pandas as pd
from utilis import *


class BPRModel:
    def __init__(self, num_items, num_users, hyper_set):
        self.u_m = None
        self.v_i = None
        self.b_i = None
        self.u_m_best = None
        self.v_i_best = None
        self.b_i_best = None
        self.hyper_set = hyper_set   # alpha_u for user vector, alpha_i for item vector, alpha_b for bias, dim, lr
        self.M = num_users
        self.N = num_items
        self.bpr_seed = np.random.RandomState(273)
        self.initialize_parameters()

    def initialize_parameters(self):
        # self.u_m = self.bpr_seed.normal(0, self.hyper_set['alpha_u']**(-1), (int(self.hyper_set['dim']), self.M))
        # self.v_i = self.bpr_seed.normal(0, self.hyper_set['alpha_i']**(-1), (int(self.hyper_set['dim']), self.N))
        # self.b_i = self.bpr_seed.normal(0, self.hyper_set['alpha_b']**(-1), int(self.N))
        self.u_m = self.bpr_seed.normal(0, self.hyper_set['sigma'], (int(self.hyper_set['dim']), self.M))  # TODO: change from alpha?
        self.v_i = self.bpr_seed.normal(0, self.hyper_set['sigma'], (int(self.hyper_set['dim']), self.N))
        self.b_i = self.bpr_seed.normal(0, self.hyper_set['sigma'], int(self.N))
        self.u_m_best = self.u_m.copy()
        self.v_i_best = self.v_i.copy()
        self.b_i_best = self.b_i.copy()

    def update_params(self, param_to_update_str, i, j, m, model_type='bias'):  # m-> user, i-> pos, j-> neg
        f = 1 if model_type == 'full' else 0
        u_m = self.u_m[:, m]
        v_i = self.v_i[:, i]
        v_j = self.v_i[:, j]
        b_i = self.b_i[i]
        b_j = self.b_i[j]
        s_m_i = np.dot(u_m, v_i)*f + b_i
        s_m_j = np.dot(u_m, v_j)*f + b_j
        if param_to_update_str == 'u_m':
            u_m_new = u_m + self.hyper_set['lr'] * (
                    (1 - sigmoid(s_m_i - s_m_j)) * (v_i - v_j) - self.hyper_set['alpha_u'] * u_m)
            return u_m_new
        elif param_to_update_str == 'v_i':
            v_i_new = v_i + self.hyper_set['lr'] * (
                    (1 - sigmoid(s_m_i - s_m_j)) * u_m - self.hyper_set['alpha_i'] * v_i)
            return v_i_new
        elif param_to_update_str == 'v_j':
            v_j_new = v_j + self.hyper_set['lr'] * (
                    (1 - sigmoid(s_m_i - s_m_j)) * (-u_m) - self.hyper_set['alpha_i'] * v_j)
            return v_j_new
        elif param_to_update_str == 'b_i':
            b_i_new = b_i + self.hyper_set['lr'] * (
                    (1 - sigmoid(s_m_i - s_m_j)) - self.hyper_set['alpha_b'] * b_i)
            return b_i_new
        elif param_to_update_str == 'b_j':
            b_j_new = b_j + self.hyper_set['lr'] * (
                    (1 - sigmoid(s_m_i - s_m_j))*(-1) - self.hyper_set['alpha_b'] * b_j)
            return b_j_new

    def predict_triplet(self, user_id, item_1, item_2):
        """

        :param user_id: model index
        :param item_1: model index
        :param item_2: model index
        :return: 0 if first item was the item that was liked OR 1 if second item was liked by the user
        """
        u_m = self.u_m_best[:, user_id]
        v_1 = self.v_i_best[:, item_1]
        v_2 = self.v_i_best[:, item_2]
        b_1 = self.b_i_best[item_1]
        b_2 = self.b_i_best[item_2]
        s_m_1 = np.dot(u_m, v_1) + b_1
        s_m_2 = np.dot(u_m, v_2) + b_2
        if s_m_1 > s_m_2:
            return 0
        elif s_m_2 > s_m_1:
            return 1
        else:
            np.random.choice(2)