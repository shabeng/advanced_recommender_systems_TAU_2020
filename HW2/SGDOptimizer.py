import numpy as np
import pandas as pd
from utilis import *
from BPR import *
from NegativeSampling import *
from PreProcess import *
import pickle


class SGDOptimizer:
    def __init__(self, mf_model, max_epoch_num, bias_epoch_num, epsilon, allowed_dec):
        self.max_epoch_num = max_epoch_num
        self.bias_epoch_num = bias_epoch_num
        self.eps = epsilon
        self.allowed_dec_on_valid = allowed_dec
        self.model = mf_model
        self.curr_t_err = - np.Inf
        self.prev_t_err = - np.Inf
        self.curr_v_err = - np.Inf
        self.prev_v_err = - np.Inf
        self.best_v_err = - np.Inf
        self.curr_dec_count = 0
        self.convergence = False
        self.right_count_t = 0
        self.right_count_v = 0
        self.best_epoch = 0
        self.best_count_v = 0
        self.curr_epoch = 1

    def run_epoch(self, train_data, user_index, item_index, model_type='bias'):
        # np.seterr(all='raise')
        self.curr_t_err = 0
        self.right_count_t = 0
        for m, i, j in train_data:
            m, i, j = user_index[m], item_index[i], item_index[j]
            new_bi = self.model.update_params('b_i', i, j, m, model_type=model_type)
            new_bj = self.model.update_params('b_j', i, j, m, model_type=model_type)
            s_m_i_j_old = self.model.b_i[i] - self.model.b_i[j]  # TODO remove later

            if model_type == 'full':
                new_um = self.model.update_params('u_m', i, j, m, model_type=model_type)
                new_vi = self.model.update_params('v_i', i, j, m, model_type=model_type)
                new_vj = self.model.update_params('v_j', i, j, m, model_type=model_type)
                s_m_i_j_old += np.dot(self.model.u_m[:, m], self.model.v_i[:, i]-self.model.v_i[:, j])  # TODO remove later

            # update the model with the new parameters
            self.model.b_i[i] = new_bi
            self.model.b_i[j] = new_bj
            s_m_i_j = new_bi - new_bj

            if model_type == 'full':
                self.model.u_m[:, m] = new_um
                self.model.v_i[:, i] = new_vi
                self.model.v_i[:, j] = new_vj
                s_m_i_j += np.dot(new_um, new_vi-new_vj)
            self.curr_t_err += np.log(sigmoid(s_m_i_j))
            if s_m_i_j > 0:
                self.right_count_t += 1
        print('Train Log Error = ', self.curr_t_err, ', Train Reg Error = ', self.calc_reg(model_type=model_type))
        self.curr_t_err -= self.calc_reg(model_type=model_type)
        return

    def calc_error(self, valid_data, user_index, item_index, model_type='bias'):
        s_m_i_j = 0
        self.curr_v_err = 0
        self.right_count_v = 0
        for m, i, j in valid_data:
            m, i, j = user_index[m], item_index[i], item_index[j]
            i_j_bias_difference = self.model.b_i[i] - self.model.b_i[j]
            if model_type == 'full':
                s_m_i_j = np.dot(self.model.u_m[:, m], (self.model.v_i[:, i] - self.model.v_i[:, j]))
            self.curr_v_err += np.log(sigmoid(s_m_i_j + i_j_bias_difference))
            if s_m_i_j > 0:
                self.right_count_v += 1
        reg_error = self.calc_reg(model_type=model_type)
        self.curr_v_err -= reg_error
        return

    def calc_reg(self, model_type='bias'):
        b_i_reg = np.sum(self.model.b_i ** 2) * (self.model.hyper_set['alpha_b'] / 2)
        if model_type == 'full':
            u_m_reg = np.sum(np.linalg.norm(self.model.u_m, axis=0)) * (self.model.hyper_set['alpha_u'] / 2)
            v_i_reg = np.sum(np.linalg.norm(self.model.v_i, axis=0)) * (self.model.hyper_set['alpha_i'] / 2)
        else:
            u_m_reg = 0
            v_i_reg = 0
        return b_i_reg + u_m_reg + v_i_reg

    def check_convergence(self, validation_data, curr_epoch, user_index, item_index, model_type='bias', is_valid=True):
        print('Current Train Objective Function =', self.curr_t_err, ', Right Count Train = ', self.right_count_t)
        delta_train = self.curr_t_err - self.prev_t_err  # expect to be positive if the model improves

        if is_valid:
            # Validation increase criterion
            self.calc_error(validation_data, user_index, item_index, model_type=model_type)
            print('Current Validation Objective Function =', self.curr_v_err, ', Right Count Validation = ', self.right_count_v)
            if model_type == 'full':
                hit_rate_k, mpr = calc_metric(validation_data, self.model, [1, 10, 50, 1000], user_index, item_index)
                print('Hit Rate @ 1, 10, 50, 1000 = ', hit_rate_k, ', MPR = ', mpr)

            delta_valid = self.curr_v_err - self.prev_v_err
            if delta_valid < 0:
                self.curr_dec_count += 1
                if self.curr_dec_count > self.allowed_dec_on_valid:
                    self.convergence = True
                    print('~~~ Validation increase criterion IN ~~~')
            if self.curr_v_err > self.best_v_err:
                self.model.u_m_best = self.model.u_m.copy()
                self.model.v_i_best = self.model.v_i.copy()
                self.model.b_i_best = self.model.b_i.copy()
                self.best_epoch = self.curr_epoch
                self.best_count_v = self.right_count_v
                self.best_v_err = self.curr_v_err

        # Max epoch num criterion
        if curr_epoch >= self.max_epoch_num:
            self.convergence = True
            self.model.u_m_best = self.model.u_m.copy()
            self.model.v_i_best = self.model.v_i.copy()
            self.model.b_i_best = self.model.b_i.copy()
            self.best_epoch = self.curr_epoch
            self.best_count_v = self.right_count_v
            self.best_v_err = self.curr_v_err
            # added now - update the model on the last epoch so the best model is not the initialized one.
            print('~~~ Max epoch num criterion IN ~~~')

        # Train epsilon change criterion - check only after finishing the second epoch with the latent vectors
        if self.curr_epoch > self.bias_epoch_num + 1:
            if delta_train < self.eps:
                self.convergence = True
                print('~~~ Train epsilon change criterion IN ~~~')

        self.prev_t_err = self.curr_t_err
        self.prev_v_err = self.curr_v_err
        return

    def train_model(self, train_data_list, validation_data, user_index, item_index, model_type='bias', is_valid=True):
        while not self.convergence:
            if self.curr_epoch > self.bias_epoch_num:
                model_type = 'full'
            print('Epoch: ', self.curr_epoch)
            self.run_epoch(train_data_list[self.curr_epoch-1], user_index, item_index, model_type=model_type)
            self.check_convergence(validation_data, self.curr_epoch, user_index, item_index, model_type=model_type,
                                   is_valid=is_valid)
            self.curr_epoch += 1
            if self.curr_epoch % 5 == 0:
                self.model.hyper_set['lr'] = self.model.hyper_set['lr']*0.9
        return


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~ Model with Best HP Set found Results for Report ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # hyper_set = {'alpha_u': 10, 'alpha_i': 10, 'alpha_b': 10, 'dim': 7, 'lr': 0.001}
    # hyper_set = {'alpha_u': 0.01, 'alpha_i': 0.01, 'alpha_b': 0.1, 'dim': 60, 'lr': 0.05, 'sigma': 0.1}
    # t_data, items_index, users_index, bpr_model, all_epochs_data_lst, valid_data = pre_process(hyper_set, epoch_num=50,
    #                                                                                            read_directory=True,
    #                                                                                            sample_method='P',
    #                                                                                            write_num=2)
    # sgd = SGDOptimizer(mf_model=bpr_model, max_epoch_num=len(all_epochs_data_lst), bias_epoch_num=2, epsilon=-np.inf,
    #                    allowed_dec=3)
    # sgd.train_model(all_epochs_data_lst, valid_data, users_index, items_index, model_type='bias')  # is_valif=False, valid_data=None
    #
    # calc_metric(valid_data, sgd.model, [1,10,50], users_index, items_index)

    # hyper_set = {'alpha_u': 0.01, 'alpha_i': 0.01, 'alpha_b': 0.1, 'dim': 60, 'lr': 0.05, 'sigma': 0.1}
    # t_data, items_index, users_index, bpr_model, all_epochs_data_lst, valid_data = pre_process(hyper_set, epoch_num=50,
    #                                                                                            read_directory=True,
    #                                                                                            sample_method='U',
    #                                                                                            write_num=2)
    # sgd = SGDOptimizer(mf_model=bpr_model, max_epoch_num=len(all_epochs_data_lst), bias_epoch_num=2, epsilon=-np.inf,
    #                    allowed_dec=3)
    # sgd.train_model(all_epochs_data_lst, valid_data, users_index, items_index, model_type='bias')
    #
    # calc_metric(valid_data, sgd.model, [1, 10, 50], users_index, items_index)

    # ~~~~~~~~~~~~~~~~~~~~~ Final Model for Test Result - No Validation Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    method = ['P', 'U']
    epoch_num = [26, 29]
    read_d = [False, False]
    model_object_dict = {}
    test_df_dict = {}
    for i in range(2):
        print('~~~~~~~~~~~~~ Starting ', method[i])
        hyper_set = {'alpha_u': 0.01, 'alpha_i': 0.01, 'alpha_b': 0.1, 'dim': 60, 'lr': 0.05, 'sigma': 0.1}
        print(method[i], ': Data Preparation')
        t_data, items_index, users_index, bpr_model, all_epochs_data_lst, valid_data = pre_process(hyper_set,
                                                                                                   epoch_num=epoch_num[i],
                                                                                                   read_directory=read_d[i],
                                                                                                   sample_method=method[i],
                                                                                                   write_num=1,
                                                                                                   if_test=True #added now
                                                                                                   )
        print('out')
        print(method[i], ': Fit Model')
        sgd = SGDOptimizer(mf_model=bpr_model, max_epoch_num=len(all_epochs_data_lst), bias_epoch_num=2, epsilon=-np.inf,
                           allowed_dec=3)
        sgd.train_model(all_epochs_data_lst, valid_data, users_index, items_index, model_type='bias', is_valid=False)
        model_object_dict[method[i]] = sgd.model
        print(method[i], ': Test Calc')
        if method[i] == 'P':
            details = 'popularity'
        else:
            details = 'random'
        test_df = PostProcess_TestResult(sgd.model, users_index, items_index, details=details)
        test_df_dict[method[i]] = test_df













