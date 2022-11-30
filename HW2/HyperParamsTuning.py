import numpy as np
import pandas as pd
from utilis import *
from BPR import *
from NegativeSampling import *
from SGDOptimizer import *
from itertools import product


class HPT:
    def __init__(self, hpt_options_dict, sample_num=100, sample_method='P'):
        self.hpt_options = hpt_options_dict
        self.sample_num = sample_num
        self.sample_method = sample_method
        self.best_hpt_set = {}
        self.best_valid_error = -np.Inf
        self.best_mpr = 1
        self.best_hit_rate = 0
        self.best_epoch_num = 0
        self.best_valid_count = 0
        self.hpt_checked_options = []
        self.hpt_checked_valid_error = []
        self.hpt_checked_mpr = []
        self.hpt_checked_hit_rate = []
        self.hpt_checked_epoch_num = []
        self.hpt_checked_valid_count = []
        self.hp_seed = np.random.RandomState(123)

    def run_one_set(self, hp_set, read_directory):
        # Run model with hp set till conv
        train, items_index, users_index, \
        bpr_model, all_epochs_data_lst, valid_data = pre_process(hp_set, read_directory=read_directory,
                                                                 sample_method=self.sample_method, write_num=2)
        sgd = SGDOptimizer(mf_model=bpr_model, max_epoch_num=len(all_epochs_data_lst), bias_epoch_num=2, epsilon=-np.Inf,
                           allowed_dec=3)
        sgd.train_model(all_epochs_data_lst, valid_data, users_index, items_index, model_type='bias')

        # Calc metric for the output model
        valid_err = sgd.best_v_err
        valid_count_right = sgd.best_count_v
        hit_rate, mpr = calc_metric(valid_data, sgd.model, [1, 10, 50, 1000], users_index, items_index)
        self.hpt_checked_options.append(hp_set)
        self.hpt_checked_valid_error.append(valid_err)
        self.hpt_checked_mpr.append(mpr)
        self.hpt_checked_hit_rate.append(hit_rate)
        self.hpt_checked_epoch_num.append(sgd.best_epoch)
        self.hpt_checked_valid_count.append(valid_count_right)

    def hp_tune(self, exp_num):
        all_hp_set_lst = list(product(*self.hpt_options.values()))
        inx = self.hp_seed.choice(len(all_hp_set_lst), self.sample_num)
        first = True
        for set_inx in tqdm(inx):
            hp_vals_set = convert_tuple_to_dict(all_hp_set_lst[set_inx])
            if first:
                self.run_one_set(hp_vals_set, read_directory=False)
                first = False
            else:
                self.run_one_set(hp_vals_set, read_directory=True)

            if self.hpt_checked_mpr[-1] < self.best_mpr:  # TODO: Choose the desired metric
                self.best_hpt_set = hp_vals_set
                self.best_valid_error = self.hpt_checked_valid_error[-1]
                self.best_mpr = self.hpt_checked_mpr[-1]
                self.best_hit_rate = self.hpt_checked_hit_rate[-1].copy()
                self.best_epoch_num = self.hpt_checked_epoch_num[-1]
                self.best_valid_count = self.hpt_checked_valid_count[-1]
                print('Found New HP Set - MRP = ', self.best_mpr, ', Hit Rate@k = ', self.best_hit_rate,
                      ', Valid Err = ', self.best_valid_error, ', Epoch = ', self.best_epoch_num,
                      ', valid_correct_count = ', self.best_valid_count)
        self.write_report_csv(exp_num)
        return

    def write_report_csv(self, exp_num):
        result_df = pd.DataFrame(self.hpt_checked_options, columns=['alpha_u', 'alpha_i', 'alpha_b', 'dim', 'lr', 'sigma'])
        result_df['valid_error'] = self.hpt_checked_valid_error
        result_df['valid_mpr'] = self.hpt_checked_mpr
        result_df['valid_hit_rate'] = self.hpt_checked_hit_rate
        result_df['epoch'] = self.hpt_checked_epoch_num
        result_df['valid_correct_count'] = self.hpt_checked_valid_count
        result_df.to_csv('HPTResults\\result_exp_%d_%s.csv' % (exp_num, self.sample_method))
        return


if __name__ == '__main__':
    hpt_options_dict = {'alpha_u': [0.001, 0.01, 0.05, 0.1],
                        'alpha_i': [0.001, 0.01, 0.05, 0.1],
                        'alpha_b': [0.001, 0.01, 0.05, 0.1],
                        'dim': [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 100, 150, 200],
                        'lr': [0.001, 0.01, 0.05, 0.1],
                        'sigma': [0.0001, 0.001, 0.01, 0.1]}
    for method in ['P', 'U']:
        hpt_object = HPT(hpt_options_dict, sample_num=100, sample_method=method)
        hpt_object.hp_tune(1)


