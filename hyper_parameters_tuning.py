from utilis import *
from SGD_optimization import *
from ALS_optimization import *
import numpy as np
from itertools import product
import time


def hyper_tune_test(train_data, validation_data, item_index, func_str, hyper_params_, n_samples=100, D_u=None, D_i=None):
    """
    Implement the Randomized Grid Search using 100 samples from the grid search space as defult.
    :param train_data: train data set array
    :param validation_data: validation data set array
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param func_str: the optimization method to be used
    :param hyper_params_: dictionary of the grid search
    :param n_samples: number od samples to be taken from the grid search space
    :param D_u: dictionary of arrays for each user with all its rating, for the ALS.
    :param D_i: dictionary of arrays for each item with all its rating, for the ALS.
    :return:
    """
    np.random.RandomState(123)
    inx = np.random.choice(len(hyper_params_), n_samples)

    sample_RMSE_validation = []
    sample_hp_vals =[]
    best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim = hyper_params_[0]
    best_rmse = 100

    for val_set_inx in inx:
        val_set = hyper_params_[val_set_inx]
        print(val_set)
        sample_hp_vals.append(val_set)
        start = time.time()
        mu, b_i_last, b_u_last, X_u_last, Y_i_last, last_train_error, last_validation_error, last_mae, last_r_squre, \
        t_error, v_error, last_rmse = func_coordinator(func_str, train_data, validation_data, item_index, val_set, D_u, D_i)

        sample_RMSE_validation.append(last_rmse)

        # Check whether the sampled hyper parameter set resulted in better rmse on the validation set
        if last_rmse < best_rmse:
            best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim = val_set
            best_rmse = last_rmse
            print('-'*80)
            print('~'*80)
            print('New Best RMSE Found!!!' + str(best_rmse))
            print('~' * 80)
            print('-' * 80)

        end = time.time()
        print('--- TOTAL TIME = ' + str(end-start))

    result_df = pd.DataFrame(sample_hp_vals, columns=['lr', 'reg_bu', 'reg_bi', 'reg_x', 'reg_y', 'sigma', 'dim'])
    result_df['all_rmse'] = sample_RMSE_validation
    return best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim, best_rmse, result_df


def func_coordinator(func_str, train_data, validation_data, item_index, hyper_params, D_u=None, D_i=None):
    """
    Call for the appropriate function for the hyper-parameter tuning process.
    :param func_str: the optimization method to be used
    :param train_data: train data set array
    :param validation_data: validation data set array
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param hyper_params: array of the hyper-parameters values.
    :param D_u: dictionary of arrays for each user with all its rating.
    :param D_i: dictionary of arrays for each item with all its rating.
    :return: the model that was found and its measures values.
    """
    if func_str == 'SGD_full':
        print('~'*80)
        print('SGD Bias Model:')
        mu, b_i_best, b_u_best, best_train_error, best_validation_error, best_rmse, best_mae, best_r_squre =\
            run_bias_model_SGD(train_data, validation_data, hyper_params, item_index)
        print('~'*80)
        print('SGD Full Model:')
        return run_full_model_SGD(train_data, validation_data, hyper_params, b_i_best, b_u_best, mu, item_index)

    elif func_str == 'ALS_full':
        print('~'*80)
        print('ALS Bias Model:')
        mu, b_i_best, b_u_best, best_train_error, best_validation_error, best_mae, best_r_squre, best_rmse = \
            run_bias_model_ALS(train_data, validation_data, hyper_params, D_u, D_i, item_index, epoch_num=5)
        print('~'*80)
        print('ALS Full Model:')
        return run_full_model_ALS(train_data, validation_data, hyper_params, D_u, D_i, b_i_best, b_u_best, mu,
                                  item_index)


if __name__ == '__main__':
    print('This process going to run for several hours. You are advised not to run the file.')
    print('The result with the best found hyper parameter set are hard-coded to the main file.')
    train_data_df = read_data_to_df('Data\\Train.csv')
    # Change the index from the data and for the model use:
    item_index = item_to_index(train_data_df)
    t_data = train_data_df.values
    v_data = read_data_to_df('Data\\Validation.csv').values

    # SGD HPT
    hyper_params = {'lr': [0.02, 0.01], 'reg_bu': [0.001, 0.01, 0.1], 'reg_bi': [0.001, 0.01, 0.1],
                    'reg_x': [0.001, 0.01, 0.1], 'reg_y': [0.001, 0.01, 0.1], 'sigma': [0.0001, 0.001, 0.01, 0.1],
                    'dim': [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 100, 150, 200]}
    all_comb_lst = list(product(*hyper_params.values()))

    sgd_start = time.time()
    best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim, best_rmse, result_df_sgd = \
        hyper_tune_test(t_data, v_data, item_index, 'SGD_full', all_comb_lst, D_u=None, D_i=None)
    sgd_end = time.time()
    print('-' * 80)
    print('~' * 80)
    print('#' * 80)
    print('- SGD Finals: -')
    print('Total Time = ' + str((sgd_end - sgd_start) / 60) + ' minutes.')
    print('Best RMSE = ' + str(best_rmse))
    print('#' * 80)
    print('~' * 80)
    print('-' * 80)
    # Report:
    best_line = [best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim, best_rmse]
    best_line_df = pd.DataFrame(best_line).T
    best_line_df.columns = ['lr', 'reg_bu', 'reg_bi', 'reg_x', 'reg_y', 'sigma', 'dim', 'all_rmse']
    result_df_sgd = result_df_sgd.append(best_line_df, ignore_index=True)
    result_df_sgd.to_csv('result_df_sgd.csv')

    # ALS HPT
    hyper_params = {'lr': [1], 'reg_bu': [0.001, 0.01, 0.1], 'reg_bi': [0.001, 0.01, 0.1],
                    'reg_x': [0.001, 0.01], 'reg_y': [0.001, 0.01], 'sigma': [0.0001, 0.001, 0.01, 0.1],
                    'dim': [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 60, 100]}
    all_comb_lst = list(product(*hyper_params.values()))

    print('Initiate D_u and D_i dictionaries')
    D_u, D_i = data_prep_ALS(train_data_df, item_index)
    als_start = time.time()
    best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim, best_rmse, result_df_als = \
        hyper_tune_test(t_data, v_data, item_index, 'ALS_full', all_comb_lst, D_u=D_u, D_i=D_i)
    als_end = time.time()
    print('-'*80)
    print('~' * 80)
    print('#' * 80)
    print('- ALS Finals: -')
    print('Total Time = ' + str((als_end-als_start)/60) + ' minutes.')
    print('Best RMSE = ' + str(best_rmse))
    print('#' * 80)
    print('~' * 80)
    print('-'*80)
    # Report:
    best_line = [best_lr, best_reg_bu, best_reg_bi, best_reg_x, best_reg_y, best_sigma, best_dim, best_rmse]
    best_line_df = pd.DataFrame(best_line).T
    best_line_df.columns = ['lr', 'reg_bu', 'reg_bi', 'reg_x', 'reg_y', 'sigma', 'dim', 'all_rmse']
    result_df_als = result_df_als.append(best_line_df, ignore_index=True)
    result_df_als.to_csv('result_df_als.csv')
