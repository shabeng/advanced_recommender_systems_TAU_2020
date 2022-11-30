import numpy as np
import pandas as pd
from SGD_optimization import *
from utilis import *


def run_bias_model_ALS(train_data, validation_data, hyper_params, D_u, D_i, item_index, epoch_num=10):
    """
    Train the matrix Factorization using the ALS optimization method and the model bias
    :param train_data: train data set array
    :param validation_data: validation data set array
    :param hyper_params: array of the hyper-parameters values.
    :param D_u: dictionary of arrays for each user with all its rating.
    :param D_i: dictionary of arrays for each item with all its rating.
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param epoch_num: the number of epochs to run the train
    :return: the model parameters, some quality measures as the rmse, mae, objective function value and more.
    """
    hyper_params = create_hyper_params_dict(hyper_params)
    mu, b_i, b_u = calc_bias(train_data)
    current_train_error = calc_error(train_data, mu, b_i, b_u, item_index)[0] + \
                          calc_reg(b_i, b_u, hyper_params, model_type='bias')
    current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index)[0] + \
                               calc_reg(b_i, b_u, hyper_params, model_type='bias')
    print('train error: ', current_train_error)
    print('validation error: ', current_validation_error)

    # Best Model:
    b_i_best = b_i
    b_u_best = b_u
    best_train_error = current_train_error
    best_validation_error = current_validation_error
    best_rmse, best_mae, best_r_squre = \
        calc_evaluation_measures(validation_data, mu, b_i_best, b_u_best, item_index, model_type='bias')

    for epoch in range(epoch_num):
        for user in D_u.keys():
            D_u_user = D_u[user]  # array: item_index_id (correct), rating

            # Ratings of the items the user ranked:
            r_ui = D_u_user[:, 1]
            # Item biases of the item the user ranked:
            b_i_D_u = b_i[D_u_user[:, 0] - 1]
            sigma_D_u = np.sum(r_ui - mu - b_i_D_u)
            multiplier_u = 1/(D_u_user.shape[0] + hyper_params['reg_bu'])

            b_u_new = multiplier_u * sigma_D_u

            b_u[user - 1] = b_u_new

        for item in D_i.keys():
            D_i_item = D_i[item]  # array: user_id, rating

            # Ratings of the item by all the users who ranked it:
            r_ui = D_i_item[:, 1]
            # User biases of all the users that ranked the item:
            b_u_D_i = b_u[D_i_item[:, 0] - 1]
            sigma_D_i = np.sum(r_ui - mu - b_u_D_i)
            multiplier_i = 1/(D_i_item.shape[0] + hyper_params['reg_bi'])

            b_i_new = multiplier_i * sigma_D_i

            b_i[item - 1] = b_i_new

        prev_train_error = current_train_error
        prev_validation_error = current_validation_error

        current_train_error = calc_error(train_data, mu, b_i, b_u, item_index)[0] + \
                              calc_reg(b_i, b_u, hyper_params, model_type='bias')
        current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index)[0] +\
                                   calc_reg(b_i, b_u, hyper_params, model_type='bias')

        if current_validation_error < best_validation_error:
            b_i_best = np.copy(b_i)
            b_u_best = np.copy(b_u)
            best_train_error = np.copy(current_train_error)
            best_validation_error = np.copy(current_validation_error)
            best_rmse, best_mae, best_r_squre = \
                calc_evaluation_measures(validation_data, mu, b_i_best, b_u_best, item_index, model_type='bias')
            epoch += 1

    print('-' * 80)
    print('epoch %d:' % epoch)
    print('train error: ', current_train_error)
    print('validation error: ', current_validation_error)
    RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='bias')
    print('RMSE: ', RMSE)
    print('MAE: ', MAE)
    print('R_square: ', R_square)
    return mu, b_i_best, b_u_best, best_train_error, best_validation_error, best_mae, best_r_squre, best_rmse


def run_full_model_ALS(train_data, validation_data, hyper_params, D_u, D_i, b_i, b_u, mu, item_index):
    """
    Train the matrix Factorization using the ALS optimization method and the full model
    :param train_data: train data set array
    :param validation_data: validation data set array
    :param hyper_params: array of the hyper-parameters values.
    :param D_u: dictionary of arrays for each user with all its rating.
    :param D_i: dictionary of arrays for each item with all its rating.
    :param b_i: the best item biases found on the bias model
    :param b_u: the best user biases found on the bias model
    :param mu: the mean rationg of the training data
    :param item_index: dictionary mapping from the data index to a continuous index.
    :return: the model parameters, some quality measures as the rmse, mae, objective function value and more.
    """
    hyper_params = create_hyper_params_dict(hyper_params)
    np.random.RandomState(146)
    X_u, Y_i = initiate_vectors(hyper_params['dim'], hyper_params['sigma'], b_u.shape[0], b_i.shape[0])
    current_train_error = calc_error(train_data, mu, b_i, b_u, item_index, model_type='full', X_u=X_u, Y_i=Y_i)[0] +\
                          calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)
    current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index,  model_type='full', X_u=X_u, Y_i=Y_i)[0] +\
                               calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)
    RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='full',
                                                   X_u=X_u, Y_i=Y_i)
    print('train error: ', current_train_error)
    print('validation error: ', current_validation_error)
    print('RMSE: ' + str(RMSE))
    allowed_inc = 0
    epoch = 1

    # Best Model:
    X_u_best = X_u
    Y_i_best = Y_i
    b_i_best = b_i
    b_u_best = b_u
    best_train_error = current_train_error
    best_validation_error = current_validation_error
    best_rmse = RMSE
    best_mae = MAE
    best_r_squre = R_square

    t_error = []
    v_error = []
    t_error.append(current_train_error)
    v_error.append(current_validation_error)

    while allowed_inc <= 4:
        current_train_error = 0
        for user in D_u.keys():
            D_u_user = D_u[user]  # array: item_index_id (correct), rating
            # Ratings of the items the user ranked:
            r_ui = D_u_user[:, 1]
            # Item biases of the item the user ranked:
            b_i_D_u = b_i[D_u_user[:, 0] - 1]
            x_u_T = X_u[:, user-1].T  # 1*d
            y_i = Y_i[:, D_u_user[:, 0] - 1]  # d*|D_u|
            b_u_old = b_u[user-1]

            # b_u update:
            x_u_T_y_i = np.dot(x_u_T, y_i)  # 1*|D_u|
            sigma_D_u = np.sum(r_ui - mu - b_i_D_u - x_u_T_y_i)
            multiplier_u = 1 / (D_u_user.shape[0] + hyper_params['reg_bu'])  # V
            b_u_new = sigma_D_u*multiplier_u
            b_u[user-1] = b_u_new

            # x_u update:
            sigma_D_u = np.sum((r_ui - mu - b_i_D_u - b_u_old)*y_i, axis=1)  # (d, )
            multiplier_u = np.sum(y_i.T[:, :, None] * y_i.T[:, None], axis=0)  # (d,d)
            multiplier_u_inv = np.linalg.inv(multiplier_u + hyper_params['reg_x']*np.identity(multiplier_u.shape[0]))   # (d,d)
            x_u_new = np.dot(multiplier_u_inv, sigma_D_u)  # (d,)
            X_u[:, user - 1] = x_u_new

        for item in D_i.keys():
            D_i_item = D_i[item]  # array: user_id, rating
            # Ratings of the userd that ranked the item:
            r_ui = D_i_item[:, 1]
            # User biases of the user that ranked the item:
            b_u_D_i = b_u[D_i_item[:, 0] - 1]
            y_i = Y_i[:, item - 1]  # d*1
            x_u = X_u[:, D_i_item[:, 0] - 1]  # d*|D_i|
            x_u_T = x_u.T  # |D_i|*d
            b_i_old = b_i[item - 1]

            # b_i update:
            x_u_T_y_i = np.dot(x_u_T, y_i)  # |D_u|*1
            sigma_D_i = np.sum(r_ui - mu - b_u_D_i - x_u_T_y_i)
            multiplier_i = 1 / (D_i_item.shape[0] + hyper_params['reg_bi'])  # V
            b_i_new = sigma_D_i * multiplier_i
            b_i[item - 1] = b_i_new

            # y_i update:
            sigma_D_i = np.sum((r_ui - mu - b_u_D_i - b_i_old) * x_u, axis=1)  # (d, )
            multiplier_i = np.sum(x_u_T[:, :, None] * x_u_T[:, None], axis=0)  # (d,d)
            multiplier_i_inv = np.linalg.inv(multiplier_i + hyper_params['reg_y']*np.identity(multiplier_i.shape[0]))  # (d,d)
            y_i_new = np.dot(multiplier_i_inv, sigma_D_i)  # (d,)
            Y_i[:, item - 1] = y_i_new

        prev_train_error = current_train_error
        prev_validation_error = current_validation_error

        current_train_error = calc_error(train_data, mu, b_i, b_u, item_index, 'full', X_u, Y_i)[0] + \
                              calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)
        current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index, 'full', X_u, Y_i)[0] + \
                                   calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)

        if epoch == 2:
            best_validation_error = current_validation_error + 1

        if current_validation_error < best_validation_error:
            b_i_best = np.copy(b_i)
            b_u_best = np.copy(b_u)
            X_u_best = np.copy(X_u)
            Y_i_best = np.copy(Y_i)
            best_train_error = np.copy(current_train_error)
            best_validation_error = np.copy(current_validation_error)
            best_rmse, best_mae, best_r_squre = \
                calc_evaluation_measures(validation_data, mu, b_i_best, b_u_best, item_index, model_type='full',
                                         X_u=X_u_best, Y_i=Y_i_best)

        if current_validation_error > prev_validation_error:
            allowed_inc += 1

        t_error.append(current_train_error)
        v_error.append(current_validation_error)
        epoch += 1

        if epoch % 10 == 0:
            print('-' * 80)
            print('epoch %d:\n' % epoch)
            print('train error: ', current_train_error)
            print('validation error: ', current_validation_error)
            RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='full',
                                                           X_u=X_u, Y_i=Y_i)
            print('RMSE: ', RMSE)
            print('MAE: ', MAE)
            print('R_square: ', R_square)
    print('Total epochs till convergence: ' + str(epoch))
    return mu, b_i_best, b_u_best, X_u_best, Y_i_best, best_train_error, best_validation_error, best_mae, best_r_squre,\
           t_error, v_error, best_rmse


def als_main(train_data, validation_data, D_u, D_i, item_index):
    als_best_hp = np.array([0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 5])  # lr, reg_bu, reg_bi, reg_x, reg_y, sigma, dim

    print('\nSTARTING  BIAS  MODEL:')
    als_mu, als_b_i_best, als_b_u_best, als_best_train_error, als_best_validation_error, als_best_mae, als_best_r_squre, als_best_rmse = run_bias_model_ALS(
        train_data, validation_data, als_best_hp, D_u, D_i, item_index, epoch_num=5)

    rmse_bias, mae_bias, r_square_bias = \
        calc_evaluation_measures(validation_data, als_mu, als_b_i_best, als_b_u_best, item_index, 'bias')

    print('\nSTARTING  FULL  MODEL:')
    als_full_mu, als_full_b_i_best, als_full_b_u_best, als_full_X_u_best, als_full_Y_i_best, als_full_best_train_error, als_full_best_validation_error, als_full_best_mae, als_full_best_r_squre, \
    als_full_t_error, als_full_v_error, als_full_best_rmse = run_full_model_ALS(train_data, validation_data, als_best_hp, D_u, D_i, als_b_i_best, als_b_u_best, als_mu, item_index)

    rmse_full, mae_full, r_square_full = \
        calc_evaluation_measures(validation_data, als_full_mu, als_full_b_i_best, als_full_b_u_best, item_index, 'full',
                                 als_full_X_u_best, als_full_Y_i_best)

    print('---- Summary: ----')
    print('Bias Model - ALS:')
    print('RMSE = ' + str(rmse_bias))
    print('MAE = ' + str(mae_bias))
    print('R^2 = ' + str(r_square_bias))

    print('Full Model - ALS:')
    print('RMSE = ' + str(rmse_full))
    print('MAE = ' + str(mae_full))
    print('R^2 = ' + str(r_square_full))

    return als_full_mu, als_full_b_i_best, als_full_b_u_best, als_full_X_u_best, als_full_Y_i_best, \
           als_full_best_train_error, als_full_best_validation_error, als_full_best_mae, als_full_best_r_squre, \
           als_full_best_rmse


if __name__ == '__main__':
    train_data_df = read_data_to_df('Data\\Train.csv')
    # Change the index from the data and for the model use:
    item_index = item_to_index(train_data_df)
    # Initiate D_u and D_i dictionaries:
    print('-' * 80)
    print('Initiate D_u and D_i dictionaries')
    D_u, D_i = data_prep_ALS(train_data_df, item_index)

    train_data = train_data_df.values
    validation_data = read_data_to_df('Data\\Validation.csv').values

    als_f_mu, als_f_b_i, als_f_b_u, als_f_X_u, als_f_Y_i, \
    als_f_train_error, als_f_validation_error, als_f_mae, als_f_r_squre, \
    als_f_rmse = als_main(train_data, validation_data, D_u, D_i, item_index)
