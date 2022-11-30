import numpy as np
import pandas as pd
import time
from utilis import *


def run_bias_model_SGD(train_data, validation_data, hyper_params, item_index, epoch_num=10):
    """
    Run only the bias model
    :param train_data: array - each row is one data point: 1st col user_id, 2nd col item_id, 3rd col rating
    :param validation_data: array - each row is one data point: 1st col user_id, 2nd col item_id, 3rd col rating
    :param hyper_params: dictionary with the value of each hyper-parameter (lr, reg_bu, reg_bi).
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param epoch_num: the number of epochs to run the train
    :return: the model parameters, some quality measures as the rmse, mae, objective function value and more.
    """
    hyper_params = create_hyper_params_dict(hyper_params)
    mu, b_i, b_u = calc_bias(train_data) # b_i is vector where the j item is in the j-1 place, b_u is vector where the j user is in the j-1 place
    current_train_error = calc_error(train_data, mu, b_i, b_u, item_index)[0] \
                          + calc_reg(b_i, b_u, hyper_params, model_type='bias')
    current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index)[0] \
                               + calc_reg(b_i, b_u, hyper_params, model_type='bias')
    RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='bias',
                                                   X_u=None, Y_i=None)
    print('train error: ', current_train_error)
    print('validation error: ', current_validation_error)

    # Best Model:
    b_i_best = b_i
    b_u_best = b_u
    best_train_error = current_train_error
    best_validation_error = current_validation_error
    best_rmse = RMSE
    best_mae = MAE
    best_r_squre = R_square

    for epoch in range(epoch_num):
        current_train_error = 0
        for u, i, r_ui in train_data:
            b_u_old = b_u[u - 1]
            b_i_old = b_i[item_index[i] - 1]
            e_ui = r_ui - clipping(mu + b_i_old + b_u_old)

            b_u_new = b_u_old + hyper_params['lr']*(e_ui - hyper_params['reg_bu']*b_u_old)
            b_u[u - 1] = b_u_new

            b_i_new = b_i_old + hyper_params['lr']*(e_ui - hyper_params['reg_bi']*b_i_old)
            b_i[item_index[i] - 1] = b_i_new

            current_train_error += e_ui**2
        prev_validation_error = current_validation_error
        current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index)[0] \
                                   + calc_reg(b_i, b_u, hyper_params, model_type='bias')
        current_train_error += calc_reg(b_i, b_u, hyper_params, model_type='bias')

        if current_validation_error < best_validation_error:
            b_i_best = np.copy(b_i)
            b_u_best = np.copy(b_u)
            best_train_error = current_train_error
            best_validation_error = current_validation_error
            best_rmse, best_mae, best_r_squre = calc_evaluation_measures(validation_data, mu, b_i_best, b_u_best,
                                                                         item_index, model_type='bias')

    print('-'*80)
    print('epoch %d' % epoch)
    print('train error: ', best_train_error)
    print('validation error: ', best_validation_error)
    RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i_best, b_u_best, item_index, model_type='bias')
    print('RMSE: ', RMSE)
    print('MAE: ', MAE)
    print('R_square: ', R_square)

    return mu, b_i_best, b_u_best, best_train_error, best_validation_error, best_rmse, best_mae, best_r_squre


def run_full_model_SGD(train_data, validation_data, hyper_params, b_i, b_u, mu, item_index, clip=True):
    """
    Train the matrix Factorization using the ALS optimization method and the full model
    :param train_data: train data set array
    :param validation_data: validation data set array
    :param hyper_params: array with the value of each hyper-parameter (lr, reg_bu, reg_bi, reg_x, reg_y, dim, sigma).
    :param b_i: the best item biases found on the bias model
    :param b_u: the best user biases found on the bias model
    :param mu: the mean rationg of the training data
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param clip: boolean whetther to use the clipping func or not.
    :return: the model parameters, some quality measures as the rmse, mae, objective function value and more.
    """
    hyper_params = create_hyper_params_dict(hyper_params)
    np.random.RandomState(2020)
    X_u, Y_i = initiate_vectors(hyper_params['dim'], hyper_params['sigma'], b_u.shape[0], b_i.shape[0])
    current_train_error = calc_error(train_data, mu, b_i, b_u, item_index, model_type='full', X_u=X_u, Y_i=Y_i)[0] + \
                          calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)
    current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index,  model_type='full', X_u=X_u, Y_i=Y_i)[0] \
                               + calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)
    RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='full', X_u=X_u, Y_i=Y_i)
    print('train error: ', current_train_error)
    print('validation error: ', current_validation_error)
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

    while allowed_inc <= 3:
        current_train_error = 0
        t_e_start = time.time()
        for u, i, r_ui in train_data:
            b_u_old = b_u[u - 1]
            b_i_old = b_i[item_index[i] - 1]
            X_u_old = X_u[:, u-1]
            Y_i_old = Y_i[:, item_index[i] - 1]

            # e_ui = r_ui - clipping(mu + b_i_old + b_u_old + np.dot(X_u_old, Y_i_old))
            if clip:
                e_ui = r_ui - clipping(mu + b_i_old + b_u_old + np.dot(X_u_old, Y_i_old))
            else:
                e_ui = r_ui - (mu + b_i_old + b_u_old + np.dot(X_u_old, Y_i_old))

            # Update step
            b_u_new = b_u_old + hyper_params['lr'] * (e_ui - hyper_params['reg_bu'] * b_u_old)
            b_u[u - 1] = b_u_new

            b_i_new = b_i_old + hyper_params['lr'] * (e_ui - hyper_params['reg_bi'] * b_i_old)
            b_i[item_index[i] - 1] = b_i_new

            X_u_new = X_u_old + hyper_params['lr']*(e_ui * Y_i_old - hyper_params['reg_x']*X_u_old)
            X_u[:, u - 1] = X_u_new

            Y_i_new = Y_i_old + hyper_params['lr'] * (e_ui * X_u_old - hyper_params['reg_y'] * Y_i_old)
            Y_i[:, item_index[i] - 1] = Y_i_new

            current_train_error += e_ui ** 2

            # e_ui_new = r_ui - clipping(mu + b_i_new + b_u_new + np.dot(X_u_new, Y_i_new))
            if clip:
                e_ui_new = r_ui - clipping(mu + b_i_new + b_u_new + np.dot(X_u_new, Y_i_new))
            else:
                e_ui_new = r_ui - (mu + b_i_new + b_u_new + np.dot(X_u_new, Y_i_new))

        t_e_end = time.time()

        prev_validation_error = current_validation_error
        current_validation_error = calc_error(validation_data, mu, b_i, b_u, item_index,  model_type='full', X_u=X_u, Y_i=Y_i)[0]  \
                                   + calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)
        current_train_error += calc_reg(b_i, b_u, hyper_params, model_type='full', X_u=X_u, Y_i=Y_i)

        if current_validation_error < best_validation_error:
            X_u_best = np.copy(X_u)
            Y_i_best = np.copy(Y_i)
            b_i_best = np.copy(b_i)
            b_u_best = np.copy(b_u)
            best_train_error = np.copy(current_train_error)
            best_validation_error = np.copy(current_validation_error)
            best_rmse, best_mae, best_r_squre = \
                calc_evaluation_measures(validation_data, mu, b_i_best, b_u_best, item_index, model_type='full', X_u=X_u_best, Y_i=Y_i_best)

        if current_validation_error > prev_validation_error:
            allowed_inc += 1

        if epoch % 10 == 0:
            print('-' * 80)
            print('epoch %d:' % epoch)
            print('time: ' + str(t_e_end - t_e_start) + ' seconds')
            print('train error: ', current_train_error)
            print('validation error: ', current_validation_error)
            RMSE, MAE, R_square = calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='full',
                                                           X_u=X_u, Y_i=Y_i)
            print('RMSE: ', RMSE)
            print('MAE: ', MAE)
            print('R_square: ', R_square)
            hyper_params['lr'] = hyper_params['lr'] * 0.9

        t_error.append(current_train_error)
        v_error.append(current_validation_error)

        if epoch >= 150 and best_rmse > 0.9:
            return mu, b_i_best, b_u_best, X_u_best, Y_i_best, best_train_error, best_validation_error, \
                   best_mae, best_r_squre, \
                   t_error, v_error, \
                   best_rmse

        epoch += 1

    print('Total epochs till convergence: ' + str(epoch))
    return mu, b_i_best, b_u_best, X_u_best, Y_i_best, best_train_error, best_validation_error, \
           best_mae, best_r_squre, \
           t_error, v_error, \
           best_rmse


def sgd_main(t_data, v_data, item_index):
    sgd_best_hp = np.array([0.01, 0.01, 0.01, 0.001, 0.01, 0.0001, 40])  # lr, reg_bu, reg_bi, reg_x, reg_y, sigma, dim

    print('\nSTARTING  BIAS  MODEL:')
    sgd_mu_b, sgd_b_i_b, sgd_b_u_b, sgd_train_error_b, sgd_validation_error_b, sgd_rmse_b, sgd_mae_b, sgd_r_squre_b = \
        run_bias_model_SGD(t_data, v_data, sgd_best_hp, item_index, epoch_num=10)

    rmse_b, mae_b, r_square_b = \
        calc_evaluation_measures(v_data, sgd_mu_b, sgd_b_i_b, sgd_b_u_b, item_index, 'bias')

    print('\nSTARTING  FULL  MODEL:')
    sgd_mu_f, sgd_b_i_f, sgd_b_u_f, sgd_X_u_f, sgd_Y_i_f, sgd_train_error_f, sgd_validation_error_f, sgd_mae_f, sgd_r_squre_f, \
    sgd_t_error_f, sgd_v_error_f, sgd_rmse_f = run_full_model_SGD(t_data, v_data, sgd_best_hp, sgd_b_i_b, sgd_b_u_b, sgd_mu_b, item_index)

    rmse_full, mae_full, r_square_full = \
        calc_evaluation_measures(v_data, sgd_mu_f, sgd_b_i_f, sgd_b_u_f, item_index, 'full', sgd_X_u_f, sgd_Y_i_f)

    print('---- Summary: ----')
    print('Bias Model - SGD:')
    print('RMSE = ' + str(rmse_b))
    print('MAE = ' + str(mae_b))
    print('R^2 = ' + str(r_square_b))

    print('Full Model - SGD:')
    print('RMSE = ' + str(rmse_full))
    print('MAE = ' + str(mae_full))
    print('R^2 = ' + str(r_square_full))

    return sgd_mu_f, sgd_b_i_f, sgd_b_u_f, sgd_X_u_f, sgd_Y_i_f, sgd_train_error_f, sgd_validation_error_f, sgd_mae_f, \
           sgd_r_squre_f, sgd_rmse_f


if __name__ == '__main__':
    t_data_df = read_data_to_df('Data\\Train.csv')
    # Change the index from the data and for the model use:
    item_index = item_to_index(t_data_df)
    # Initiate D_u and D_i dictionaries:
    print('-' * 80)

    t_data = t_data_df.values
    v_data = read_data_to_df('Data\\Validation.csv').values

    sgd_mu_f, sgd_b_i_f, sgd_b_u_f, sgd_X_u_f, sgd_Y_i_f, sgd_train_error_f, sgd_validation_error_f, sgd_mae_f, \
    sgd_r_squre_f, sgd_rmse_f = sgd_main(t_data, v_data, item_index)
