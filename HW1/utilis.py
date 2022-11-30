import pandas as pd
import numpy as np


def read_data_to_df(path_str):
    """
    Read the specified data file. Assume the code is with the same directory as the Data folder.
    :param path_str: The file path
    :return: data frame of the data
    """
    data_df = pd.read_csv(path_str, header=0)
    return data_df


def item_to_index(data_df):
    """
    Make a dictionary from item_id on the data to continue range of number (since there are missing item id in between).
    :param data_df: dataframe with the columns - User_ID_Alias, Movie_ID_Alias, Ratings_Rating
    :return: dictionary with keys as the item_id from data and values as the new continue id (starting from 1).
    """
    # Change the index from the data and for the model use:
    item_index = {}
    i = 0
    for num in sorted(data_df['Movie_ID_Alias'].unique()):
        item_index[num] = i + 1
        i = i + 1
    return item_index


def data_prep_ALS(data_df, item_index_dict):
    """
    Instead of searching for the subset of the dataset for specific user or item, these are calculated in advance once
    as dictionary of arrays.
    :param data_df: dataframe with the columns - User_ID_Alias, Movie_ID_Alias, Ratings_Rating
    :param item_index_dict: dictionary with keys as the item_id from data and values as the new continue id
            (starting from 1).
    :return: D_u, D_i: dictionary with key as the id and the values as array with all the relevant row from the data
            (starting from 1 and with the correct id for the items).
    """
    # Initiate D_u and D_i dictionaries:
    # D_u:
    D_u = {}
    print('Start D_u')
    for num in sorted(data_df['User_ID_Alias'].unique()):
        d_u_array = data_df.loc[data_df['User_ID_Alias'] == num][['Movie_ID_Alias', 'Ratings_Rating']]
        d_u_array['item_index'] = d_u_array.apply(lambda row: item_index_dict[row[0]], axis=1)
        D_u[num] = d_u_array[['item_index', 'Ratings_Rating']].values

    # D_i:
    D_i = {}
    print('Start D_i')
    for num in sorted(data_df['Movie_ID_Alias'].unique()):
        d_i_array = data_df.loc[data_df['Movie_ID_Alias'] == num][['User_ID_Alias', 'Ratings_Rating']].values
        inx_num = item_index_dict[num]
        D_i[inx_num] = d_i_array
    return D_u, D_i


def calc_bias(train_data):
    """
    Calculate initial biases of the model with the mean ratings of the training set.
    :param train_data: train data set array
    :return:
    """
    data_df = pd.DataFrame(train_data, columns=['user_id', 'item_id', 'rating'])
    mu = data_df.mean(axis=0)['rating']
    b_i = data_df.groupby(['item_id']).mean()['rating'].values - mu
    b_u = data_df.groupby(['user_id']).mean()['rating'].values - mu

    return mu, b_i, b_u


def calc_error(data, mu, b_i, b_u, item_index, model_type='bias', X_u=None, Y_i=None):
    """
    Calculate the sum square errors and sum absolute errors between the true rating and the model prediction on given data.
    :param data: data array with true ratings - validation or train
    :param mu: the mean rationg of the training data
    :param b_i: model parameter
    :param b_u: model parameter
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param model_type: bias of full to determine the prediction
    :param X_u: model parameter
    :param Y_i: model parameter
    :return: sum square errors and sum absolute errors
    """
    total_error_squre = 0
    total_error_abs = 0
    for u, i, r_ui in data:
        if i in item_index:
            if model_type == 'bias':
                personalization = 0
            elif model_type == 'full':
                personalization = np.dot(X_u[:, u-1], Y_i[:, item_index[i]-1])

            e_ui = r_ui - clipping(mu + b_i[item_index[i] - 1] + b_u[u - 1] + personalization)

        else:
            e_ui = r_ui - clipping(mu + b_i.mean() + b_u[u-1])

        total_error_squre += e_ui**2
        total_error_abs += np.abs(e_ui)
    return total_error_squre, total_error_abs


def calc_reg(b_i, b_u, hyper_params, model_type='bias', X_u=None, Y_i=None):
    """
    Calculate the regularization element of the model's objective function
    :param b_i: model parameter
    :param b_u: model parameter
    :param hyper_params: array of the hyper-parameters values.
    :param model_type: bias or full
    :param X_u: model parameter
    :param Y_i: model parameter
    :return:
    """
    b_i_reg = np.sum(b_i**2)*hyper_params['reg_bi']
    b_u_reg = np.sum(b_u ** 2) * hyper_params['reg_bu']
    if model_type == 'full':
        x_u_reg = np.sum(np.linalg.norm(X_u, axis=0))*hyper_params['reg_x']
        y_i_reg = np.sum(np.linalg.norm(Y_i, axis=0)) * hyper_params['reg_y']
    else:
        x_u_reg = 0
        y_i_reg = 0
    return b_i_reg + b_u_reg + x_u_reg + y_i_reg


def clipping(prediction):
    """
    Clipping function between 1 to 5.
    :param prediction: the predicted rating
    :return: the predicted  rating after clipping
    """
    if prediction > 5:
        return 5
    elif prediction < 1:
        return 1
    else:
        return prediction


def initiate_vectors(d, std, u_count, i_count):
    """
    Initialization of the latent vectors with d dimention and sigme std.
    :param d: the dimension of the vectors
    :param std: the normal std to sample from (mean is zero)
    :param u_count: the number of the users.
    :param i_count: the number of the items.
    :return: latent vectors
    """
    X_u = np.random.normal(0, std, (int(d), u_count))
    Y_i = np.random.normal(0, std, (int(d), i_count))
    return X_u, Y_i


def calc_evaluation_measures(validation_data, mu, b_i, b_u, item_index, model_type='bias', X_u=None, Y_i=None):
    """
    Calculating the model measures - RMSE, MAE, R_square
    :param validation_data: data set array
    :param mu: model parameter, int
    :param b_i: model parameter, vector
    :param b_u: model parameter, vector
    :param item_index: dictionary mapping from the data index to a continuous index.
    :param model_type: bias or full
    :param X_u: model parameter, array
    :param Y_i: model parameter, array
    :return: RMSE, MAE, R_square of the given data and model
    """
    squre, absolute = calc_error(validation_data, mu, b_i, b_u, item_index, model_type, X_u, Y_i)
    RMSE = np.sqrt(squre/validation_data.shape[0])
    MAE = absolute/validation_data.shape[0]

    SS_tot = np.sum((validation_data[:, 2] - validation_data.mean(axis=0)[2])**2)
    R_square = 1 - squre/SS_tot
    return RMSE, MAE, R_square


def create_hyper_params_dict(array):
    """
    Transform the hyper parameters set from array to dictionary
    :param array: hp values as array
    :return: hp values as dict
    """
    if type(array) == dict:
        return array
    hyper_names = ['lr', 'reg_bu', 'reg_bi', 'reg_x', 'reg_y', 'sigma', 'dim']
    dic = {}
    for i, val in enumerate(array):
        dic[hyper_names[i]] = val
    return dic


def predict_test_rating(mu, b_u, b_i, x_u, y_i, opt_method, item_index):
    """
    Producing the csv file of the test prediction
    :param mu: model parameter, int
    :param b_u: model parameter, array
    :param b_i: model parameter, array
    :param x_u: model parameter, array
    :param y_i: model parameter, array
    :param opt_method: for the file name as in the instruction
    :param item_index: dictionary mapping from the data index to a continuous index.
    :return: the test prediction as data frame and saving the csv file
    """
    if opt_method == 'SGD':
        prefix = 'A'
    else:
        prefix = 'B'
    file_name = prefix + '_204155550_204351019.csv'

    test_data_df = read_data_to_df('Data\\Test.csv')
    predicted_ratings = []

    for index, row in test_data_df.iterrows():
        u, i = row

        # Model Params
        b_u_curr = b_u[u-1]
        x_u_curr = x_u[:, u-1]

        if i in item_index:
            b_i_curr = b_i[item_index[i] - 1]
            y_i_curr = y_i[:, item_index[i] - 1]

            # Prediction using clipping function
            r_ui_pred = clipping(mu + b_i_curr + b_u_curr + np.dot(x_u_curr, y_i_curr))

        else:  # Cold-Start Prediction for item that is not on train
            r_ui_pred = clipping(mu + b_u_curr + b_i.mean())

        predicted_ratings.append(r_ui_pred)

    test_data_df['Rating'] = predicted_ratings
    test_data_df.to_csv(file_name, index=False)

    return test_data_df


def calc_evaluation_mean_model(data, mu):
    """
    Calculating the model measures - RMSE, MAE, R_square on the mean model
    :param data: data array
    :param mu: the mean rationg of the training data
    :return: RMSE, MAE, R_square
    """
    err_2 = 0
    err_abs = 0
    for u, i, r_ui in data:
        err_2 += (r_ui - mu)**2
        err_abs += np.abs(r_ui - mu)

    RMSE = np.sqrt(err_2/data.shape[0])
    MAE = err_abs/data.shape[0]

    SS_tot = np.sum((data[:, 2] - data.mean(axis=0)[2])**2)
    R_square = 1 - err_2/SS_tot

    return RMSE, MAE, R_square