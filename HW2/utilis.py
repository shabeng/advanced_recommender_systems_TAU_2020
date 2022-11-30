import numpy as np
import pandas as pd
from collections import Counter
import pickle


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dev_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def dev_log_sigmoid(x):
    pass


def read_data_to_df(path_str):
    """
    Read the specified data file. Assume the code is with the same directory as the Data folder.
    :param path_str: The file path
    :return: data frame of the data
    """
    data_df = pd.read_csv(path_str, header=0)
    return data_df


def create_index(data_df, col_name='ItemID'):
    """
    Make a dictionary from item_id on the data to continue range of number (since there are missing item id in between).
    :param data_df: data frame with the columns - User_ID_Alias, Movie_ID_Alias, Ratings_Rating
    :param col_name: string, ItemID or UserID
    :return: dictionary with keys as the item_id from data and values as the new continue id (starting from 1).
    """
    # Change the index from the data and for the model use:
    index_dict = {}
    i = 0
    for num in sorted(data_df[col_name].unique()):
        index_dict[num] = i
        i = i + 1
    return index_dict


def convert_tuple_to_dict(tup):
    hp_dict = {}
    hp_name = ['alpha_u', 'alpha_i', 'alpha_b', 'dim', 'lr', 'sigma']
    for k in range(len(tup)):
        hp_dict[hp_name[k]] = tup[k]
    return hp_dict


def calc_hit_rate_at_k_user(items_scores, k_lst, n):
    """

    :param items_scores:
    :param k_lst:
    :param n: model index of the hidden positive item
    :return:
    """
    hit_k_lst = []
    for k in k_lst:
        ids = np.argpartition(items_scores, -k)[-k:]
        hit_k_lst.append(int(n in ids))
    return np.array(hit_k_lst)


def calc_pr_user(items_scores, n):
    """

    :param items_scores:
    :param n: model index of the hidden positive item
    :return:
    """
    rank = np.empty(items_scores.size, dtype=np.int64)
    rank[items_scores.argsort()] = np.arange(items_scores.size)[::-1]
    return (rank[n] + 1)/items_scores.shape[0]


def metric_user(bpr_model, m, n, k_lst):
    """

    :param bpr_model:
    :param m: user_data_id
    :param n: left_out_pos_item
    :param k_lst:
    :return:
    """
    u_m = bpr_model.u_m_best[:, m]
    items_scores = np.dot(u_m.T, bpr_model.v_i_best) + bpr_model.b_i_best
    hit_rate_k = calc_hit_rate_at_k_user(items_scores, k_lst, n)  # 0, 1
    pr = calc_pr_user(items_scores, n)  # percentile
    return hit_rate_k, pr


def calc_metric(validation_data, bpr_model, k_lst, users_index, items_index):
    hit_rate_k = np.zeros(len(k_lst))
    mpr = 0
    M = validation_data.shape[0]
    for m, i, j in validation_data:
        m, n = users_index[m], items_index[i]
        hit_rate_k_array, pr_ = metric_user(bpr_model, m, n, k_lst)
        mpr += pr_
        hit_rate_k += hit_rate_k_array
    return hit_rate_k/M, mpr/M


