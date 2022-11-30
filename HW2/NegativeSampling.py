import numpy as np
import pandas as pd
from utilis import *
from tqdm import tqdm


def negative_sample(method, positive_items_per_user_index, item_index, user_index, items_freq_count, neg_multiplier=1, replace=True):
    """
    :param method: string, P or U
    :param positive_items_per_user_index: key -> user_id, value -> array of positive items
    :param item_index:
    :param user_index:
    :param items_freq_count:
    :param neg_multiplier:
    :param replace:
    :return: dictionary: key->user_id, value -> array of negative samples (neg_multiplier samples per user)
    """
    all_users_neg_samples_dict = {}
    seed = np.random.RandomState(1993)
    for user_id in user_index.keys():
        positive_items = positive_items_per_user_index[user_id]
        negative_items = np.array(list(set(item_index.keys()) - set(positive_items)))
        if method == 'U':
            negative_samples = seed.choice(negative_items, len(positive_items)*neg_multiplier, replace=replace)
        elif method == 'P':
            negative_items_count = np.array([items_freq_count[x] for x in negative_items])
            negative_items_total_count = sum(negative_items_count)
            negative_items_dist = negative_items_count / negative_items_total_count
            negative_samples = seed.choice(negative_items, len(positive_items)*neg_multiplier, p=negative_items_dist, replace=replace)

        all_users_neg_samples_dict[user_id] = negative_samples
    return all_users_neg_samples_dict


def train_valid_split(neg_samples_dict, positive_items_per_user_index, num_epochs):
    validation_data = np.zeros((1, 3), dtype=int)
    all_epochs_data_list = []
    pos_item_valid_per_user = {}
    for epoch in tqdm(range(num_epochs)):
        epoch_data = np.zeros((1, 3), dtype=int)

        for user in positive_items_per_user_index.keys():
            row_num = positive_items_per_user_index[user].shape[0]
            user_data = np.vstack(([user]*row_num, positive_items_per_user_index[user],
                                   neg_samples_dict[user][row_num*epoch:row_num*(epoch+1)])).T
            if epoch == 0:
                num_row_for_valid = np.random.choice(row_num)
                validation_example = user_data[num_row_for_valid, :].reshape(1, 3)
                pos_item_valid = validation_example[:, 1]
                pos_item_valid_per_user[user] = pos_item_valid
                # mask = positive_items_per_user_index[user] != pos_item_valid
                # print(positive_items_per_user_index[user].shape)
                # positive_items_per_user_index[user] = positive_items_per_user_index[user][mask]
                # print(positive_items_per_user_index[user].shape)
                validation_data = np.concatenate((validation_data, validation_example))

            mask = user_data[:, 1] != pos_item_valid_per_user[user]
            user_data = user_data[mask]
            epoch_data = np.concatenate((epoch_data, user_data))

        all_epochs_data_list.append(np.delete(epoch_data, 0, 0))
    return all_epochs_data_list, np.delete(validation_data, 0, 0)


def train_no_valid(neg_samples_dict, positive_items_per_user_index, num_epochs):
    all_epochs_data_list = []
    for epoch in tqdm(range(num_epochs)):
        epoch_data = np.zeros((1, 3), dtype=int)

        for user in positive_items_per_user_index.keys():
            row_num = positive_items_per_user_index[user].shape[0]
            user_data = np.vstack(([user]*row_num, positive_items_per_user_index[user],
                                   neg_samples_dict[user][row_num*epoch:row_num*(epoch+1)])).T
            epoch_data = np.concatenate((epoch_data, user_data))

        all_epochs_data_list.append(np.delete(epoch_data, 0, 0))
    return all_epochs_data_list


def count_items_freq(data):  # item_index
    # data['ItemID'] = data.apply(lambda row: item_index[row[1]], axis=1)
    counter_dict = Counter(data[:, 1])
    return counter_dict


def create_pos_items_per_user(data_array, user_index):
    pos_items_per_user_dict = {}
    for user_id in user_index.keys():
        mask = data_array[:, 0] == user_id
        pos_item_ids_array = data_array[mask][:, 1]
        pos_items_per_user_dict[user_id] = pos_item_ids_array
    return pos_items_per_user_dict


if __name__ == '__main__':
    epoch_num = 3
    train = read_data_to_df('Data\\Train.csv')
    item_index = create_index(train, col_name='ItemID')
    users_index = create_index(train, col_name='UserID')
    c = count_items_freq(train.values)
    pos_items_per_user_dicti = create_pos_items_per_user(train.values, users_index)

    print('U:')
    neg_samp = negative_sample('U', pos_items_per_user_dicti, item_index, users_index, c, neg_multiplier=epoch_num, replace=True)

    # all_epochs_data_lst, valid_data = train_valid_split(neg_samp, pos_items_per_user_dicti, epoch_num)
    all_epochs_data_lst_no_validation_u = train_no_valid(neg_samp, pos_items_per_user_dicti, epoch_num)

    print('P:')
    neg_samp = negative_sample('P', pos_items_per_user_dicti, item_index, users_index, c, neg_multiplier=epoch_num, replace=True)

    # all_epochs_data_lst, valid_data = train_valid_split(neg_samp, pos_items_per_user_dicti, epoch_num)
    all_epochs_data_lst_no_validation_p = train_no_valid(neg_samp, pos_items_per_user_dicti, epoch_num)

