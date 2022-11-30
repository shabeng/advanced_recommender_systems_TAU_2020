from utilis import *
from BPR import *
from NegativeSampling import *


def pre_process(hyper_set, epoch_num=50, read_directory=True, write_num=1, sample_method='P', if_test=False):
    train = read_data_to_df('Data\\Train.csv')
    items_index = create_index(train, col_name='ItemID')
    users_index = create_index(train, col_name='UserID')

    if read_directory:
        with open('ModelData\\train_epochs_lst%d%s.pickle' % (write_num, sample_method), 'rb') as handle:
            all_epochs_data_lst = pickle.load(handle)
        with open('ModelData\\validation%d%s.pickle' % (write_num, sample_method), 'rb') as handle:
            valid_data = pickle.load(handle)
        epochs_num = len(all_epochs_data_lst)

    else:
        epochs_num = epoch_num
        c = count_items_freq(train.values)
        pos_items_per_user_dicti = create_pos_items_per_user(train.values, users_index)
        neg_samp = negative_sample(sample_method, pos_items_per_user_dicti, items_index, users_index, c,
                                   neg_multiplier=epochs_num, replace=True)
        if not if_test:
            all_epochs_data_lst, valid_data = train_valid_split(neg_samp, pos_items_per_user_dicti, epochs_num)
            with open('ModelData\\train_epochs_lst%d%s.pickle' % (write_num, sample_method), 'wb') as handle:
                pickle.dump(all_epochs_data_lst, handle)

            with open('ModelData\\validation%d%s.pickle' % (write_num, sample_method), 'wb') as handle:
                pickle.dump(valid_data, handle)
        else:
            all_epochs_data_lst = train_no_valid(neg_samp, pos_items_per_user_dicti, epochs_num)

            with open('ModelData\\train_epochs_lst_test%d%s.pickle' % (write_num, sample_method), 'wb') as handle:
                pickle.dump(all_epochs_data_lst, handle)

            valid_data = None

    num_items = len(items_index)
    num_users = len(users_index)
    hyper_set = hyper_set
    bpr_model = BPRModel(num_items=num_items, num_users=num_users, hyper_set=hyper_set)
    return train, items_index, users_index, bpr_model, all_epochs_data_lst, valid_data


def PostProcess_TestResult(model_obj, users_index, items_index, details='popularity'):
    """
    Create the csv files for the test data
    :param model_obj: the last bpr model - after choosing the best hp set and run on entire data! (no validation)
    :param users_index: mapping from data to model indices
    :param items_index:
    :param details: random    OR    popularity
    :return: saves csv with our result
    """
    file_directory = 'Data\\' + details.title() + 'Test.csv'
    test_data_df = pd.read_csv(file_directory)
    bit_classification_col = []
    for index, row in test_data_df.iterrows():
        if (row['Item1'] not in items_index) and (row['Item2'] in items_index):
            b_1 = model_obj.b_i_best.mean()
            b_2 = model_obj.b_i_best[items_index[row['Item2']]]
            bit_classification_col.append(int(b_2 > b_1) if b_2 != b_1 else np.random.choice(2))
            print('IN bias - first')
        elif (row['Item2'] not in items_index) and (row['Item1'] in items_index):
            b_2 = model_obj.b_i_best.mean()
            b_1 = model_obj.b_i_best[items_index[row['Item1']]]
            bit_classification_col.append(int(b_2 > b_1) if b_2 != b_1 else np.random.choice(2))
            print('IN bias - second')
        elif (row['Item2'] not in items_index) and (row['Item1'] not in items_index):
            bit_classification_col.append(np.random.choice(2))
            print('IN random - third')
        else:
            m, item_1, item_2 = users_index[row['UserID']], items_index[row['Item1']], items_index[row['Item2']]
            bit_classification_col.append(model_obj.predict_triplet(m, item_1, item_2))

    test_data_df['bitClassification'] = bit_classification_col
    test_data_df.to_csv(details + '_ID1_ID2.csv', index=False)
    return test_data_df
