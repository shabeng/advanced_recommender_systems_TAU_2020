import pandas as pd
from SGD_optimization import *
from ALS_optimization import *
from utilis import *
from hyper_parameters_tuning import *


train_data_df = read_data_to_df('Data\\Train.csv')

# Change the index from the data and for the model use:
item_index = item_to_index(train_data_df)

# Initiate D_u and D_i dictionaries:
print('-'*80)
print('Initiate D_u and D_i dictionaries')
D_u, D_i = data_prep_ALS(train_data_df, item_index)

train_data = train_data_df.values

validation_data = read_data_to_df('Data\\Validation.csv').values


# SGD Final model:
sgd_mu_f, sgd_b_i_f, sgd_b_u_f, sgd_X_u_f, sgd_Y_i_f, sgd_train_error_f, sgd_validation_error_f, sgd_mae_f, \
sgd_r_squre_f, sgd_rmse_f = sgd_main(train_data, validation_data, item_index)

# ALS Final model:
ls_f_mu, als_f_b_i, als_f_b_u, als_f_X_u, als_f_Y_i, \
    als_f_train_error, als_f_validation_error, als_f_mae, als_f_r_squre, \
    als_f_rmse = als_main(train_data, validation_data, D_u, D_i, item_index)


