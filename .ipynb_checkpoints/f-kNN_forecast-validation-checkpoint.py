import os, glob, subprocess, datetime, sys, itertools, warnings

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats
import properscoring as ps

from scipy.integrate import quad
from scipy import interpolate
from scipy.stats import multivariate_normal
from scipy.interpolate import make_smoothing_spline

from sklearn.model_selection import KFold, LeaveOneOut


import scores 
import ffc 


warnings.filterwarnings("ignore")

path_to_data = '/Users/Guille/Desktop/dynamic_update/data'

# ----------------------------- Experiment -----------------------------

# Resource
resource = 'wind'

# Asset index
i_asset = 1

# ----------------------------- Load Data ------------------------------

# Time Zone
T_zone = 24

ac_ = pd.read_csv(path_to_data + '/actuals/' + resource + '_actual_5min_site_2017.csv')
#print(ac_.columns[i_asset + 1])

asset_name = ac_.columns[i_asset + 1].replace(' ', '_')

dates_    = ac_[['Time']].to_numpy()[T_zone*12:]
ac_       = ac_[[ac_.columns[i_asset + 1]]].to_numpy()[T_zone*12:]
ac_tr_    = ac_.reshape(int(ac_.shape[0]/288), 288)[:-1, :]
dates_tr_ = dates_.reshape(int(dates_.shape[0]/288), 288)[:-1, :]
print(ac_tr_.shape, dates_tr_.shape)

ac_ = pd.read_csv(path_to_data + '/actuals/' + resource + '_actual_5min_site_2018.csv')
#print(ac_.columns[i_asset + 1])

dates_    = ac_[['Time']].to_numpy()[T_zone*12:]
ac_       = ac_[[ac_.columns[i_asset + 1]]].to_numpy()[T_zone*12:]
ac_ts_    = ac_.reshape(int(ac_.shape[0]/288), 288)[:-1, :]
dates_ts_ = dates_.reshape(int(dates_.shape[0]/288), 288)[:-1, :]
print(ac_ts_.shape, dates_ts_.shape)

# ------------------------ Load Day-Head Forecast ------------------------

# Time Zone
T_zone = 17

fc_ = pd.read_csv(path_to_data + '/actuals/' + resource + '_day_ahead_forecast_2018.csv')
fc_ = fc_[[fc_.columns[i_asset + 3]]].to_numpy()[T_zone:-(24 - T_zone)][:, 0]
#print(fc_.shape)

x_fc_  = np.linspace(0, fc_.shape[0] - 1, fc_.shape[0], dtype = int)*12
x_ac_  = np.linspace(0, ac_.shape[0] - 1, ac_.shape[0], dtype = int)[:-11]
fc_tr_ = interpolate.interp1d(x_fc_, fc_, kind = 'linear')(x_ac_)
fc_tr_ = fc_tr_[:-277]
fc_tr_ = fc_tr_.reshape(int(fc_tr_.shape[0]/288), 288)
print(fc_tr_.shape)

fc_ = pd.read_csv(path_to_data + '/actuals/' + resource + '_day_ahead_forecast_2018.csv')
fc_ = fc_[[fc_.columns[i_asset + 3]]].to_numpy()[T_zone:-(24 - T_zone)][:, 0]
#print(fc_.shape)

x_fc_  = np.linspace(0, fc_.shape[0] - 1, fc_.shape[0], dtype = int)*12
x_ac_  = np.linspace(0, ac_.shape[0] - 1, ac_.shape[0], dtype = int)[:-11]
fc_ts_ = interpolate.interp1d(x_fc_, fc_, kind = 'linear')(x_ac_)
fc_ts_ = fc_ts_[:-277]
fc_ts_ = fc_ts_.reshape(int(fc_ts_.shape[0]/288), 288)
print(fc_ts_.shape)

# ----------------------------- Testing Sample ------------------------------



def _test_sample(ac_tr_, ac_ts_, fc_ts_, dates_, t_event, i_day = 0):
    
    t_ = np.linspace(0, ac_ts_.shape[1] - 1, ac_ts_.shape[1], dtype = int)
    #print(t_.shape, ac_ts_.shape)

    date_ = dates_[i_day, :]

    t_ts_ = t_[t_event:]
    t_tr_ = t_[:t_event]

    f_    = ac_ts_[i_day, :]
    f_tr_ = ac_ts_[i_day, :t_event]
    f_ts_ = ac_ts_[i_day, t_event:]
    #print(f_.shape, f_tr_.shape, f_ts_.shape)

    F_    = ac_tr_
    F_tr_ = ac_tr_[:, :t_event]
    F_ts_ = ac_tr_[:, t_event:]
    #print(F_.shape, F_tr_.shape, F_ts_.shape)

    fct_    = fc_ts_[i_day, :]
    fct_tr_ = fc_ts_[i_day, :t_event]
    fct_ts_ = fc_ts_[i_day, t_event:]
    #print(fct_.shape, fct_tr_.shape, fct_ts_.shape)


    return F_tr_, F_ts_, f_tr_, f_ts_, fct_tr_, fct_ts_, t_tr_, t_ts_


def _scores(f_ts_, f_ts_hat_, s_ts_hat_):

    LogS = scores._logarithmic_score(f_ts_, f_ts_hat_, s_ts_hat_).sum()
    CRPS = scores._crps(f_ts_, f_ts_hat_, s_ts_hat_).sum()
    KS   = scores._ks(f_ts_, f_ts_hat_, s_ts_hat_)
    
    return np.array([LogS, CRPS, KS])

def _cross_validation(ac_tr_, fc_tr_, dates_tr_, params_, interval = 200):

    k = 0
    scores_val_ = []
    for idx_tr_, idx_ts_ in LeaveOneOut().split(ac_tr_):

        F_tr_, F_ts_, f_tr_, f_ts_, fct_tr_, fct_ts_, t_tr_, t_ts_ = _test_sample(ac_tr_[idx_tr_, :], 
                                                                                  ac_tr_[idx_ts_, :], 
                                                                                  fc_tr_[idx_ts_, :], 
                                                                                  dates_tr_[idx_ts_, :], 
                                                                                  t_event = interval)
        
        
        f_ts_hat_, S_ts_hat_ = ffc._distance(F_tr_, f_tr_, F_ts_, fct_ts_,
                                             forget_rate  = params_['forget_rate'], 
                                             update_rate  = params_['update_rate'], 
                                             length_scale = params_['length_scale'],
                                             smoothing    = params_['smoothing'],
                                             lamdba       = params_['lamdba'], viz = False)
        
        s_ts_hat_ = np.sqrt(np.diagonal(S_ts_hat_))
        
        scores_ = _scores(f_ts_, f_ts_hat_, s_ts_hat_)
        print(k, scores_)
        
        scores_val_.append(scores_)
        k += 1

    return np.mean(np.stack(scores_val_), axis = 0)
    
    

    
forget_rate_  = [1, 5, 10]
update_rate_  = [0, 1000, 10000]
lamdba_       = [0, 10, 1000]
edf_prob_     = [0.001, 0.0005, 0.0001]
length_scale_ = [10, 100, 1000]

all_params_ = list(itertools.product(forget_rate_, update_rate_, length_scale_, edf_prob_))

df_ = []

params_ = {'forget_rate': 5,
           'update_rate': 0,
           'length_scale': 100,
           'smoothing': 0,
           'lamdba': 0}


scores_ = _cross_validation(ac_tr_, fc_tr_, dates_tr_, params_, interval = 200)
print(scores_)

# for k in range(len(all_params_)):

#     forget_rate  = all_params_[k][0]
#     update_rate  = all_params_[k][1]
#     length_scale = all_params_[k][2]
#     edf_prob     = all_params_[k][3]
#     lamdba       = 0
#     smoothing    = 2

#     params_ = [k, forget_rate, update_rate, length_scale, edf_prob, lamdba, smoothing]
#     params_ = pd.DataFrame(np.array(params_)[:, np.newaxis].T, columns = ['index', 
#                                                                           'forget_rate', 
#                                                                           'update_rate', 
#                                                                           'length_scale', 
#                                                                           'edf_prob', 
#                                                                           'lamdba', 
#                                                                           'smoothing'])
#     print(params_)

#     for yearday in range(0, 363, 1):

#         meta_ = [resource, asset_name, yearday, interval]
#         meta_ = pd.DataFrame(np.array(meta_)[:, np.newaxis].T, columns = ['resource', 
#                                                                           'asset', 
#                                                                           'year_day', 
#                                                                           'interval'])

#                                                  N_scen       = 100)
        
#         scores_ = _evaludate_update(f_ts_, f_ts_hat_, S_ts_hat_, F_scen_)
        
#         df_.append(pd.concat([params_, meta_, scores_], axis = 1))

# df_ = pd.concat(df_, axis = 0).reset_index(drop = True)

# df_.to_csv(path_to_data + '/{}-{}-{}.csv'.format(resource, asset_name, interval), index = False)