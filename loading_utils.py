import os, glob, datetime

import pandas as pd
import numpy as np

#from scipy.interpolate import make_smoothing_spline
from scipy import interpolate

# Loading and processing metadata
def _process_metadata(path, file_name):
    
    meta_ = pd.read_excel(path + file_name)
    meta_ = meta_.rename(columns = {'lati': 'lat', 
                                    'longi': 'lon', 
                                    'Facility.Name': 'name', 
                                    'Capacity': 'capacity'})    
    
    return meta_[['name', 'lon', 'lat', 'capacity']].set_index('name')

# Loading and processing of historical curves for the training dataset
def _process_training_curves(meta_, T, K, path, file_name):

    ac_tr_  = pd.read_csv(path + file_name)
    ac_tr_  = ac_tr_.iloc[T:-T]
    assets_ = ac_tr_.columns[1:K+1]

    dates_ = ac_tr_[['Time']].to_numpy()
    
    # Consitent asset ordering
    ac_tr_ = ac_tr_[assets_].to_numpy()
    #print(ac_tr_.shape)

    # Format random curves and dates
    dates_tr_ = dates_.reshape(int(dates_.shape[0]/T), T)
    F_tr_     = ac_tr_.reshape(int(ac_tr_.shape[0]/T), T, ac_tr_.shape[1])
    #print(dates_tr_.shape, F_tr_.shape)

    # Normalized between 0 and 1 by Max Power
    p_ = np.max(ac_tr_, axis = 0)
    # Utilize Max power in metadata
    #p_ = meta_.loc[assets_, 'capacity'].to_numpy()
    for i in range(p_.shape[0]):
        F_tr_[..., i] /= p_[i]
    print(F_tr_.min(), F_tr_.max())

    # Asset coordiantes
    x_tr_ = np.concatenate([np.tile(meta_.loc[assets_, ['lon', 'lat']].to_numpy()[i, :][:, np.newaxis], 
                                    F_tr_.shape[0]).T for i in range(p_.shape[0])], axis = 0)    
    
    # Format random curves
    F_tr_ = np.concatenate([F_tr_[..., i] for i in range(p_.shape[0])], axis = 0)

    # Curve dates
    T_tr_ = np.concatenate([dates_tr_[:, 0] for i in range(p_.shape[0])], axis = 0)
    T_tr_ = np.stack([T_tr_[i][:10] for i in range(T_tr_.shape[0])], axis = 0)
    
    return F_tr_, T_tr_, x_tr_, assets_, p_

# Loading and processing of historical curves for the testing dataset
def _process_testing_curves(meta_, assets_, p_, T, K, path, file_name):

    ac_ts_ = pd.read_csv(path + file_name)
    #print(ac_ts_.shape)

    dates_ = ac_ts_[['Time']].to_numpy()
    
    # Consitent asset ordering
    ac_ts_ = ac_ts_[assets_].to_numpy()
    #print(dates_.shape, ac_ts_.shape)

    # Format random curves and dates
    dates_ts_ = dates_.reshape(int(dates_.shape[0]/T), T)
    F_ts_     = ac_ts_.reshape(int(ac_ts_.shape[0]/T), T, ac_ts_.shape[1])
    #print(dates_ts_.shape, F_ts_.shape)

    # Normalized between 0 and 1 by Max Power
    for i in range(p_.shape[0]):
        F_ts_[..., i] /= p_[i]
    print(F_ts_.min(), F_ts_.max())

    # Asset coordiantes
    x_ts_ = meta_.loc[assets_, ['lon', 'lat']].to_numpy()
    
    # Curve dates
    T_ts_ = np.stack([dates_ts_[i, 0][:10] for i in range(dates_ts_.shape[0])], axis = 0)
    
    return F_ts_, T_ts_, x_ts_

# Loading and processing of historical day-ahead forecast for the training dataset
def _process_traning_forecasts(assets_, p_, T, K, path, file_name):

    fc_ = pd.read_csv(path + file_name)
    
    # Correct time zone in the forecast
    TZ_fc = 5
    fc_['Forecast_time'] = fc_['Forecast_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') 
                                                      - datetime.timedelta(hours = TZ_fc + 0.5))
    
    # Consitent asset ordering
    fc_ = fc_[assets_].to_numpy()

    # Interpolate 24h day-ahead forecast to 5min intervals
    E_tr_ = []
    x_fc_ = (np.linspace(0, fc_.shape[0] - 1, fc_.shape[0]) + .5)*12
    x_ac_ = np.linspace(0, fc_.shape[0]*12 - 1, fc_.shape[0]*12)
    for k in range(K):
        fc_tr_ = interpolate.interp1d(x_fc_, fc_[:, k], kind       = 'nearest-up', 
                                                        fill_value = "extrapolate")(x_ac_)
        E_tr_.append(fc_tr_.reshape(int(fc_tr_.shape[0]/T), T)[..., np.newaxis])
    E_tr_ = np.concatenate(E_tr_, axis = 2)

    # Normalized between 0 and 1 by Max Power
    for a in range(p_.shape[0]): 
        E_tr_[..., a] /= p_[a] 

    # Regularized so unfeaseble capacity factors does not appear
    E_tr_[E_tr_ > 1.] = 1.
    E_tr_[E_tr_ < 0.] = 0.
    print(E_tr_.min(), E_tr_.max())

    return np.concatenate([E_tr_[..., i] for i in range(p_.shape[0])], axis = 0)

# Loading and processing of historical day-ahead forecast for the testing dataset
def _process_testing_forecasts(assets_, p_, T, K, path, file_name):


    fc_ = pd.read_csv(path + file_name, index_col = None)
    
    # Correct time zone in the forecast
    TZ_fc = 6
    fc_['horizon_time'] = fc_['horizon_time'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y %H:%M') - 
                                                    datetime.timedelta(hours = TZ_fc + 0.5))
    
    # Consitent asset ordering
    fc_ = fc_[assets_].to_numpy()

    # Interpolate 24h day-ahead forecast to 5min intervals
    E_ts_ = []
    x_fc_ = (np.linspace(0, fc_.shape[0] - 1, fc_.shape[0]) + .5)*12
    x_ac_ = np.linspace(0, fc_.shape[0]*12 - 1, fc_.shape[0]*12)
    for k in range(K):
        fc_ts_ = interpolate.interp1d(x_fc_, fc_[:, k], kind       = 'nearest-up', 
                                                        fill_value = "extrapolate")(x_ac_)
        E_ts_.append(fc_ts_.reshape(int(fc_ts_.shape[0]/T), T)[..., np.newaxis])
    E_ts_ = np.concatenate(E_ts_, axis = 2)

    # Normalized between 0 and 1 by Max Power
    for a in range(p_.shape[0]): 
        E_ts_[..., a] /= p_[a] 

    # Regularized so unfeaseble capacity factors does not appear
    E_ts_[E_ts_ > 1.] = 1.
    E_ts_[E_ts_ < 0.] = 0.
    print(E_ts_.min(), E_ts_.max())

    return E_ts_