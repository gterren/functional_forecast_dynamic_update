import pickle as pkl
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.linear_model import LinearRegression

## -------------------------- LOAD DATA ---------------------------------
resource = 'wind'
unbiased = True


# Bias-correction of day-ahead forecast
def _bias_corrected_forecast(F_, E_):
    _models = {}
    E_unbias_ = np.full_like(E_, np.nan)
    for t in range(F_.shape[1]):
        for a in range(F_.shape[2]):
            # Fit linear model with intercept
            _model = LinearRegression(fit_intercept=True)
            _model = _model.fit(E_[:, t, a].reshape(-1, 1), F_[:, t, a])
            # Unbiased forecast
            E_unbias_[:, t, a] = _model.predict(E_[:, t, a].reshape(-1, 1))
    return E_unbias_

def _update_hyper_base(df1, df2, _inits):
    df = df1.copy()
    for interval in _inits:
        for parameter in list(df2.columns[-5:]):
            idx_ = ((df2['time'] == interval) 
                    & (df2['initialization'] == _inits[interval]))
            df.loc[parameter, interval] = np.around(
                df2.loc[idx_, parameter].values[0], 3
            )
    return df
    
def hyperparameters(_inits,
                    resource, 
                    method, 
                    aggregation, 
                    exp_description,
                    path_to_validation):

    file_name = f'{resource}_{method}_{aggregation}-hyper-base.csv'
    hyper_base = pd.read_csv(path_to_validation / f"{file_name}")
    hyper_base = hyper_base.set_index("parameter")
    hyper_base.columns = hyper_base.columns.astype(int)
    #print(hyper_base)
    
    file_name = f'{resource}_{method}_{aggregation}-hyper-{exp_description}.csv'
    hyper_val = pd.read_csv(path_to_validation / f"{resource}/{file_name}")
    
    hyper_ = _update_hyper_base(hyper_base, hyper_val, _inits)
    
    file_name = f'{resource}_{method}_{aggregation}-envelope-{exp_description}.csv'
    envelope_ = pd.read_csv(path_to_validation / f"{resource}/{file_name}")
    envelope_ = envelope_.set_index("time")

    return hyper_, envelope_
    
## LOAD BIASED DATA
def preprocessed_dataset(unbiased, 
                         path_to_training_data,
                         path_to_testing_data,
                         T = 288):
    
    # Load 2017 data as training set
    with open(path_to_training_data, 'rb') as f:
        _data = pkl.load(f)
    
    assets_tr_ = _data["assets"]
    X_tr_ = _data["locations"]
    dates_tr_ = _data["dates"]
    F_tr_ = _data["observations"]
    E_tr_ = _data["forecasts"]
    #print(assets_tr_.shape, F_tr_.shape, E_tr_.shape)
    
    # Reshape to day x interval x asset format
    F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
    E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
    T_tr_ = dates_tr_.reshape(int(dates_tr_.shape[0]/T), T)

    #print(F_tr_.shape, E_tr_.shape)
    
    # Load 2018 data as testing set
    with open(path_to_testing_data, 'rb') as f:
        _data = pkl.load(f)
    
    assets_ts_ = _data["assets"]
    X_ts_ = _data["locations"]
    dates_ts_ = _data["dates"]
    F_ts_ = _data["observations"]
    E_ts_ = _data["forecasts"]
    #print(assets_ts_.shape, F_ts_.shape, E_ts_.shape)
    
    # Reshape to day x interval x asset format
    F_ts_ = F_ts_.reshape(int(F_ts_.shape[0]/T), T, F_ts_.shape[1])
    E_ts_ = E_ts_.reshape(int(E_ts_.shape[0]/T), T, E_ts_.shape[1])
    T_ts_ = dates_ts_.reshape(int(dates_ts_.shape[0]/T), T)
    #print(F_ts_.shape, E_ts_.shape)

    #print(dt_.shape, dx_.shape)
    # Short testing set with training set order
    order = {v: i for i, v in enumerate(assets_tr_)}
    idx_ = np.argsort([order[x] for x in assets_ts_])
    assets_ts_ = assets_ts_[idx_]
    X_ts_ = X_ts_[idx_]
    F_ts_ = F_ts_[:, :, idx_]
    E_ts_ = E_ts_[:, :, idx_]
    #print(F_ts_.shape, E_ts_.shape)
    
    # From generation to capacity factor
    p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
    p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
    #print(p_tr_.shape, p_ts_.shape)
    
    F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
    F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
    E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
    E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))

    # Unbias the forecasts
    if unbiased:
        E_tr_ = _bias_corrected_forecast(F_tr_, E_tr_)
        E_ts_ = _bias_corrected_forecast(F_ts_, E_ts_)
        # print(E_tr_.shape, E_ts_.shape)
    
    # No possible capacity factor is larger than 1 or smaller than 0
    F_tr_ = np.clip(F_tr_, 0., 1.)
    F_ts_ = np.clip(F_ts_, 0., 1.)
    E_tr_ = np.clip(E_tr_, 0., 1.)
    E_ts_ = np.clip(E_ts_, 0., 1.)
    F_tr_ /= F_tr_.max()
    F_ts_ /= F_ts_.max()
    E_tr_ /= E_tr_.max()
    E_ts_ /= E_ts_.max()
    # print(F_tr_.min(), F_tr_.max(), E_tr_.min(), E_tr_.max())
    # print(F_ts_.min(), F_ts_.max(), E_ts_.min(), E_ts_.max())

    #print(E_tr_bias_.shape, E_ts_bias_.shape)

    # Format training set from day x interval x asset to 
    # [day * asset] x interval
    T_tr_ = np.concatenate(
        [T_tr_ for k in range(assets_tr_.shape[0])], axis = 0
    )
    
    assets_tr_ = np.concatenate(
        [np.tile(assets_tr_[k], (F_tr_.shape[0], 1)) 
         for k in range(assets_tr_.shape[0])], axis = 0
    )

    X_tr_ = np.concatenate(
        [np.tile(X_tr_[k, :], (F_tr_.shape[0], 1)) 
         for k in range(X_tr_.shape[0])], axis = 0
    )

    F_tr_ = np.concatenate(
        [F_tr_[..., k] 
         for k in range(F_tr_.shape[2])], axis = 0
    )

    E_tr_ = np.concatenate(
        [E_tr_[..., k] 
         for k in range(E_tr_.shape[2])], axis = 0
    )
    
    t_tr_ = np.array(
        [datetime.strptime(t_tr, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday 
         for t_tr in T_tr_[:, 0]]
    ) - 1

    t_ts_ = np.array(
        [datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday 
         for t_ts in T_ts_[:, 0]]
    ) - 1
    #print(t_tr_.shape, t_ts_.shape)
    
    dt_ = np.array([t * 5 for t in range(T)])
    dx_ = pd.to_datetime(
        pd.DataFrame({"time": dt_p_}).time, unit = "m"
    ).dt.strftime("%H:%M").to_numpy()
    #print(dt_.shape, dx_.shape)

    return (F_tr_, F_ts_, 
            E_tr_, E_ts_, 
            X_tr_, X_ts_,
            T_tr_, T_ts_,
            assets_tr_, assets_ts_, 
            t_tr_, t_ts_,
            dt_, dx_)

## LOAD UNBIASED DATA WITH LINEAR INTERPOLATION
def processed_dataset(unbiased, 
                      path_to_training_data,
                      path_to_testing_data,  
                      T = 288):
    
    # Load 2017 data as training set
    with open(path_to_training_data, 'rb') as f:
        _data = pkl.load(f)
    
    assets_tr_ = _data["assets"]
    F_tr_ = _data["observations"]
    E_tr_ = _data["forecasts"]
    #print(assets_tr_.shape, F_tr_.shape, E_tr_.shape)
    
    # Reshape to day x interval x asset format
    F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
    E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
    #print(F_tr_.shape, E_tr_.shape)
    
    # Load 2018 data as testing set
    with open(path_to_testing_data, 'rb') as f:
        _data = pkl.load(f)
    
    assets_ts_ = _data["assets"]
    F_ts_ = _data["observations"]
    E_ts_ = _data["forecasts"]
    #print(assets_ts_.shape, F_ts_.shape, E_ts_.shape)
    
    # Reshape to day x interval x asset format
    F_ts_ = F_ts_.reshape(int(F_ts_.shape[0]/T), T, F_ts_.shape[1])
    E_ts_ = E_ts_.reshape(int(E_ts_.shape[0]/T), T, E_ts_.shape[1])
    #print(F_ts_.shape, E_ts_.shape)
    
    # Short testing set with training set order
    order = {v: i for i, v in enumerate(assets_tr_)}
    idx_ = np.argsort([order[x] for x in assets_ts_])
    F_ts_ = F_ts_[:, :, idx_]
    E_ts_ = E_ts_[:, :, idx_]
    #print(F_ts_.shape, E_ts_.shape)
    
    # From generation to capacity factor
    p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
    p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
    #print(p_tr_.shape, p_ts_.shape)
    
    F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
    F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
    E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
    E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))
    
    # Bias-correct the forecasts
    if unbiased:
        E_tr_ = _bias_corrected_forecast(F_tr_, E_tr_)
        E_ts_ = _bias_corrected_forecast(F_ts_, E_ts_)
        # print(E_tr_.shape, E_ts_.shape)
    
    # No possible capacity factor is larger than 1 or smaller than 0
    F_tr_ = np.clip(F_tr_, 0., 1.)
    F_ts_ = np.clip(F_ts_, 0., 1.)
    E_tr_ = np.clip(E_tr_, 0., 1.)
    E_ts_ = np.clip(E_ts_, 0., 1.)
    F_tr_ /= F_tr_.max()
    F_ts_ /= F_ts_.max()
    E_tr_ /= E_tr_.max()
    E_ts_ /= E_ts_.max()
    # print(F_tr_.min(), F_tr_.max(), E_tr_.min(), E_tr_.max())
    # print(F_ts_.min(), F_ts_.max(), E_ts_.min(), E_ts_.max())
    
    # Format training set from day x interval x asset 
    # to [day * asset] x interval
    E_ts_lin_ = E_ts_.copy()
    E_tr_lin_ = np.concatenate(
        [E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0
    )
    #print(E_ts_lin_.shape, E_tr_lin_.shape)
    return E_tr_lin_, E_ts_lin_

# ## LOAD UNBIASED DATA
# # Load 2017 data as training set
# with open(path_to_data + f"/preprocessed_{resource}_2017.pkl", 'rb') as f:
#     _data = pkl.load(f)

# assets_tr_ = _data["assets"]
# X_tr_ = _data["locations"]
# dates_tr_ = _data["dates"]
# F_tr_ = _data["observations"]
# E_tr_ = _data["forecasts"]
# #print(assets_tr_.shape, X_tr_.shape, dates_tr_.shape, F_tr_.shape, E_tr_.shape)

# # Reshape to day x interval x asset format
# F_tr_ = F_tr_.reshape(int(F_tr_.shape[0]/T), T, F_tr_.shape[1])
# E_tr_ = E_tr_.reshape(int(E_tr_.shape[0]/T), T, E_tr_.shape[1])
# T_tr_ = dates_tr_.reshape(int(dates_tr_.shape[0]/T), T)
# #print(F_tr_.shape, E_tr_.shape, T_tr_.shape)

# # Load 2018 data as testing set
# with open(path_to_data + f"/preprocessed_{resource}_2018.pkl", 'rb') as f:
#     _data = pkl.load(f)

# assets_ts_ = _data["assets"]
# X_ts_ = _data["locations"]
# dates_ts_ = _data["dates"]
# F_ts_ = _data["observations"]
# E_ts_ = _data["forecasts"]
# #print(assets_ts_.shape, X_ts_.shape, dates_ts_.shape, F_ts_.shape, E_ts_.shape)

# # Reshape to day x interval x asset format
# F_ts_ = F_ts_.reshape(int(F_ts_.shape[0]/T), T, F_ts_.shape[1])
# E_ts_ = E_ts_.reshape(int(E_ts_.shape[0]/T), T, E_ts_.shape[1])
# T_ts_ = dates_ts_.reshape(int(dates_ts_.shape[0]/T), T)
# #print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# dt_ = np.array([t * 5 for t in range(T)])
# dx_ = pd.to_datetime(pd.DataFrame({"time": dt_}).time, unit = "m").dt.strftime("%H:%M").to_numpy()
# #print(dt_.shape, dx_.shape)

# # Short testing set with training set order
# order = {v: i for i, v in enumerate(assets_tr_)}
# idx_ = np.argsort([order[x] for x in assets_ts_])
# assets_ts_ = assets_ts_[idx_]
# X_ts_ = X_ts_[idx_]
# F_ts_ = F_ts_[:, :, idx_]
# E_ts_ = E_ts_[:, :, idx_]
# #print(F_ts_.shape, E_ts_.shape, T_ts_.shape)

# # From generation to capacity factor
# p_tr_ = np.max(np.max(F_tr_, axis = 0), axis = 0)
# p_ts_ = np.max(np.max(F_ts_, axis = 0), axis = 0)
# #print(p_tr_.shape, p_ts_.shape)

# F_tr_ /= np.tile(p_tr_, (F_tr_.shape[0], F_tr_.shape[1], 1))
# E_tr_ /= np.tile(p_tr_, (E_tr_.shape[0], E_tr_.shape[1], 1))
# F_ts_ /= np.tile(p_ts_, (F_ts_.shape[0], F_ts_.shape[1], 1))
# E_ts_ /= np.tile(p_ts_, (E_ts_.shape[0], E_ts_.shape[1], 1))
# # print(F_tr_.min(), F_tr_.max())
# # print(E_tr_.min(), E_tr_.max())
# # print(F_ts_.min(), F_ts_.max())
# # print(E_ts_.min(), E_ts_.max())

# # Unbias the forecasts
# if unbiased:
#     E_tr_ = _bias_corrected_forecast(F_tr_, E_tr_)
#     E_ts_ = _bias_corrected_forecast(F_ts_, E_ts_)
#     print(E_tr_.shape, E_ts_.shape)

# # No possible capacity factor is larger than 1 or smaller than 0
# F_tr_ = np.clip(F_tr_, 0., 1.)
# F_ts_ = np.clip(F_ts_, 0., 1.)
# E_tr_ = np.clip(E_tr_, 0., 1.)
# E_ts_ = np.clip(E_ts_, 0., 1.)
# F_tr_ /= F_tr_.max()
# F_ts_ /= F_ts_.max()
# E_tr_ /= E_tr_.max()
# E_ts_ /= E_ts_.max()
# print(F_tr_.min(), F_tr_.max(), E_tr_.min(), E_tr_.max())
# print(F_ts_.min(), F_ts_.max(), E_ts_.min(), E_ts_.max())

# # Format training set from day x interval x asset to [day * asset] x interval
# T_tr_ = np.concatenate([T_tr_ for k in range(assets_tr_.shape[0])], axis = 0)
# assets_tr_ = np.concatenate([np.tile(assets_tr_[k], (F_tr_.shape[0], 1)) for k in range(assets_tr_.shape[0])], axis = 0)
# X_tr_ = np.concatenate([np.tile(X_tr_[k, :], (F_tr_.shape[0], 1)) for k in range(X_tr_.shape[0])], axis = 0)
# F_tr_ = np.concatenate([F_tr_[..., k] for k in range(F_tr_.shape[2])], axis = 0)
# E_tr_ = np.concatenate([E_tr_[..., k] for k in range(E_tr_.shape[2])], axis = 0)
# #print(X_tr_.shape, assets_tr_.shape, F_tr_.shape, E_tr_.shape, T_tr_.shape)
# #print(X_ts_.shape, assets_ts_.shape, F_ts_.shape, E_ts_.shape, T_ts_.shape)

# t_tr_ = np.array([datetime.strptime(t_tr, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_tr in T_tr_[:, 0]]) - 1
# t_ts_ = np.array([datetime.strptime(t_ts, "%Y-%m-%d %H:%M:%S").timetuple().tm_yday for t_ts in T_ts_[:, 0]]) - 1
# #print(t_tr_.shape, t_ts_.shape)

# dt_ = np.array([t * 5 for t in range(T)])
# dt_p_ = dt_ + 5
# dx_ = pd.to_datetime(pd.DataFrame({"time": dt_p_}).time, unit = "m").dt.strftime("%H:%M").to_numpy()
# #print(dt_.shape, dx_.shape)