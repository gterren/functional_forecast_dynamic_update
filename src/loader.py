import pickle as pkl
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.linear_model import LinearRegression


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

def pit(_inits, resource, method, aggregation, interval, exp_description, path_to_validation):


    
    pit_ = pd.read_csv(
        path_to_validation / f"{resource}/{resource}_{method}_{aggregation}-PIT_{interval}-{exp_description}.csv"
    )

    pit_ = pit_.loc[
        pit_['initialization'] == _inits[interval]
    ].reset_index(drop = True)

    return pit_.drop(columns = ['initialization', 'time']).to_numpy()


def _update_hyper_base(df1, df2, _inits):

    df = df1.copy()
    for interval in _inits:
        for parameter in df1.index:
            if parameter not in df2.columns:
                continue
            else:
                idx_ = (
                    (df2['time'] == interval) 
                    & (df2['initialization'] == _inits[interval])
                )
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

    # file_name = f'{resource}_{method}_{aggregation}-hyper-base.csv'
    # hyper_base = pd.read_csv(path_to_validation / f"{file_name}")
    # hyper_base = hyper_base.set_index("parameter")
    # hyper_base.columns = hyper_base.columns.astype(int)
    # #print(hyper_base)
    
    file_name = f'{resource}_{method}_{aggregation}-hyper-{exp_description}.csv'
    
    hyper_val = pd.read_csv(path_to_validation / f"{resource}/{file_name}")

    file_name = f'{resource}_{method}_{aggregation}-envelope-{exp_description}.csv'
    
    envelope_ = pd.read_csv(path_to_validation / f"{resource}/{file_name}")

    envelope_df = []
    hyper_df = []

    for interval in _inits:
        idx = (
            (envelope_['time'] == interval)
            & (envelope_['iteration'] == _inits[interval])
        )
        envelope_df.append(envelope_.loc[idx])

        idx = (
            (hyper_val['time'] == interval)
            & (hyper_val['initialization'] == _inits[interval])
        )
        hyper_df.append(hyper_val.loc[idx])

    hyper_df = pd.concat(
        hyper_df, axis = 0
    ).reset_index(drop = True)
    
    hyper_df = hyper_df[
        [hyper_df.columns[1]] + list(hyper_df.columns[14:])
    ]
    
    hyper_df = hyper_df.set_index("time").T

    envelope_df = pd.concat(
        envelope_df, axis = 0
    ).reset_index(drop = True)

    envelope_df = envelope_df.set_index("time")

    return hyper_df, envelope_df
    
def _get_envelope_table(_init, hyper_, envelope_, score, PATH,
        alphas = [0.1, 0.2]
    ):
        
    envelope_ = envelope_.reset_index(drop = False)
    times = np.array(hyper_.columns)

    envelope_ = pd.concat(
        [envelope_.loc[envelope_['alpha'] == alpha] 
         for alpha in alphas], axis = 0
    )

    fraction_table_ = (
        envelope_
        .pivot_table(
            index=["alpha", "distance"],
            columns="time",
            values="fraction",
            aggfunc="mean"
        )
        .reset_index()
    )

    fraction_table_.columns.name = None

    # --------------------------------------------------
    # 2. Add ECDF rows for scores only.
    #    ECDF has no fraction, so time columns remain empty.
    # --------------------------------------------------
    ecdf_ = envelope_.loc[envelope_["distance"] == "ECDF"].copy()

    score_rows_ = pd.concat(
        [
            envelope_[["time", "alpha", "distance", "FCS", "FIS", "SCP"]],
            ecdf_[["time", "alpha", "distance", "FCS", "FIS", "SCP"]]
        ], axis=0, ignore_index=True
    )

    avg_scores_ = (
        score_rows_
        .groupby(["alpha", "distance"], as_index=False)
        .agg({
            "FCS": "mean",
            "FIS": "mean",
            "SCP": "mean"
        })
    )

    # Use avg_scores_ as the base so ECDF rows are preserved
    summary_ = (
        avg_scores_
        .merge(fraction_table_, on=["alpha", "distance"], how="left")
    )

    # Put time columns before score columns
    summary_ = summary_[
        ["alpha", "distance"] + list(times) + ["FCS", "FIS", "SCP"]
    ]

    summary_ = summary_.round(2)

    # Set row index for LaTeX
    summary_ = summary_.set_index(["alpha", "distance"])

    summary_.index.names = [
        r"$\boldsymbol{\alpha}$",
        r"\textbf{Method}"
    ]

    # MultiIndex columns
    time_cols = [(r"\textbf{Time}", t) for t in times]
    score_cols = [
        (r"\textbf{Score}", "FCS"),
        (r"\textbf{Score}", "FIS"),
        (r"\textbf{Score}", "SCP")
    ]

    summary_.columns = pd.MultiIndex.from_tuples(time_cols + score_cols)

    return summary_


def envelope_table(
    _init, hyper_, envelope_, path_to_validation
    ):
        
    FCS_ = _get_envelope_table(
        _init,
        hyper_, 
        envelope_, 
        score = 'FCS',
        PATH = path_to_validation,
    )
    
    FIS_ = _get_envelope_table(
        _init,
        hyper_, 
        envelope_, 
        score = 'FIS',
        PATH = path_to_validation,
    )
    
    # Add top-level criterion headers
    FIS_.columns = pd.MultiIndex.from_tuples(
        [(r'\textbf{FIS criterion}', *col) 
         for col in FIS_.columns]
    )
    
    FCS_.columns = pd.MultiIndex.from_tuples(
        [(r'\textbf{FCS criterion}', *col) 
         for col in FCS_.columns]
    )
    
    # Concatenate tables to the right
    summary_ = pd.concat([FIS_, FCS_], axis=1)
    # print(summary_)
    
    # Generate latex code
    latex_ = summary_.to_latex(
        float_format="%.2f",
        multirow=True,
        multicolumn=True,
        column_format="l|l|ccc|ccc|ccc|ccc",
        escape=False
    )
    
    latex_ = latex_.replace('l2', r'$\ell_2$')
    latex_ = latex_.replace('F', r'')
    latex_ = latex_.replace('fknn', r'$\omega_i (f_\star, e_\star)$')
    latex_ = latex_.replace('sup', r'$\sup$')
    
    return summary_, latex_

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
        pd.DataFrame({"time": dt_ + 5}).time, unit = "m"
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


# load dataset
def processed_zone_dataset(zone, 
                           path_to_data, 
                           N_test_samples, 
                           unbiased = False):
    #print(region, N_test_samples)

    # Load dataset
    with open(path_to_data, "rb") as f:
        _DATA = pkl.load(f)
        
    _dataset = _DATA['Dataset']
    
    regions_ = list(_DATA['Graph'].keys())
    region = regions_[zone]

    dt_tr_ = _dataset[region]['datetime_actuals'][:-N_test_samples, ...]
    dt_ts_ = _dataset[region]['datetime_actuals'][-N_test_samples:, ...]
    #print(dt_tr_.shape, dt_ts_.shape)
    
    d_tr_ = _dataset[region]['datetime_actuals'][:-N_test_samples, 0, 0, 24]
    d_ts_ = _dataset[region]['datetime_actuals'][-N_test_samples:, 0, 0, 24]
    d_tr_ = pd.to_datetime(d_tr_).dayofyear.to_numpy()
    d_ts_ = pd.to_datetime(d_ts_).dayofyear.to_numpy()
    #print(d_tr_.shape, d_ts_.shape)

    datetime_tr_ = _dataset[region]['datetime_actuals'][:-N_test_samples, ...]
    datetime_ts_ = _dataset[region]['datetime_actuals'][-N_test_samples:, ...]
    #print(datetime_tr_.shape, datetime_ts_.shape)

    F_tr_ = _dataset[region]['actuals'][:-N_test_samples, ...] 
    F_ts_ = _dataset[region]['actuals'][-N_test_samples:, ...]
    #print(F_tr_.shape, F_ts_.shape)

    if unbiased:
        E_tr_ = _dataset[region]['unbiased_forecasts'][:-N_test_samples, ...] 
        E_ts_ = _dataset[region]['unbiased_forecasts'][-N_test_samples:, ...]
    else:
        E_tr_ = _dataset[region]['forecasts'][:-N_test_samples, ...] 
        E_ts_ = _dataset[region]['forecasts'][-N_test_samples:, ...]
    #print(E_tr_.shape, E_ts_.shape)

    X_tr_ = _dataset[region]['neighbors'][:-N_test_samples, ...]
    X_ts_ = _dataset[region]['neighbors'][-N_test_samples:, ...]
    #print(X_tr_.shape, X_ts_.shape)

    # Formating 
    idx_  = X_ts_[..., 1] == 0.
    F_ts_ = F_ts_[idx_, ...]
    E_ts_ = E_ts_[idx_, ...]
    X_ts_ = X_ts_[idx_, ...]
    datetime_ts_ = datetime_ts_[idx_, ...]
    #print(F_ts_.shape, E_ts_.shape, d_ts_.shape, X_ts_.shape, datetime_ts_.shape)

    F_tr_ = np.concatenate([F_tr_[:, i, ...] for i in range(F_tr_.shape[1])], axis = 0)
    E_tr_ = np.concatenate([E_tr_[:, i, ...] for i in range(E_tr_.shape[1])], axis = 0)
    d_tr_ = np.concatenate([d_tr_ for i in range(X_tr_.shape[1])], axis = 0)
    X_tr_ = np.concatenate([X_tr_[:, i, ...] for i in range(X_tr_.shape[1])], axis = 0)
    datetime_tr_ = np.concatenate([datetime_tr_[:, i, ...] for i in range(datetime_tr_.shape[1])], axis = 0)
    #print(E_tr_.shape, E_tr_.shape, d_tr_.shape, X_tr_.shape, datetime_tr_.shape)

    dt_ = np.linspace(0, dt_tr_.shape[-1] - 1, dt_ts_.shape[-1])*12*5
    #print(dt_.shape)
    
    return (F_tr_, F_ts_, 
            E_tr_, E_ts_, 
            d_tr_, d_ts_, 
            datetime_tr_, datetime_ts_, 
            X_tr_, X_ts_, 
            dt_, regions_, region)

def get_stats(
    _init, 
    resource, 
    method, 
    aggregation, 
    exp_description, 
    T, 
    PATH, 
    agg="max",
    ):
    
    """
    Load STATS files for each interval, select the optimal initialization,
    aggregate across horizon columns, and merge all intervals by region/day.

    agg options: "max", "mean", "median"
    """
    agg_funcs = {"max": np.max,
                 "mean": np.mean,
                 "median": np.median}

    stats_list_ = []
    funcs_list_ = []
    for interval in _init:
        
        stats_ = pd.read_csv(
            PATH / f"{resource}/{resource}_{method}_{aggregation}-STATS_{interval}-{exp_description}.csv"
        ).rename(columns={"asset": "region"})        

        funcs_ = pd.read_csv(
            PATH / f"{resource}/{resource}_{method}_{aggregation}-functions_{interval}-{exp_description}.csv"
        ).rename(columns={"asset": "region"})        

        stats_ = stats_.loc[
            stats_["initialization"] == _init[interval]
        ].reset_index(drop=True)

        funcs_ = funcs_.loc[
            (funcs_["initialization"] == _init[interval]) & 
            (funcs_["type"] == "median")
        ].reset_index(drop=True)
        
        # Aggregate over the first T - interval columns
        stats_[interval] = agg_funcs[agg](
            stats_[stats_.columns[:(T - interval)]].to_numpy(), axis=1
        )
        
        # Aggregate over the first T - interval columns
        funcs_[interval] = agg_funcs[agg](
            funcs_[funcs_.columns[:(T - interval)]].to_numpy(), axis=1
        )
        
        stats_ = stats_[["region", "day", interval]].copy()
        funcs_ = funcs_[["region", "day", interval]].copy()

        stats_["region"] = stats_["region"].astype(int)
        stats_["day"] = stats_["day"].astype(int)

        funcs_["region"] = funcs_["region"].astype(int)
        funcs_["day"] = funcs_["day"].astype(int)

        stats_list_.append(stats_)
        funcs_list_.append(funcs_)

    stats_ = stats_list_[0]
    funcs_ = funcs_list_[0]

    for df_stats_, df_funcs_ in zip(stats_list_[1:], funcs_list_[1:]):
        stats_ = stats_.merge(df_stats_, on=["region", "day"], how="left")
        funcs_ = funcs_.merge(df_funcs_, on=["region", "day"], how="left")

    return stats_, funcs_

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