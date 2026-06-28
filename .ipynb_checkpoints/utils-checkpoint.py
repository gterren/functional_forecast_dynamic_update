import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.linear_model import LinearRegression

#from scipy.integrate import quad
#from scipy import interpolate
#from scipy.stats import multivariate_normal, norm
#from scipy.interpolate import make_smoothing_spline

# Interpolate model hyperparameters across intervals
def _get_hyper(hyper_, param, time):
    x_ = hyper_.loc[param].to_numpy()
    time_ = hyper_.loc[param].index.to_numpy()
    return np.interp(time, time_, x_)

# Interpolate samples in bands
def _get_band_fraction(df_, alpha_, dist, interval):
    k_ = []
    for alpha in alpha_:
        idx_ = (df_['alpha'] == alpha) & (df_['distance'] == dist)
        x_ = df_.loc[idx_, 'fraction'].to_numpy()
        interval_ = df_.loc[idx_].index.to_numpy()
        k_.append(np.interp(interval, interval_, x_))
    return np.array(k_)

# Kolmogorov–Smirnov (KS) score
def _KS(pit_):
    pit_ = np.asarray(pit_, dtype=float)
    pit_ = pit_[np.isfinite(pit_)]
    pit_ = np.clip(pit_, 0.0, 1.0)
    return stats.kstest(pit_, 'uniform').statistic

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


    