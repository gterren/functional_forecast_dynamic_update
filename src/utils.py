import pandas as pd
import numpy as np
import scipy.stats as stats

# Interpolate model hyperparameters across intervals
def get_hyper(hyper_, param, time):
    x_ = hyper_.loc[param].to_numpy()
    time_ = hyper_.loc[param].index.to_numpy()
    return np.interp(time, time_, x_)

# Interpolate samples in bands
def get_band_fraction(df_, alpha_, interval, distance, score):
    k_ = []
    for alpha in alpha_:
        idx_ = ((df_['alpha'] == alpha) 
                & (df_['distance'] == distance)
                & (df_['score'] == score))
   
        x_ = df_.loc[idx_, 'fraction'].to_numpy()
        y_ = df_.loc[idx_].index.to_numpy()
        k_.append(np.interp(interval, y_, x_))
    return np.array(k_)

# Kolmogorov–Smirnov (KS) score
def KS(pit_):
    pit_ = np.asarray(pit_, dtype=float)
    pit_ = pit_[np.isfinite(pit_)]
    pit_ = np.clip(pit_, 0.0, 1.0)
    return stats.kstest(pit_, 'uniform').statistic

def mask_intervals(F_, E_, t_, day, day_window):
    idx_ = np.absolute(t_ - day) < day_window
    return (np.sum(F_[idx_, :], axis=0) + np.sum(E_[idx_, :], axis=0)) > 1.0

