import os, glob, datetime

import pandas as pd
import numpy as np
import scipy.stats as stats

from scipy.stats import multivariate_normal, norm

from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF

# Calculate weighted (w_) distance between X_ and x_
def _euclidian_dist(X_, x_, w_ = []):
    if len(w_) == 0:
        w_ = np.ones(x_.shape)
    w_ = w_ / w_.sum()
    d_ = np.zeros((X_.shape[0],))
    for i in range(X_.shape[0]):
        d_[i] = w_.T @ (X_[i, :] - x_) ** 2
    return d_

# Radial Basis function kernel based on distance (d_)
def _rbf_kernel(d_, length_scale):
    w_ = np.exp(-d_ / length_scale)
    return w_  # /w_.sum()

def _inv_dist(d_, length_scale):
    w_ = 1.0 / (d_ + length_scale)
    return w_  # /w_.sum()

# Define exponential growth function
def _exponential_growth(t, growth_rate):
    tau_ = np.linspace(t - 1, 0, t)
    phi_ = np.exp(np.log(0.5)*tau_/(growth_rate*12))
    return phi_

# Define exponential decay function
def _exponential_decay(S, decay_rate):
    s_   = np.linspace(0, S - 1, S)
    psi_ = np.exp(np.log(0.5)*s_/(decay_rate*12))
    return psi_    

def _logistic(x_, k):
    return 1. - 1.0 / (1.0 + np.exp(np.log(999) * x_ / (k*60/2)))

# Linear Inverse Exponential function
def _LIE(x_, t, T, nu, trust_rate, k = 2.5, alpha = 1.):
    x_ = x_ - T*5 + nu*5*12
    x_ = k*x_/(nu*5*12 - 5)
    y_ = np.where(x_ > 0, -x_, -alpha*(np.exp(x_) - 1))
    y_ = (y_ + k)/(k + alpha)
    return trust_rate*y_

def _haversine_dist(x_1_, x_2_):
    """
    Calculate the distance between two points on Earth using the Haversine formula.

    Args:
        x_1_ (float): Longitude and latitude of the first point in degrees.
        x_2_ (float): Longitude and latitude of the second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers

    dlat_ = np.deg2rad(x_2_[:, 1]) - np.deg2rad(x_1_[1])
    dlon_ = np.deg2rad(x_2_[:, 0]) - np.deg2rad(x_1_[0])

    theta = (np.sin(dlat_ / 2) ** 2
             + np.cos(np.deg2rad(x_1_[1]))
             * np.cos(np.deg2rad(x_2_[:, 1]))
             * np.sin(dlon_ / 2) ** 2)

    return 2.0 * R * np.arcsin(np.sqrt(theta))

# Periodic distance to rank samples by day of the year
def _periodic_dist(d, gamma, 
                   day_to_degree = 360/365, 
                   degree_to_rad = np.pi/180):
    return np.sin(0.5*day_to_degree*(d - gamma)*degree_to_rad)**2

# Define a function to calculate quantiles
def _KDE_quantile(_KDE, q_, x_min=0.0, x_max=1.0, n_samples=1000):
    """
    Calculates the quantile for a given probability using KDE.

    Parameters:
    _KDE: Kernel density estimate object (e.g., from scipy.stats.gaussian_kde).
    q:    Probability value (between 0 and 1) for which to calculate the quantile.

    Returns:
    The quantile value.
    """

    # Calculate CDF
    x_ = np.linspace(x_min, x_max, n_samples)
    # z_ = np.exp(_KDE.score_samples(x_[:, np.newaxis]))
    w_ = np.cumsum(np.exp(_KDE.score_samples(x_[:, np.newaxis])))
    # Normalize CDF
    w_ /= w_[-1]

    return np.interp(np.array(q_), w_, x_), np.interp(1.0 - np.array(q_), w_, x_)

# Silverman's Rule
def _silverman_rule(x_):
    IQR = np.percentile(x_, 75) - np.percentile(x_, 25)
    return 0.9 * min(np.std(x_), IQR / 1.34) * x_.shape[0] ** (-1 / 5)

# Filtering scenarios when they are above the upper threshold or 
# below the lower threshold
def _scenario_filtering(w_, d_h_, xi, kappa_min, kappa_max):

    sigma = 0
    # Filter by similarity
    idx_          = np.arange(w_.shape[0], dtype = int)
    idx_neigbors_ = idx_[w_ >= xi]

    if idx_neigbors_.shape[0] < kappa_min:  
        # Increase similarity threshold 
        idx_spatial_ = idx_[w_ >= np.sort(w_)[::-1][kappa_min]][:kappa_min]

    else: 
        # Rank neigbors by haversine distance
        idx_spatial_rank_ = np.argsort(d_h_[idx_neigbors_])

        # Select the kappa_max closest neibors
        idx_spatial_ = idx_neigbors_[idx_spatial_rank_][:kappa_max]

        # What is the distance thresold?
        sigma = d_h_[idx_spatial_].max()

    return idx_neigbors_, idx_spatial_, sigma

def _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, t_tr_, dt_, f_, e_, x_, t_ts,
                                  forget_rate_f = 1.,
                                  forget_rate_e = .5,
                                  length_scale_f = .1,
                                  length_scale_e = .75,
                                  lookup_rate = .05,
                                  trust_rate = 0.0175,
                                  nu = 340,
                                  gamma = 30,
                                  xi = 0.99,
                                  kappa_min = 500,
                                  kappa_max = 1500,
                                  idx_hours_ = False,     
                                  p_fusion = 0.):

    kappa_min = int(kappa_min)
    kappa_max = int(kappa_max)
    #print(1)

    # Get constants
    T    = E_tr_.shape[1]
    t    = f_.shape[0]
    tau_ = dt_[:t]
    s_   = dt_[t:]
    #print(2)

    # phi: importance weights based on past time distance
    phi_ = _exponential_growth(t, forget_rate_f)
    # psi: importance weights based on past and future time distance
    psi_1_ = _exponential_growth(t, forget_rate_e)
    psi_2_ = _exponential_decay(T - t, lookup_rate)
    psi_   = np.concatenate([psi_1_, psi_2_], axis = 0)
    #eta_ = _logistic(s_ - t*5 - nu*60., trust_rate)
    eta_ = _LIE(s_[::-1], t, T, nu, trust_rate)
    #print(3)

    # Only for solar
    phi_[~idx_hours_[:t]] = 0.
    psi_[~idx_hours_]     = 0.
    #print(4)

    # d: Temporal distance between samples 
    d_p_  = _periodic_dist(t_tr_, t_ts)
    Gamma = _periodic_dist(t_ts, t_ts + gamma)
    idx_temporal_ = np.arange(d_p_.shape[0], dtype = int)[d_p_ <= Gamma]
    d_p_          = d_p_[idx_temporal_]
    #print(5)

    F_tr_p_ = F_tr_[idx_temporal_, :].copy()
    E_tr_p_ = E_tr_[idx_temporal_, :].copy()

    # d: Euclidean similarity distance between samples weighted by importance weights
    d_f_ = _euclidian_dist(F_tr_p_[:, :t], f_, w_ = phi_)
    d_e_ = _euclidian_dist(E_tr_p_, e_, w_ = psi_)
    #print(6)

    # d: Haverise spatial distance between samples
    d_h_ = _haversine_dist(x_, x_tr_[idx_temporal_, :])
    #print(7)

    # w: normalized weights distance across observations based on the exponential link function
    w_f_ = _rbf_kernel(d_f_, length_scale_f)
    w_e_ = _rbf_kernel(d_e_, length_scale_e)
    W_ = np.stack([w_f_, w_e_])
    w_ = np.min(W_, axis = 0)
    #print(8)

    idx_neigbors_, idx_spatial_, sigma = _scenario_filtering(w_, d_h_, xi, kappa_min, kappa_max)
    #print(9)

    # Fuse scenarios with day-ahead forecasts
    M_ = np.zeros((idx_spatial_.shape[0], eta_.shape[0]))
    F_ = np.zeros((idx_spatial_.shape[0], eta_.shape[0]))
    E_ = np.zeros((idx_spatial_.shape[0], eta_.shape[0]))

    m_0_ = np.zeros((idx_spatial_.shape[0], 1))
    eps = 1e-8
    for i, j in zip(idx_spatial_, range(idx_spatial_.shape[0])):

        F_[j, :] = F_tr_p_[i, t:] 
        E_[j, :] = E_tr_p_[i, t:] 
        m_0_[j]  = F_tr_p_[i, t - 1]

        p0 = np.sum(F_[j, :] == 0)/F_[j, :].shape[0]
        pf = p_fusion * (1. - trust_rate) / np.clip(1. - p0, eps, 1. - eps)
        u  = np.random.uniform(0., 1., size=1)[0]

        if u < pf:
            M_[j, :] = F_[j, :]
        else:
            M_[j, :] = F_[j, :] * (1. - eta_)  + E_[j, :] * eta_

    w_p_ = w_[idx_spatial_]/w_[idx_spatial_].sum()
    w_pp_ = w_[idx_spatial_]
    focal_curve_ = M_.T @ w_p_
    
    _meta = {'phi': phi_,
             'psi': psi_,
             'eta': eta_,
             'd_f': d_f_,
             'd_e': d_e_,
             'd_h': d_h_,
             'd_p': d_p_,
             'w_f': w_f_,
             'w_e': w_e_,
             'w': w_,
             'w_p': w_p_,
             'w_pp': w_pp_,
             'idx_temporal': idx_temporal_,
             'idx_neigbors': idx_neigbors_,
             'idx_spatial': idx_spatial_,
             'xi': xi,
             't_ts': t_ts,
             'Gamma': Gamma,
             'sigma': sigma,
             'm_0': m_0_,
             'focal_curve': focal_curve_,
             'F': F_,
             'E': E_}

    return _meta, M_


# Filtering scenarios when they are above the upper threshold or 
# below the lower threshold
def _scenario_filtering_v2(w_, d_h_, d_p_, Gamma, xi, kappa_min, kappa_max):

    sigma = 0
    # Filter by similarity
    idx_ = np.arange(w_.shape[0], dtype = int)
    idx_neigbors_ = idx_[w_ >= xi]
    idx_temporal_ = idx_[idx_neigbors_][d_p_[idx_neigbors_] <= Gamma]
        
    if idx_temporal_.shape[0] < kappa_min:  
        # Increase similarity threshold 
        idx_spatial_ = idx_[w_ >= np.sort(w_)[::-1][kappa_min]][:kappa_min]

    else: 
        # Rank neigbors by haversine distance
        idx_spatial_rank_ = np.argsort(d_h_[idx_temporal_])

        # Select the kappa_max closest neibors
        idx_spatial_ = idx_neigbors_[idx_spatial_rank_][:kappa_max]

        # What is the distance thresold?
        sigma = d_h_[idx_spatial_].max()

    return idx_temporal_, idx_neigbors_, idx_spatial_, sigma
    
def _fknn_forecast_dynamic_update_v2(F_tr_, E_tr_, x_tr_, t_tr_, dt_, f_, e_, x_, t_ts,
                                     forget_rate_f = 1.,
                                     forget_rate_e = .5,
                                     length_scale_f = .1,
                                     length_scale_e = .75,
                                     lookup_rate = .05,
                                     trust_rate = 0.0175,
                                     nu = 340,
                                     gamma = 30,
                                     xi = 0.99,
                                     kappa_min = 500,
                                     kappa_max = 1500,
                                     idx_hours_ = False,     
                                     p_fusion = 0.):

    kappa_min = int(kappa_min)
    kappa_max = int(kappa_max)
    #print(1)

    # Get constants
    T    = E_tr_.shape[1]
    t    = f_.shape[0]
    tau_ = dt_[:t]
    s_   = dt_[t:]
    #print(2)

    # phi: importance weights based on past time distance
    phi_ = _exponential_growth(t, forget_rate_f)
    # psi: importance weights based on past and future time distance
    psi_1_ = _exponential_growth(t, forget_rate_e)
    psi_2_ = _exponential_decay(T - t, lookup_rate)
    psi_   = np.concatenate([psi_1_, psi_2_], axis = 0)
    #eta_ = _logistic(s_ - t*5 - nu*60., trust_rate)
    eta_ = _LIE(s_[::-1], t, T, nu, trust_rate)
    #print(3)

    # Only for solar
    phi_[~idx_hours_[:t]] = 0.
    psi_[~idx_hours_]     = 0.
    #print(4)

    # d: Temporal distance between samples 
    d_p_  = _periodic_dist(t_tr_, t_ts)
    Gamma = _periodic_dist(t_ts, t_ts + gamma)
    #print(5)

    F_tr_p_ = F_tr_.copy()
    E_tr_p_ = E_tr_.copy()
    
    # d: Euclidean similarity distance between samples weighted by importance weights
    d_f_ = _euclidian_dist(F_tr_p_[:, :t], f_, w_ = phi_)
    d_e_ = _euclidian_dist(E_tr_p_, e_, w_ = psi_)
    #print(6)

    # d: Haverise spatial distance between samples
    d_h_ = _haversine_dist(x_, x_tr_)
    #print(7)

    # w: normalized weights distance across observations based on the exponential link function
    w_f_ = _rbf_kernel(d_f_, length_scale_f)
    w_e_ = _rbf_kernel(d_e_, length_scale_e)
    W_ = np.stack([w_f_, w_e_])
    w_ = np.min(W_, axis = 0)
    #print(8)

    (idx_temporal_, 
     idx_neigbors_, 
     idx_spatial_, 
     sigma) = _scenario_filtering_v2(w_, 
                                     d_h_, 
                                     d_p_, 
                                     Gamma, 
                                     xi, 
                                     kappa_min, 
                                     kappa_max)
    #print(9)

    # Fuse scenarios with day-ahead forecasts
    M_ = np.zeros((idx_spatial_.shape[0], eta_.shape[0]))
    F_ = np.zeros((idx_spatial_.shape[0], eta_.shape[0]))
    E_ = np.zeros((idx_spatial_.shape[0], eta_.shape[0]))

    m_0_ = np.zeros((idx_spatial_.shape[0], 1))
    eps = 1e-8
    for i, j in zip(idx_spatial_, range(idx_spatial_.shape[0])):

        F_[j, :] = F_tr_p_[i, t:] 
        E_[j, :] = E_tr_p_[i, t:] 
        m_0_[j]  = F_tr_p_[i, t - 1]

        p0 = np.sum(F_[j, :] == 0)/F_[j, :].shape[0]
        pf = p_fusion * (1. - trust_rate) / np.clip(1. - p0, eps, 1. - eps)
        u  = np.random.uniform(0., 1., size=1)[0]

        if u < pf:
            M_[j, :] = F_[j, :]
        else:
            M_[j, :] = F_[j, :] * (1. - eta_)  + E_[j, :] * eta_

    w_p_ = w_[idx_spatial_]/w_[idx_spatial_].sum()
    w_pp_ = w_[idx_spatial_]
    focal_curve_ = M_.T @ w_p_
    
    _meta = {'phi': phi_,
             'psi': psi_,
             'eta': eta_,
             'd_f': d_f_,
             'd_e': d_e_,
             'd_h': d_h_,
             'd_p': d_p_,
             'w_f': w_f_,
             'w_e': w_e_,
             'w': w_,
             'w_p': w_p_,
             'w_pp': w_pp_,
             'idx_temporal': idx_temporal_,
             'idx_neigbors': idx_neigbors_,
             'idx_spatial': idx_spatial_,
             'xi': xi,
             't_ts': t_ts,
             'Gamma': Gamma,
             'sigma': sigma,
             'm_0': m_0_,
             'focal_curve': focal_curve_,
             'F': F_,
             'E': E_}

    return _meta, M_
