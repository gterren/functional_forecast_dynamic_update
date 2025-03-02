import os, glob, subprocess, datetime

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats
import properscoring as ps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scores import *
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from scipy import interpolate
from ipywidgets import *
from sklearn.linear_model import BayesianRidge, LinearRegression
from scipy.stats import multivariate_normal, norm
from scipy.interpolate import make_smoothing_spline


def _gen_constraint(f_, f_min, f_max):
    f_[f_ < f_min] = f_min
    f_[f_ > f_max] = f_max
    return f_

def _update_rate(N, update_rate = 1):
    x_ = np.linspace(0, N - 1, N)
    if update_rate == 0:
        w_ = np.ones((N,))
    else:
        w_ = np.exp(-x_/update_rate)
    return x_, w_

def _forget_rate(N, forget_rate = 1):
    x_ = np.linspace(0, N - 1, N)
    if forget_rate == 0:
        w_ = np.ones((N,))
    else:
        w_ = np.exp(-x_/forget_rate)

    return x_, w_[::-1]

def _dist(X_, x_, w_):
    d_ = np.zeros((X_.shape[0], ))
    for i in range(X_.shape[0]):
        d_[i] = w_.T @ (X_[i, :] - x_)**2
    return d_

def _kernel(d_, length_scale):
    w_ = np.exp(-d_/length_scale)
    return w_/w_.sum()

def _smoothing(F_, f_, lamdba):
    
    if bool(lamdba):
        x_        = np.linspace(0, f_.shape[0] - 1, f_.shape[0])
        F_smooth_ = F_.copy()
        f_smooth_ = make_smoothing_spline(x_, f_, lam = lamdba)(x_)
        for i in range(F_.shape[0]):
            F_smooth_[i, :] = make_smoothing_spline(x_, F_[i, :], lam = lamdba)(x_)
        F_ = F_smooth_.copy()
        f_ = f_smooth_.copy()
    
    return F_, f_

# Fuse day-ahead forecast with real-time forecast
def _update_forecast(F_ac_, f_hat_, f_fc_, update_rate, viz = False):
    if bool(update_rate):
        z_, eta_  = _update_rate(F_ac_.shape[1], update_rate = update_rate)
        w_        = 1. - eta_/eta_.max()
        f_update_ = f_hat_*(1. - w_) + f_fc_*w_

        if viz:
            plt.figure(figsize = (10, 2))
            plt.title('Update Rate')
            plt.plot(z_, w_)
            plt.ylim(-0.1,1.1)
            plt.show()

            plt.figure(figsize = (10, 2))
            plt.plot(f_hat_, label = 'real-time (fc)')
            plt.plot(f_fc_, label = 'day-ahead (fc)')
            plt.plot(f_update_, label = 'update (fc)')
            plt.ylim(-0.1,)
            plt.legend()
            plt.show()

        return f_update_.copy()
    else:
        return f_fc_
    
    
def _distance(F_, f_, F_ac_, f_fc_, forget_rate  = 1,
                                    update_rate  = 0,
                                    length_scale = 100, 
                                    lamdba       = 0,
                                    smoothing    = 0,
                                    N_scen       = 0, 
                                    viz          = False):

    f_min = F_.min()
    f_max = F_.max()
    
    # Smoothing observed mean and actuals
    if (smoothing == 0) | (smoothing == 2): 
        F_, f_ = _smoothing(F_, f_, lamdba)

    # Calculate forget rate
    z_, phi_ = _forget_rate(f_.shape[0], forget_rate)
    w_       = phi_/phi_.sum()

    if viz:
        plt.figure(figsize = (10, 2))
        plt.title('Forgeting rate')
        plt.plot(z_, phi_)
        plt.ylim(-0.1,)
        plt.show()
    
    d_ = _dist(F_, f_, w_)
    w_ = _kernel(d_, length_scale = length_scale)
    x_ = np.linspace(0, w_.shape[0] - 1, w_.shape[0])

    if viz:
        plt.figure(figsize = (10, 2))
        plt.title('Samples Weights')
        plt.plot(x_, w_)
        plt.show()
        
        _cmap = plt.get_cmap('inferno')
        _norm = plt.Normalize(w_.min(), w_.max())
        c_    = _cmap(_norm(w_))
        idx_  = np.argsort(w_)
        
        plt.figure(figsize = (10, 2))
        for i in range(w_.shape[0]):
            plt.plot(np.arange(F_.shape[1]), F_[idx_[i], :], c  = c_[idx_[i], :], 
                                                             lw = 1)
            plt.plot(np.arange(F_ac_.shape[1]) + F_.shape[1], F_ac_[idx_[i], :], c  = c_[idx_[i], :], 
                                                                                 lw = 1)
        plt.axvline(F_.shape[1], c  = 'lime', 
                                 lw = 2.5)
        plt.show
        
    # Mean function
    f_hat_ = F_ac_.T @ w_ 
    
    # Fuse day-ahead forecast with real-time forecast
    f_hat_ = _update_forecast(F_ac_, f_hat_, f_fc_, update_rate, viz = viz)
    
    # Smoothing unobserved mean and actuals
    if (smoothing == 1) | (smoothing == 2): 
        F_ac_, f_hat_ = _smoothing(F_ac_, f_hat_, lamdba)
    
    # Covariance Function
    F_hat_ = np.repeat(f_hat_[:, np.newaxis], F_ac_.shape[0], axis = 1).T
    S_hat_ = (F_ac_ - F_hat_).T @ np.diag(w_) @ (F_ac_ - F_hat_)
    
    if bool(N_scen):
        # Generate predictive scenarios
        F_scen_ = multivariate_normal(f_hat_, S_hat_, allow_singular = True).rvs(N_scen)
        # Regularize Scenarios
        F_scen_ = _gen_constraint(F_scen_, f_min, f_max)
        return F_scen_

    else:
        # Regularize Mean
        f_hat_  = _gen_constraint(f_hat_, f_min, f_max)
        return f_hat_, S_hat_
