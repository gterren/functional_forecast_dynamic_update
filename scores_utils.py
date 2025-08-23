import os, glob, subprocess, datetime, sys

import pandas as pd
import numpy as np
import pickle as pkl

from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

def _interval_score(y_true, forecast_mean, forecast_std, z, alpha):
    """
    Calculate the interval score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_std: Standard deviation of the predicted distribution
    - alpha: Significance level (default 0.05 for 90% confidence interval)
    
    Returns:
    - interval_score: The calculated interval score
    """
    
    y_pred_lower = forecast_mean - z * forecast_std  # 90% lower bound (using normal quantile)
    y_pred_upper = forecast_mean + z * forecast_std  # 90% upper bound (using normal quantile)
        
    # Penalty for observation outside lower bound
    penalty_lower = 2.*np.maximum(0, y_pred_lower - y_true)/alpha
    
    # Penalty for observation outside upper bound
    penalty_upper = 2.*np.maximum(0, y_true - y_pred_upper)/alpha
    
    # Interval width penalty
    penalty_width = y_pred_upper - y_pred_lower
    
    # Total interval score
    score = penalty_lower + penalty_upper + penalty_width

    return score

def _logarithmic_score(y_true, forecast_mean, forecast_std):
    """
    Calculate the logarithmic score for probabilistic forecasts.
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_std: Standard deviation of the predicted distribution
    
    Returns:
    - log_score: The calculated logarithmic score (negative log likelihood)
    """
    
    # Calculate the probability density at the observed values (using PDF of normal distribution) and 
    # compute the logarithmic score: negative log of the pdf
    
    log_scores_ = [- norm.logpdf(y_true[i], loc = forecast_mean[i], scale = forecast_std[i]) for i in range(y_true.shape[0])]

    return np.array(log_scores_)


def _crps(y_true, forecast_mean, forecast_std):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a probabilistic forecast.
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_std: Standard deviation of the predicted distribution
    
    Returns:
    - crps_score: The calculated CRPS
    """
    
    crps_ = [ps.crps_gaussian(y_true[i], mu=forecast_mean[i], sig=forecast_std[i])for i in range(y_true.shape[0])]
    
    return np.array(crps_)


def _crps_monte_carlo(y_true, forecast_samples):
    """
    Compute the CRPS using Monte Carlo simulation.

    Parameters:
    - y_true: The observed value (a scalar or array of true values).
    - forecast_samples: An array of samples drawn from the forecast distribution.

    Returns:
    - crps_score: The CRPS score for a single observation.
    """

    crps_ = [ps.crps_ensemble(y_true[i], forecast_samples[:, i]) for i in range(y_true.shape[0])]

    return np.array(crps_)

def _pit(y_true, forecast_mean, forecast_std):
    """
    Calculate the Probabily Integral Transform (PIT) for a probabilistic forecast.
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_std: Standard deviation of the predicted distribution
    
    Returns:
    - mean and std: The calculated PIT
    """
            
    u_samples = norm.cdf(y_true, loc=forecast_mean, scale=forecast_std)

    return np.array([u_samples.mean(), u_samples.std()])

def _coverage_score(y_true, forecast_mean, forecast_std, z, alpha):
    """
    Calculate the coverage score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_std: Standard deviation of the predicted distribution
    - z-score: confidence interval z score
    - alpha: Significance level (default 0.05 for 90% confidence interval)
    
    Returns:
    - coverage_score: The calculated interval score
    """
    
    y_pred_lower = forecast_mean - z * forecast_std  # 90% lower bound (using normal quantile)
    y_pred_upper = forecast_mean + z * forecast_std  # 90% upper bound (using normal quantile)
        
    coverage = 0
    for i in range(y_true.shape[0]): 
        if (y_true[i] < y_pred_lower[i]) or (y_true[i] > y_pred_upper[i]):
            coverage += 0
        else:
            coverage += 1
    
    coverage /= y_true.shape[0]

    return coverage #- (1. - 2.*alpha)


def _weighted_interval_score(y_true, forecast_mean, forecast_cov,
                             alpha_ = [0.05, 0.1, 0.2, 0.4, 0.8],
                             z_     = [2.3, 1.96, 1.65, 1.28, 0.84]):
    """
    Calculate the interval score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_cov: Standard deviation of the predicted distribution
    - alpha: Significance level (default 0.05 for 90% confidence interval)
    - z: z-score for a normal distribution 

    Returns:
    - WIS: float, the Weighted Interval Score.
    """
    
    forecast_std = np.sqrt(np.diagonal(forecast_cov))
  
    # Calculate the interval score
    w_  = np.array(alpha_)/2.
    is_ = np.array([_interval_score(y_true, forecast_mean, forecast_std, z, alpha) 
                    for z, alpha in zip(z_, alpha_)])
    term0 = 1./(len(z_) + 1/2.)
    term1 = 1/2. * np.absolute(y_true - forecast_mean)
    
    for k in range(w_.shape[0]):
        is_[k, :] = w_[k] * is_[k, :]
    term2 = np.sum(is_, axis = 0)
        
    return term1 * (term1 + term2)
    
    
def _ks(y_true, forecast_mean, forecast_std, nbins = 100):
    """
    Calculate the Kolmogorov–Smirnov statistic for a normal dist.
    
    Parameters:
    - y_true: Observed (true) values
    - forecast_mean: Mean of the predicted distribution (e.g., mean of the normal distribution)
    - forecast_std: Standard deviation of the predicted distribution
    
    Returns:
    - ks: Kolmogorov–Smirnov statistic
    """
    
    u_samples_  = norm.cdf(y_true, loc = forecast_mean, scale = forecast_std)
    hist_, bin_ = np.histogram(u_samples_, nbins, density=True)
    bins_       = (bin_[:-1] + bin_[1:])/2.
    #r_ = np.cumsum(hist_) - np.cumsum(np.ones(bins_.shape))

    ks = np.sqrt(np.mean((np.cumsum(hist_) - np.cumsum(np.ones(bins_.shape)))**2))/hist_.shape[0]
    
    return ks

def _empirical_coverage_score(y_true, _lower, _upper, alpha_):
    """`
    Calculate the coverage score for probabilistic forecasts with an interval [lower, upper]
    
    Parameters:
    - y_: Observed (true) values
    - lower_: lower confidence bound
    - upper_: upper confidence dound
    
    Returns:
    - coverage_score: The calculated interval score
    """

    def _coverage_score(y_true, lower_, upper_):
        coverage = 0
        for i in range(y_true.shape[0]): 
            if (y_true[i] < lower_[i]) or (y_true[i] > upper_[i]):
                coverage += 0
            else:
                coverage += 1
        return coverage / y_true.shape[0]


    cs_ = np.zeros((len(alpha_),))
    for i in range(len(alpha_)):
        cs_[i] = _coverage_score(y_true, _lower[f'{alpha_[i]}'], _upper[f'{alpha_[i]}'])
    
    return cs_


def _empirical_interval_score(y_true, y_pred_lower, y_pred_upper, alpha):
    """
    Calculate the interval score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - y_pred_upper: upper confidence interval for significance level alpha
    - y_pred_lower: low confidence interval for significance level alpha
    - alpha: Significance level (default 0.05 for 90% confidence interval)
    
    Returns:
    - interval_score: The calculated interval score
    """
        
    # Penalty for observation outside the lower bound
    penalty_lower = 2.*np.maximum(0, y_pred_lower - y_true)/alpha
    
    # Penalty for observation outside the upper bound
    penalty_upper = 2.*np.maximum(0, y_true - y_pred_upper)/alpha
    
    # Interval width penalty
    penalty_width = y_pred_upper - y_pred_lower
    
    # Total interval score
    return penalty_lower + penalty_upper + penalty_width

def _weighted_empirical_interval_score(y_true, y_pred, _y_pred_lower, _y_pred_upper, alpha_):
    """
    Calculate the interval score for probabilistic forecasts with an interval [lower, upper].
    
    Parameters:
    - y_true: Observed (true) values
    - _y_pred_lower: dictionary with upper confidence interval for all significance levels alpha
    - _y_pred_lower: dictionary with lower confidence interval for all significance levels alpha
    - alpha: all significance level alpha (default 0.05 for 90% confidence interval)

    Returns:
    - WIS: float, the Weighted Interval Score.
    """

    # Calculate the interval score
    w0  = 1/2.
    w_  = np.array(alpha_)/2.
    is_ = np.zeros((y_true.shape[0], w_.shape[0]))
    for i in range(len(alpha_)):
        is_[:, i] = _empirical_interval_score(y_true, 
                                              _y_pred_lower[f'{alpha_[i]}'], 
                                              _y_pred_upper[f'{alpha_[i]}'],
                                              alpha_[i])
    
    term0 = 1./(len(alpha_) + 1/2.)
    term1 = w0 * np.absolute(y_true - y_pred)

    for i in range(w_.shape[0]):
        is_[:, i] = w_[i] * is_[:, i]
    term2 = np.sum(is_, axis = 1)
        
    return term0 * (term1 + term2)

def _ignorance_scores(f_ts_, f_ts_hat_, s_ts_hat_):
         
    # Calculate the interval score
    IS975 = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 2.3, 0.05).mean()
    IS95  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.96, 0.1).mean()
    IS90  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.65, 0.2).mean()
    IS80  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.28, 0.4).mean()
    IS60  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 0.84, 0.8).mean()
    
    return pd.DataFrame(np.array([IS975, IS95, IS90, IS80, IS60])[:, np.newaxis].T, columns = ['IS975', 
                                                                                               'IS95', 
                                                                                               'IS90', 
                                                                                               'IS80', 
                                                                                               'IS60'])
    
def _coverages(f_ts_, f_ts_hat_, s_ts_hat_):
         
    CS975 = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 2.3, 0.05)
    CS95  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.96, 0.1)
    CS90  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.65, 0.2)
    CS80  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.28, 0.4)
    CS60  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 0.84, 0.8)
    
    return pd.DataFrame(np.array([CS975, CS95, CS90, CS80, CS60])[:, np.newaxis].T, columns = ['CS975', 
                                                                                               'CS95', 
                                                                                               'CS90', 
                                                                                               'CS80', 
                                                                                               'CS60'])
    
def _proper_scores(f_ts_, f_ts_hat_, s_ts_hat_):

    LogS = _logarithmic_score(f_ts_, f_ts_hat_, s_ts_hat_).sum()
    CRPS = _crps(f_ts_, f_ts_hat_, s_ts_hat_).sum()
    
    return pd.DataFrame(np.array([LogS, CRPS])[:, np.newaxis].T, columns = ['LogS', 
                                                                            'CRPS'])

def _errors(ac_, fc_):
    
    def __rmse(ac_, fc_):
        return np.sqrt(np.mean((ac_ - fc_)**2))

    def __mae(ac_, fc_):
        return np.mean(np.absolute(ac_ - fc_))

    def __mbe(ac_, fc_):
        return np.mean(ac_ - fc_)

    RMSE = __rmse(ac_, fc_)
    MAE  = __mae(ac_, fc_)
    MBE  = __mbe(ac_, fc_)
    
    return pd.DataFrame(np.array([RMSE, MAE, MBE])[:, np.newaxis].T, columns = ['RMSE', 
                                                                                'MAE', 
                                                                                'MBE'])
 
    
# def _evaludate_update(f_ts_, f_ts_hat_, S_ts_hat_, F_scen_):
    
#     s_ts_hat_ = np.sqrt(np.diagonal(S_ts_hat_))
     
#     # Calculate the interval score
#     IS975 = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 2.3, 0.05).mean()
#     IS95  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.96, 0.1).mean()
#     IS90  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.65, 0.2).mean()
#     IS80  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.28, 0.4).mean()
#     IS60  = _interval_score(f_ts_, f_ts_hat_, s_ts_hat_, 0.84, 0.8).mean()
#     IS_   = [IS975, IS95, IS90, IS80, IS60]
#     #print(IS_)

#     CS975 = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 2.3, 0.05)
#     CS95  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.96, 0.1)
#     CS90  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.65, 0.2)
#     CS80  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 1.28, 0.4)
#     CS60  = _coverage_score(f_ts_, f_ts_hat_, s_ts_hat_, 0.84, 0.8)
#     CS_   = [CS975, CS95, CS90, CS80, CS60]
#     #print(CS_)
    
#     WIS = _weighted_interval_score(f_ts_, f_ts_hat_, S_ts_hat_).mean()
#     #print(WIS_)

#     LogS = _logarithmic_score(f_ts_, f_ts_hat_, s_ts_hat_).sum()
#     CRPS = _crps(f_ts_, f_ts_hat_, s_ts_hat_).sum()

#     # CRPS = _crps_monte_carlo(f_ts_, F_scen_).sum()
#     # print(CRPS)

#     PIT_ = _pit(f_ts_, f_ts_hat_, s_ts_hat_)

#     column_names  = ['IS975', 'IS95', 'IS90', 'IS80', 'IS60']
#     column_names += ['CS975', 'CS95', 'CS90', 'CS80', 'CS60']
#     column_names += ['PITmean', 'PITstd', 'LogS', 'CRPS', 'WIS']

#     return pd.DataFrame(np.array(IS_ + CS_ + PIT_ + [LogS, CRPS, WIS])[:, np.newaxis].T, 
#                         columns = column_names).T

# #scores_ = _evaludate_update(f_ts_, f_ts_hat_, S_ts_hat_, F_scen_)