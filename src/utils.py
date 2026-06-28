import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from statsmodels.distributions.empirical_distribution import ECDF

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
    
# Energy Score (ES) for an ensemble forecast.
def _energy_score(Y_ensamble_, y_, beta = 1.):
    """
    Parameters
    ----------
    Y_ensamble_ : (m, d) array
        m scenarios in d dimensions.
    y_ : (d,) array
        observation vector.
    beta : float
        exponent in (0, 2]. beta=1 is standard energy score.
        Uses ||x - y||^beta.

    Returns
    -------
    float
        Energy score (lower is better).
    """

    m, d = Y_ensamble_.shape

    # term1 = mean ||Xi - y||^beta
    term1 = np.mean(np.linalg.norm(Y_ensamble_ - y_, axis=1) ** beta)

    # term2 = 0.5 * mean_{i,j} ||Xi - Xj||^beta
    # compute pairwise distances efficiently via broadcasting
    X_ = Y_ensamble_[:, None, :] - Y_ensamble_[None, :, :]

    term2 = 0.5 * np.mean(np.linalg.norm(X_, axis=2) ** beta)

    return term1 - term2

# Empirical Probability Integral Transform (PIT) for an ensemble forecast.
def _empirical_PIT(y_obs, Y_ens, seed=1234):
    """
    Parameters
    ----------
    obs:  shape (N,)
    preds: shape (N, M) — row i are forecast samples for case i
    returns: array (N,) with PIT values in (0,1)
    """
    y_obs = np.around(np.asarray(y_obs), 4)
    Y_ens = np.around(np.asarray(Y_ens), 4)
    N, M  = Y_ens.shape
    u_ = np.zeros((N, ))

    # Create a new random number generator
    rng = np.random.default_rng(seed)

    for i in range(N):
        _right_ecdf = ECDF(Y_ens[i, :], side='right')
        _left_ecdf  = ECDF(Y_ens[i, :], side='left')

        u_[i,]  = _right_ecdf(y_obs[i])  
        u_[i,] -= (_right_ecdf(y_obs[i]) - _left_ecdf(y_obs[i]))*rng.random()

    return u_


# Calculate the interval score (IS) for probabilistic forecasts with an interval [lower, upper].
def _interval_score(y_true_, lower_, upper_, alpha):
    """
    Parameters:
    - y_true: Observed (true) values
    - y_pred_upper: upper confidence interval for significance level alpha
    - y_pred_lower: low confidence interval for significance level alpha
    - alpha: Significance level (default 0.05 for 90% confidence interval)
    
    Returns:
    - interval_score: The calculated interval score
    """
        
    # Penalty for observation outside the lower bound
    penalty_lower = 2.*np.maximum(0, lower_ - y_true_)/alpha
    
    # Penalty for observation outside the upper bound
    penalty_upper = 2.*np.maximum(0, y_true_ - upper_)/alpha
    
    # Interval width penalty
    penalty_width = upper_ - lower_
    
    # Total interval score
    return penalty_lower + penalty_upper + penalty_width

# Calculate the weighted interval score (WIS) for probabilistic forecasts in multiple intervals
def _weighted_interval_score(y_true, y_pred, _y_pred_lower, _y_pred_upper, alpha_):
    """
    Parameters:
    - y_true: Observed (true) values
    - _y_pred_lower: dictionary with upper confidence interval for all significance levels alpha
    - _y_pred_lower: dictionary with lower confidence interval for all significance levels alpha
    - alpha: all significance level alpha (default 0.05 for 95% confidence interval)
    Returns:
    - WIS: float, the Weighted Interval Score.
    """

    # Calculate the interval score
    w0  = 1/2.
    w_  = np.array(alpha_)/2.
    is_ = np.zeros((y_true.shape[0], w_.shape[0]))
    for i in range(len(alpha_)):
        is_[:, i] = _interval_score(y_true, 
                                    _y_pred_lower[f'{alpha_[i]}'], 
                                    _y_pred_upper[f'{alpha_[i]}'],
                                    alpha_[i])
    
    term0 = 1./(len(alpha_) + 1/2.)
    term1 = w0 * np.absolute(y_true - y_pred)

    for i in range(w_.shape[0]):
        is_[:, i] = w_[i] * is_[:, i]
    term2 = np.sum(is_, axis = 1)
        
    return term0 * (term1 + term2)


# Compute empirical coverage of prediction intervals.
def _coverage_score(y_true_, lower_, upper_):
    """
    Parameters
    ----------
    y_true_ : array-like
        True observed values.
    lower_ : array-like
        Lower bounds of prediction intervals.
    upper_ : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    float
        Proportion of observations where the true value lies within
        the corresponding prediction interval [lower_, upper_].

    Notes
    -----
    Coverage is defined as:
        (# of points where lower_ <= y_true_ <= upper_) / total points
    """
    
    inside = (y_true_ >= lower_) & (y_true_ <= upper_)
    return inside.mean()

# Simultaneous coverage for a single trajectory.
def _simultaneous_coverage(y_true_, lower_, upper_):
    """
    Parameters
    ----------
    y_true_ : array-like
        Observed trajectory/function.

    lower_ : array-like
        Lower prediction band.

    upper_ : array-like
        Upper prediction band.

    Returns
    -------
    int
        1 if the entire trajectory is covered,
        0 otherwise.
    """

    y_true_ = np.asarray(y_true_)
    lower_ = np.asarray(lower_)
    upper_ = np.asarray(upper_)

    inside = (y_true_ >= lower_) & (y_true_ <= upper_)

    return 1.*np.all(inside)

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