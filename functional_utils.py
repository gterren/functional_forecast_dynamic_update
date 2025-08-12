import os, glob, subprocess, datetime

import pandas as pd
import numpy as np
import pickle as pkl

from statsmodels.distributions.empirical_distribution import ECDF

path_to_fPCA   = '/Users/Guille/Desktop/dynamic_update/software/fPCA'
path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/software/fDepth'
path_to_data   = '/Users/Guille/Desktop/dynamic_update/data'

# Fit Functional PCA
def _fPCA_fit(X_, path):
        
    # Save input data
    X_.to_csv(path + '/curves_train.csv', header = False, index = False)

    # fPCA .R routine
    subprocess.call(['Rscript', path + '/fPCA_train.R'], stdout = subprocess.DEVNULL)

    # Read output data
    mu_  = pd.read_csv(path + '/mu.csv', index_col = None, header = None).to_numpy()
    phi_ = pd.read_csv(path + '/factor.csv', index_col = None, header = None).to_numpy()
    xi_  = pd.read_csv(path + '/loadings.csv', index_col = None, header = None).to_numpy()

    return [mu_, phi_, xi_]

# Fit and Predict Functional PCA
def _fPCA_pred(X_tr_, X_ts_, path):
    
    # Save input data
    X_tr_.to_csv(path + '/curves_train.csv', header = False, index = False)
    X_ts_.to_csv(path + '/curves_test.csv', header = False, index = False)

    # fPCA .R routine
    subprocess.call(['Rscript', path + '/fPCA_test.R'], stdout = subprocess.DEVNULL)
            
    # Read training output data
    mu_  = pd.read_csv(path + '/mu.csv', index_col = None, header = None)
    phi_ = pd.read_csv(path + '/factor.csv', index_col = None, header = None)
    xi_  = pd.read_csv(path + '/loadings.csv', index_col = None, header = None)

    # Read testing output data
    xi_hat_ = pd.read_csv(path + '/pred_loadings.csv', index_col = None, header = None)

    return xi_hat_, [mu_, phi_, xi_]


# Functional Depths 
def _fDepth(X_, depth, path):
    
    # Save input data
    pd.DataFrame(X_).to_csv(path + '/curves.csv', header = False, index = False)

    # Modified Band Depth .R routine
    if depth == 'MBD':   
        subprocess.call(['Rscript', path + '/fDepth_MBD.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    elif depth == 'BD':  
        subprocess.call(['Rscript', path + '/fDepth_BD.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Directional Quantile .R routine
    elif depth == 'DQ':  
        subprocess.call(['Rscript', path + '/fDepth_DQ.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Extreme Rank Length .R routine
    elif depth == 'ERL': 
        subprocess.call(['Rscript', path + '/fDepth_ERL.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Extreme Depth .R routine
    elif depth == 'ED':  
        subprocess.call(['Rscript', path + '/fDepth_ED.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Modal Depth .R routine
    elif depth == 'MD':  
        subprocess.call(['Rscript', path + '/fDepth_MD.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Integrated Depth .R routine
    elif depth == 'ID':  
        subprocess.call(['Rscript', path + '/fDepth_ID.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # L-inf Depth .R routine
    elif depth == 'LD':  
        subprocess.call(['Rscript', path + '/fDepth_LD.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Random Projection Depth .R routine
    elif depth == 'RP':  
        subprocess.call(['Rscript', path + '/fDepth_RP.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Random Tukey Depth .R routine
    elif depth == 'RT':  
        subprocess.call(['Rscript', path + '/fDepth_RT.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Magnitude-Shape Plot (MS Plot): Mean Outlyingness | path Outlyingness .R routine
    elif depth == 'MSplot':  
        subprocess.call(['Rscript', path + '/fDepth_MSplot.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    # Outliergram: Modified Band Depth | Modified Epigraph Index .R routine
    elif depth == 'Outliergram':  
        subprocess.call(['Rscript', path + '/fDepth_Outliergram.R'], 
                        stdout = subprocess.DEVNULL, 
                        stderr = subprocess.STDOUT)
    else:
        print('Does not exist')
        
    # Read output data
    return pd.read_csv(path + '/fDepth.csv', index_col = None)

# Functional Quantiles 
def _fQuantile(X_, path):
    
    # Save input data
    pd.DataFrame(X_).to_csv(path + '/curves.csv', header = False, index = False)

    # Directional Quantile .R routine
    subprocess.call(['Rscript', path + '/fDepth_DQ_multi-quantile.R'], 
                    stdout = subprocess.DEVNULL, 
                    stderr = subprocess.STDOUT)

    # Read output data
    return pd.read_csv(path + '/fDepth.csv', index_col = None)

def _directional_quantile(curves, q_lower=0.025, q_upper=0.975):
    """
    Compute directional quantile outlyingness for each curve.

    Parameters
    ----------
    curves: np.ndarray, shape (n, T)
        Functional data: n curves sampled at T points.
    q_lower: float
        Lower quantile for scaling (default 0.025).
    q_upper: float
        Upper quantile (default 0.975).

    Returns
    -------
    dq : np.ndarray, shape (n,)
        Directional quantile (max scaled deviation) per curve.
    """
    n, T   = curves.shape
    mean_t = np.mean(curves, axis=0)
    lower  = np.quantile(curves, q_lower, axis=0)
    upper  = np.quantile(curves, q_upper, axis=0)

    # Avoid division by zero: float eps
    denom_low = mean_t - lower
    denom_up  = upper - mean_t
    denom_low[denom_low == 0] = np.finfo(float).eps
    denom_up[denom_up == 0]   = np.finfo(float).eps

    # Compute scaled deviations for each curve at each time
    diffs = curves - mean_t  # (n,T)
    scaled = np.where(diffs >= 0,
                      np.abs(diffs) / denom_up,
                      np.abs(diffs) / denom_low)

    return np.max(scaled, axis=1)

def _modified_band_depth(curves):
    """
    Compute Modified Band Depth (J=2) using vectorized NumPy operations.
    
    Parameters:
    -----------
    curves: ndarray, shape (n_samples, n_times)
        Functional data matrix.
    
    Returns:
    --------
    mbd: ndarray, shape (n_samples,)
        MBD scores for each function.
    """
    n, T  = curves.shape
    pairs = np.array([(j, k) for j in range(n) for k in range(j+1, n)])
    lower = np.minimum(curves[pairs[:, 0]][:, None, :], curves[pairs[:, 1]][:, None, :])
    upper = np.maximum(curves[pairs[:, 0]][:, None, :], curves[pairs[:, 1]][:, None, :])
    mbd   = np.zeros(n)
    
    # Check inclusion across all bands and time points
    for i in range(n):
        inside = np.logical_and(lower <= curves[i], curves[i] <= upper)
        mbd[i] = inside.sum() / (len(pairs) * T)
    
    return mbd


# @njit
# def _modified_band_depth(curves):
#     """
#     Fast MBD computation for all curves in the dataset.
    
#     Parameters:
#     - data: numpy array of shape (n_curves, n_timepoints)
    
#     Returns:
#     - mbd: numpy array of MBD values for each curve (length n_curves)
#     """
#     n, m = curves.shape
#     mbd = np.zeros(n)
#     total_pairs = n * (n - 1) / 2.0

#     for k in range(n):  # Parallel loop over curves
#         depth = 0.0
#         for i in range(n):
#             for j in range(i + 1, n):
#                 # Compute pointwise band between curve i and j
#                 band_min = np.minimum(curves[i], curves[j])
#                 band_max = np.maximum(curves[i], curves[j])
#                 # Count how often curve k is within the band
#                 count = 0
#                 for t in range(m):
#                     if band_min[t] <= curves[k][t] <= band_max[t]:
#                         count += 1
#                 depth += count / m
#         mbd[k] = depth / total_pairs
#     return mbd


def _eQuantile(_eCDF, q_):
    """
    Calculates quantiles from an ECDF.

    Args:
    _eCDF: function from statsmodels api
    q_: A list or numpy array of quantiles to calculate (values between 0 and 1).

    Returns:
    _Q: A dictionary where keys are the input quantiles and values are the corresponding
    Quantile values from the ECDF.
    """

    return np.array([_eCDF.x[np.searchsorted(_eCDF.y, q)] for q in q_])

# Derive confidence intervals from Directional Quantiles
def _confidence_intervals_from_DQ(M_, alpha_, zeta_):
    
    _y_pred_upper = {}
    _y_pred_lower = {}

    # Calculate functional Directional Quantiles (DQ)
    DQ_ = _directional_quantile(M_)
    
    for i in range(len(alpha_)):

        I_  = np.argsort(np.absolute(DQ_))[::-1]

        scen_                         = M_[I_[int(M_.shape[0] * zeta_[i]):],]
        _y_pred_upper[f'{alpha_[i]}'] = np.max(scen_, axis = 0)
        _y_pred_lower[f'{alpha_[i]}'] = np.min(scen_, axis = 0)

    m_ = np.median(M_, axis = 0)

    return m_, _y_pred_upper, _y_pred_lower

# Derive confidence intervals from a functional depth metric
def _confidence_intervals_from_MBD(M_, alpha_, zeta_):

    # Calculate Modified Band Depth ranking
    MBD_ = _modified_band_depth(M_)
    I_   = np.argsort(MBD_)
    _y_pred_upper = {}
    _y_pred_lower = {}
    for i in range(len(alpha_)):
        scen_                         =  M_[I_[int(M_.shape[0] * zeta_[i]):],]
        _y_pred_upper[f'{alpha_[i]}'] = np.max(scen_, axis = 0)
        _y_pred_lower[f'{alpha_[i]}'] = np.min(scen_, axis = 0)

    m_ = np.median(M_, axis = 0)

    return m_, _y_pred_upper, _y_pred_lower

# Derive confidence intervals from a functional depth metric
def _confidence_intervals_from_eCDF(M_, alpha_, zeta_):    

    _y_pred_upper = {}
    _y_pred_lower = {}
    for i in range(len(alpha_)):

        _y_pred_lower[f'{alpha_[i]}'] = np.stack([_eQuantile(ECDF(M_[:, j]), [zeta_[i]/2.])
                                                  for j in range(M_.shape[1])])[:, 0]
        _y_pred_upper[f'{alpha_[i]}'] = np.stack([_eQuantile(ECDF(M_[:, j]), [1. - (zeta_[i]/2.)])
                                                  for j in range(M_.shape[1])])[:, 0]

    m_ = np.median(M_, axis = 0)

    return m_, _y_pred_upper, _y_pred_lower


# X_tr_ = pd.DataFrame(scs_)
# X_ts_ = pd.DataFrame(scs_[:11, :])

# print(X_tr_.shape, X_ts_.shape)

# # Fit Functional PCA
# fPCA_ = _fPCA_fit(X_tr_, path_to_fPCA)
# print(fPCA_[0].shape, fPCA_[1].shape, fPCA_[2].shape)

# # Fit and Predict Functional PCA
# xi_hat_, fPCA_ = _fPCA_pred(X_tr_, X_ts_, path_to_fPCA)
# print(xi_hat_.shape)

# # Functional Depths 
# depth_ = _fDepth(X_tr_, 'MSplot', path_to_fDepth)
# print(depth_.shape)


# Fit Functional PCA

#mu_tr_ = np.mean(ac_tr_, axis = 1)
#print(mu_tr_.shape)
# X_tr_ = pd.DataFrame(ac_tr_)
# fPCA_ = _fPCA_fit(X_tr_, path_to_fPCA)
# print(fPCA_[0].shape, fPCA_[1].shape, fPCA_[2].shape)

# plt.figure()
# plt.plot(fPCA_[0])
# plt.show()

# plt.figure()
# plt.plot(fPCA_[1][:, 0])
# plt.plot(fPCA_[1][:, 5])
# plt.plot(fPCA_[1][:, 10])
# plt.plot(fPCA_[1][:, 15])
# plt.plot(fPCA_[1][:, -1])
# plt.show()

# plt.figure()
# plt.plot(fPCA_[2].T)
# plt.show()

# from sklearn.linear_model import LinearRegression

# def _factor_fc(f_hat_, fc_, t_day, trust_rate):

#     x_, y_   = _forget_rate(f_hat_[t_day:], trust_rate)
#     lambda_  = y_/y_[-1]
#     lambda_ *= (288 - t_day)/288
    
#     return lambda_*fc_[t_day:] + (1 - lambda_)*f_hat_[t_day:]

# def _fPCA_fc(N_factors = 22, f_min = 0., f_max = np.inf):
    
#     xi_hat_     = LinearRegression(fit_intercept = False).fit(Phi_tr_, f_tr_).coef_
#     f_fPCA_hat_ = Phi_ts_ @ xi_hat_
#     print(f_.shape, f_fPCA_hat_.shape)
#     f_fun_hat_  = _gen_constraint(f_fun_hat_, f_min, f_max)[t_day:, ]
#     # f_fc_fknn_ = _factor_fc(f_fknn_hat_, fc_, t_day, trust_rate = 100.)
#     return


# path_to_data = '/Users/Guille/Desktop/dynamic_update/data'

# folder = 'SimDat_20180722'

# resource = 'wind'
# i_asset  = 1

# fPCA_ = np.sort(glob.glob(path_to_data + '/' + folder + '/fPCA_sc' + '/' + resource + '/*'))
# sc_   = np.sort(glob.glob(path_to_data + '/' + folder + '/' + resource + '/*'))
# print(fPCA_[i_asset])
# print(sc_[i_asset])


# with open(fPCA_[i_asset], 'rb') as _file:
#     fPCA_ = pkl.load(_file)
    
# mu_       = np.array(fPCA_[0])
# factors_  = fPCA_[1]
# loadings_ = fPCA_[2]
# print(mu_.shape, factors_.shape, loadings_.shape)

# file_sc_ = pd.read_csv(sc_[i_asset])

# ac_  = file_sc_.loc[0, file_sc_.columns[2:]].to_numpy()
# fc_  = file_sc_.loc[1, file_sc_.columns[2:]].to_numpy()
# scs_ = file_sc_.loc[2:, file_sc_.columns[2:]].to_numpy()
# print(ac_.shape, fc_.shape, scs_.shape)

# plt.figure()
# plt.plot(ac_, 'r')
# plt.plot(fc_, 'k')
# plt.plot(mu_, 'b')
# plt.show()

# plt.figure()
# for i in range(factors_.shape[1]):
#     plt.plot(factors_[:, i])
# plt.show()


# def _pred(loadings_, mu_, factors_):
#     return mu_ + loadings_ @ factors_.T

# i_sc = 200

# sc_hat_ = _pred(loadings_[i_sc, :], mu_, factors_)
# sc_     = scs_[i_sc]
# print(sc_.shape, sc_hat_.shape)

# print(loadings_.shape, factors_.shape)

# N_factors = 15
# alpha = 5e3



# _cmap = plt.get_cmap('inferno')


# for i in [2, 8, 12, 16, 24]:
#     print(i)
#     y_  = ac_[:i + 1] - mu_[:i + 1]
#     xi_ = factors_[:i + 1, :N_factors]
    
#     _model = BayesianRidge(tol = 1e-20, fit_intercept = False).fit(xi_, y_)
#     y_hat_, s2_ = _model.predict(factors_[:, :N_factors], return_std = True) + mu_
    
#     xi_hat_ = _model.coef_
    
#     print(loadings_.shape, xi_hat_.shape)

#     dist_ = np.array([loadings_[i, :N_factors] - xi_hat_ for i in range(loadings_.shape[0])])
#     print(dist_.shape) 
#     dist_ = np.exp(-np.sum(dist_**2, axis = 1)/alpha)
#     dist_ = dist_/dist_.sum()
#     print(dist_.min(), dist_.max())
#     # Define a normalization
#     _norm = plt.Normalize(dist_.min(), dist_.max())
#     print(dist_.shape)
#     colors_ = _cmap(_norm(dist_))
#     print(scs_.shape, colors_.shape)
    
#     idx_ = np.argsort(dist_)
#     plt.figure()
#     for i in range(dist_.shape[0]):
#         plt.plot(scs_[idx_[i], :].T, c = colors_[idx_[i], :], lw = 0.5)
#     plt.show
    
# #     plt.figure()
# #     plt.plot(s2_)
# #     plt.show()
    
#     plt.figure()
#     plt.plot(scs_.T, 'gray')
#     plt.plot(y_hat_, 'k')
#     plt.plot(ac_, 'r')
#     plt.plot(mu_, 'b')
#     plt.show()