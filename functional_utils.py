import os, glob, subprocess, datetime

import pandas as pd
import numpy as np
import pickle as pkl

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


# Functional Depths for POD HPC
def _fDepth4POD(X_, depth, path):
    
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