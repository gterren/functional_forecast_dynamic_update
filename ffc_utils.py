import numpy as np

# Calculate weighted (w_) distance between X_ and x_
def _euclidian_dist(X_, x_, w_=[]):
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

# Define exponential growth function
def _exponential_growth(t, growth_rate):
    tau_ = np.linspace(t - 1, 0, t)
    phi_ = np.exp(np.log(0.5)*tau_/(growth_rate*12))
    return phi_

# Define exponential dacay function
def _exponential_decay(S, decay_rate):
    s_   = np.linspace(0, S - 1, S)
    psi_ = np.exp(np.log(0.5)*s_/(decay_rate*12))
    return psi_    

def _logistic(x_, k):
    return 1. - 1.0 / (1.0 + np.exp(np.log(999) * x_ / (k*60/2)))

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
    
    theta = np.sin(dlat_/2)**2 + np.cos(np.deg2rad(x_1_[1]))*np.cos(np.deg2rad(x_2_[:, 1]))*np.sin(dlon_/2)**2
    
    return 2.*R*np.arcsin(np.sqrt(theta))

# Define a function to calculate quantiles
def _KDE_quantile(_KDE, q_, x_min     = 0., 
                            x_max     = 1., 
                            n_samples = 1000):
    
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
    #z_ = np.exp(_KDE.score_samples(x_[:, np.newaxis]))
    w_ = np.cumsum(np.exp(_KDE.score_samples(x_[:, np.newaxis])))
    # Normalize CDF
    w_ /= w_[-1] 
    
    return np.interp(np.array(q_), w_, x_), np.interp(1. - np.array(q_), w_, x_)

# Silverman's Rule
def _silverman_rule(x_):
    IQR = np.percentile(x_, 75) - np.percentile(x_, 25)
    return 0.9 * min(np.std(x_), IQR / 1.34) * x_.shape[0] ** (-1 / 5)

# Periodic distance to rank samples by day of the year
def _periodic_dist(d, gamma, 
                   day_to_degree = 360/365, 
                   degree_to_rad = np.pi/180):
    
    return np.sin(0.5*day_to_degree*(d - gamma)*degree_to_rad)**2

# Filtering scenarios when they are above the upper threshold or below the lower threshold
def _scenario_filtering(W_, d_h_, d_p_, xi, Gamma, kappa_min, kappa_max):

    status = 0
    sigma  = 0

    # Similarity ranking
    idx_rank_ = np.argmin(W_, axis=0)

    # Similarity filter
    w_ = np.min(W_, axis=0)
    idx_bool_ = w_ >= xi

    # Index from selected scenarios
    idx_1_ = np.arange(w_.shape[0])[idx_bool_]
    
    # Filter by temporal distance
    if idx_bool_.sum() > kappa_max:
        #print("(1) Filtering by date: ")
        idx_bool_p_ = idx_bool_ & (d_p_ <= Gamma)

        if idx_bool_p_.sum() > kappa_min:
            status = 2
            idx_bool_ = idx_bool_p_.copy()
        else:
            status = 0
            Gamma  = 0
            #print(" Bypass filtering by date: ")
    else:
        Gamma = 0

    idx_2_ = np.arange(w_.shape[0])[idx_bool_]

    # Filter by spatial distance
    if idx_bool_.sum() > kappa_max:
        #print("(2) Filtering by distance: ")
        status += 1
        for i in range(1000):
            sigma       = np.sort(d_h_[idx_bool_])[kappa_max + i]
            idx_bool_p_ = idx_bool_ & (d_h_ <= sigma)
            
            if idx_bool_p_.sum() > kappa_min:
                idx_bool_ = idx_bool_p_.copy()
                break

    idx_3_ = np.arange(w_.shape[0])[idx_bool_]

    if idx_bool_.sum() < kappa_min:
        #print("Increasing similarity threshold: ")
        status    = 4
        idx_bool_ = w_ >= np.sort(w_)[::-1][kappa_min]

    idx_4_ = np.arange(w_.shape[0])[idx_bool_]
    print(idx_1_.shape, idx_2_.shape, idx_3_.shape,  idx_4_.shape)

    return w_, idx_1_, idx_2_, idx_3_, idx_4_, Gamma, sigma, status

def _fknn_forecast_dynamic_update(F_tr_, E_tr_, x_tr_, t_tr_, dt_, f_, e_, x_, t_ts,
                                  forget_rate_f  = 1.,
                                  forget_rate_e  = .5,
                                  length_scale_f = .1,
                                  length_scale_e = .75,
                                  lookup_rate    = .05,
                                  trust_rate     = 0.0175,
                                  nu             = 340,
                                  gamma          = 30,
                                  xi             = 0.99,
                                  kappa_min      = 100,
                                  kappa_max      = 1000):

    # Get constants
    T    = E_tr_.shape[1]
    t    = f_.shape[0]
    tau_ = dt_[:t]
    s_   = dt_[t:]

    # phi: importance weights based on past time distance
    phi_ = _exponential_growth(t, forget_rate_f)
    # psi: importance weights based on past and future time distance
    psi_1_ = _exponential_growth(t, forget_rate_e)
    psi_2_ = _exponential_decay(T - t, lookup_rate)
    psi_   = np.concatenate([psi_1_, psi_2_], axis = 0)

    # d: Euclidean distance between samples weighted by importance weights
    d_f_ = _euclidian_dist(F_tr_[:, :t], f_, w_ = phi_)
    d_e_ = _euclidian_dist(E_tr_, e_, w_ = psi_)
    d_h_ = _haversine_dist(x_, x_tr_)
    d_p_ = _periodic_dist(t_tr_, t_ts)
    # print(x_tr_.shape, x_ts_.shape, d_s_.shape)
    # print(t_tr_.shape, t_ts_.shape, d_t_.shape)

    # w: normalized weights distance across observations based exponential link function
    w_f_ = _rbf_kernel(d_f_, length_scale_f)
    w_e_ = _rbf_kernel(d_e_, length_scale_e)
    W_   = np.stack([w_f_, w_e_])

    Gamma = _periodic_dist(t_ts, t_ts + gamma)

    (w_, 
    idx_1_, 
    idx_2_, 
    idx_3_, 
    idx_4_, 
    Gamma,
    sigma, 
    status) = _scenario_filtering(W_, d_h_, d_p_, xi, Gamma, kappa_min, kappa_max)

    eta_ = _logistic(s_ - t*5 - nu*60., trust_rate)

    # Fuse scenarios with day-ahead forecasts
    M_ = np.zeros((idx_4_.shape[0], eta_.shape[0]))
    for i, j in zip(idx_4_, range(idx_4_.shape[0])):
        M_[j, :] = F_tr_[i, t:]*(1. - eta_) + E_tr_[i, t:]*eta_

    _meta = {
        'phi': phi_,
        'psi': psi_,
        'eta': eta_,
        'd_f': d_f_,
        'd_e': d_e_,
        'd_h': d_h_,
        'd_p': d_p_,
        'w_f': w_f_,
        'w_e': w_e_,
        'w': w_,
        'idx_1': idx_1_,
        'idx_2': idx_2_,
        'idx_3': idx_3_,
        'idx_4': idx_4_,
        'xi': xi,
        't_ts': t_ts,
        'Gamma': Gamma,
        'gamma': gamma,
        'sigma': sigma,
    }

    return _meta, M_, status
