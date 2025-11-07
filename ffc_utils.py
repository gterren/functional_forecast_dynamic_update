import numpy as np

from skfda.representation.basis import Fourier, BSpline
from skfda import FDataGrid

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
def _scenario_filtering_v2(W_, d_h_, d_p_, xi, Gamma, kappa_min, kappa_max):

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

# Filtering scenarios when they are above the upper threshold or below the lower threshold
def _scenario_filtering(W_, d_h_, d_p_, gamma, xi, kappa_min, kappa_max):

    sigma  = 0
    idx_spatial_  = None
    idx_temporal_ = None
    # Filter by similarity
    idx_          = np.arange(d_p_.shape[0], dtype = int)
    w_            = np.min(W_, axis = 0)
    idx_neigbors_ = idx_[w_ >= xi]
    idx_final_    = idx_neigbors_.copy()

    if idx_neigbors_.shape[0] > kappa_max:

        # Filter by temporal distance
        idx_temporal_ = idx_[d_p_ <= gamma]
        idx_temporal_ = np.intersect1d(idx_neigbors_, idx_temporal_)
        if idx_.shape[0] < kappa_min:
            idx_temporal_ = idx_neigbors_.copy()

        # Filter by spatial distance
        idx_spatial_rank_ = np.argsort(d_h_[idx_temporal_])
        idx_spatial_      = idx_temporal_[idx_spatial_rank_][:kappa_max]
        if idx_spatial_.shape[0] < kappa_min:
            idx_spatial_ = idx_temporal_.copy()
        else:
            sigma = d_h_[idx_spatial_].max()

        idx_final_ = idx_spatial_.copy()

    if idx_neigbors_.shape[0] < kappa_min:
        # Increase similarity threshold
        idx_final_ = idx_[w_ >= np.sort(w_)[::-1][kappa_min - 1]]
        
    return w_, idx_neigbors_, idx_temporal_, idx_spatial_, idx_final_, sigma

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
                                  kappa_min      = 500,
                                  kappa_max      = 1500,
                                  idx_hours_     = False):

    kappa_min = int(kappa_min)
    kappa_max = int(kappa_max)

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

    # Only for solar
    phi_[~idx_hours_[:t]] = 0.
    psi_[~idx_hours_]     = 0.

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
    sigma) = _scenario_filtering(W_, d_h_, d_p_, Gamma, xi, kappa_min, kappa_max)

    #eta_ = _logistic(s_ - t*5 - nu*60., trust_rate)
    eta_ = _LIE(s_[::-1], t, T, nu, trust_rate)

    # Fuse scenarios with day-ahead forecasts
    M_   = np.zeros((idx_4_.shape[0], eta_.shape[0]))
    m_0_ = np.zeros((idx_4_.shape[0], 1))
    for i, j in zip(idx_4_, range(idx_4_.shape[0])):
        M_[j, :] = F_tr_[i, t:]*(1. - eta_) + E_tr_[i, t:]*eta_
        m_0_[j]  = F_tr_[i, t - 1]

    w_p_         = w_[idx_4_]/w_[idx_4_].sum()
    focal_curve_ = M_.T @ w_p_
    
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
        'sigma': sigma,
        'm_0': m_0_,
        'focal_curve': focal_curve_,
    }

    return _meta, M_

def _focal_curve_envelope(_depth, Y_, dt_, dist_, max_iter = 100):
    """
    Envelope algorithm to obtain functional neighbourhoods.

    Parameters
    ----------
    data : dict
        Dictionary with keys:
            - "x": np.ndarray of grid points (n_points,)
            - "y": np.ndarray of function values (n_points, n_curves)
    focal : int or str
        Index (or column name) of the focal curve to envelope.
    plot : bool, optional
        Whether to plot the selected curves in each iteration.
    max_iter : int, optional
        Maximum number of iterations before stopping.

    Returns
    -------
    dict
        Dictionary with key 'Jordered' containing the ordered list
        of selected curve indices.
    """
    
    # Compute depth to find the focal-curve
    _fd_filtered = FDataGrid(data_matrix = Y_.T,
                                   grid_points = dt_)
    
    filtered_depth_ = _depth(_fd_filtered)
    idx_focal       = np.argsort(-filtered_depth_)[0]
    f_              = Y_[:, idx_focal]

    if isinstance(dist_, str):
        # Distances from curves to the focal-curve
        if dist_ == 'sup':
            dist_ = np.max(np.abs(Y_.T - f_), axis = 1)
        elif dist_ == 'l2':
            dist_ = np.sum((Y_.T - f_)**2, axis = 1)
            
    # Initialize
    idx_subsample = []
    idx_          = [i for i in range(Y_.shape[1]) if i != idx_focal]
    iter_depth    = [0]
    iteration     = 0

    while len(idx_) > 1:
        
        # New interation
        iteration += 1
        
        # Sort curves by distance
        idx_sorted_dist_   = [idx_[i] for i in np.argsort(dist_[idx_])]
        idx_iter_subsample = [idx_sorted_dist_[0]]
        idx_candidates     = idx_sorted_dist_[1:]
        # Iterative envelope selection
        remaining_points = set(dt_)
        while remaining_points and idx_candidates:
            idx_next = idx_candidates[0]
            combined = idx_iter_subsample + [idx_next]
            
            # Check if envelops
            sign_ = np.sign(Y_[:, combined].T - f_)
            Ji_   = np.where(np.abs(np.sum(sign_, axis = 0)) < len(combined))[0]
            
            if len(remaining_points - set(dt_[Ji_]) ) == len(remaining_points):
                # Does not envelope
                idx_candidates.pop(0)
            else:
                remaining_points -= set(dt_[Ji_])
                
                idx_iter_subsample.append(idx_next)
                
                idx_candidates = [c for c in idx_candidates if c not in idx_iter_subsample]
                idx_           = [c for c in idx_ if c not in idx_iter_subsample]

        # Compute functional depth 
        _fd_subset = FDataGrid(data_matrix = (Y_[:, [idx_focal] 
                                                 + idx_subsample 
                                                 + idx_iter_subsample]).T,
                               grid_points = dt_)
    
        depth_             = _depth(_fd_subset)
        idx_depth_         = np.argsort(-depth_)
        idx_ordered_depth_ = np.array([idx_focal] + idx_subsample + idx_iter_subsample)[idx_depth_]

        # How deep the new set of curves is?
        depth_percentile = 1 - np.where(idx_ordered_depth_ == idx_focal)[0][0]/(len(idx_ordered_depth_) - 1)
        
        iter_depth.append(depth_percentile)

        # Accept subsample if depth improves
        if max(iter_depth[:-1]) <= iter_depth[-1]:
            idx_subsample.extend(idx_iter_subsample)
            
        # Stop if there is not more candidate curves
        if not idx_candidates:
            break
            
        # Stop if max_iter is reached
        if iteration >= max_iter:
            break

    # Selected curves
    idx_sel_ = [idx_focal] + idx_subsample
    
    # Final selected curves ordered by depth
    _fd_filtered = FDataGrid(data_matrix = Y_[:, idx_sel_].T,
                             grid_points = dt_)
        
    filtered_depth_             = _depth(_fd_filtered)
    idx_filtered_depth_         = np.argsort(-filtered_depth_)
    idx_ordered_filtered_depth_ = np.array(idx_sel_)[idx_filtered_depth_]

    return Y_[:, idx_ordered_filtered_depth_]


# Downsample collection of curvers
def _functional_downsampling(X_, x_, dt_, subsample, n_basis = 20):

    n_samples, n_points = X_.shape

    X_ = np.concatenate([x_, X_], axis = 1)
    # Ensure the length is divisible by subsample
    dt    = dt_[1] - dt_[0]
    t_    = dt_[-n_points-1:]
    t_ds_ = np.linspace(t_[1], t_[-1], int(n_points/subsample))

    # Create an FDataGrid object
    data_ = [X_[i, :] for i in range(n_samples)]
    _fd   = FDataGrid(data_matrix = data_, 
                      grid_points = t_)

    # Interpolate first (useful if data are unevenly spaced or need smoothing)
    _fd_interp = _fd.to_basis(BSpline(n_basis = n_basis))
    M_interp_  = _fd_interp.to_grid(t_)
    M_interp_  = np.stack([M_interp_.data_matrix[i] 
                           for i in range(n_samples)])[..., 0]
    
    # Re-evaluate existing data
    M_interp_ds_ = _fd_interp.to_grid(t_ds_)
    M_interp_ds_ = np.stack([M_interp_ds_.data_matrix[i] 
                             for i in range(n_samples)])[..., 0]

    return M_interp_, M_interp_ds_, t_ds_

# Derive confidence intervals from a functional depth metric
def _functional_confidence_band(J_, k):
    return J_[0, :], np.max(J_[1:k, :], axis = 0), np.min(J_[1:k, :], axis = 0)
