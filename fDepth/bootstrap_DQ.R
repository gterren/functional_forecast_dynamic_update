#!/usr/bin/Rscript

print('Running bootstrap_DQ.R with bootstrap confidence bands ...')

library(fdaoutlier)
library(fda.usc)

# Parameters
path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

# Bootstrap procedure
set.seed(123)  # For reproducibility

n_bootstrap_interations = 250 # More iterations for stable bands
n_bootstrap_samples     = 20

# Load original data
curves_ = as.matrix(read.csv(paste0(path_to_fDepth, 'curves.csv'), header = FALSE))
alphas_ = as.matrix(read.csv(paste0(path_to_fDepth, 'bands.csv'), header = FALSE))

n_points = ncol(curves_)
n_curves = nrow(curves_)
n_bands  = nrow(alphas_)

# Store bootstrap samples (mean curve in each bootstrap)
bootstrap_sup_bands_ = array(NA, dim = c(n_bands, n_points))
bootstrap_inf_bands_ = array(NA, dim = c(n_bands, n_points))

for (i in 1:n_bands) {
  
  alpha = alphas_[i,]
  n_selected_curves = n_bootstrap_samples*(1 - alpha)
  
  # Store bootstrap samples (mean curve in each bootstrap)
  bootstrap_sup_ = array(NA, dim = c(n_bootstrap_interations, n_points))
  bootstrap_inf_ = array(NA, dim = c(n_bootstrap_interations, n_points))
  
  for (j in 1:n_bootstrap_interations) {
    
    bootstrap_indices_ = sample(1:n_curves, size = n_bootstrap_samples, replace = FALSE)
    bootstrap_curves_  = curves_[bootstrap_indices_, ]

    depth_scores_   = directional_quantile(bootstrap_curves_, quantiles = c(0.05, 0.95))
    depth_scores_   = max(depth_scores_) - depth_scores_
    deepth_indices_ = as.matrix(order(depth_scores_, decreasing = TRUE)[1:n_selected_curves])
    
    bootstrap_sup_[j,] = apply(bootstrap_curves_[deepth_indices_, ], 2, max)
    bootstrap_inf_[j,] = apply(bootstrap_curves_[deepth_indices_, ], 2, min)
    
  }
  
  # Compute confidence bands
  bootstrap_sup_bands_[i,] = as.matrix(apply(bootstrap_sup_, 2, quantile, 1 - alpha/2.))
  bootstrap_inf_bands_[i,] = as.matrix(apply(bootstrap_inf_, 2, quantile, alpha/2.))
  
}

depth_scores_  = directional_quantile(curves_, quantiles = c(0.05, 0.95))
depth_scores_  = max(depth_scores_) - depth_scores_
deepest_curve_ = curves_[order(depth_scores_, decreasing = TRUE)[1],]

# Save deepest curve
write.table(data.frame(deepest_curve_), paste(path_to_fDepth, 'deepest.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

# Save functional upper conficence bands
write.table(data.frame(bootstrap_sup_bands_), paste(path_to_fDepth, 'sup_bands.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

# Save functional lower conficence bands
write.table(data.frame(bootstrap_inf_bands_), paste(path_to_fDepth, 'inf_bands.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

print('...end running')
