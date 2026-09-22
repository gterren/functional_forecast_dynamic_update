#!/usr/bin/Rscript

print('Running fDepth_MSplot.csv ... ')

library(fdaoutlier)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

#Dai, W., and Genton, M. G. (2018). Multivariate functional data visualization and outlier detection. Journal of Computational and Graphical Statistics, 27(4), 923-934.
#Dai, W., and Genton, M. G. (2019). Directional outlyingness for multivariate functional data. Computational Statistics & Data Analysis, 131, 50-65.
#Hardin, J., and Rocke, D. M. (2005). The distribution of robust distances. Journal of Computational and Graphical Statistics, 14(4), 928-946.

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, 'curves.csv', sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Modified Band Depth
MSplot  = msplot(curves_)
MS_mean = MSplot$mean_outlyingness
MS_var  = MSplot$var_outlyingness

# Save functional depth scores
write.table(data.frame(MS_mean, MS_var), paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

print('...end running')
