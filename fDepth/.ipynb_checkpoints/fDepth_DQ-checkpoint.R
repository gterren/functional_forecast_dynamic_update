#!/usr/bin/Rscript

print('Running fDepth_DQ.R ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Directional Quantile
DQ_95 = directional_quantile(curves_, quantiles = c(0.05, 0.95))
DQ_95 = max(DQ_95) - DQ_95

# Save functional depth scores
write.table(data.frame(DQ_95), paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

print('...end running')
