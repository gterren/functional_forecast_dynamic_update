#!/usr/bin/Rscript

print('Running fDepth_DQ_multi-quantile.R ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Directional Quantile
DQ_90 = directional_quantile(curves_, quantiles = c(0.05, 0.95))
DQ_80 = directional_quantile(curves_, quantiles = c(0.1, 0.9))
DQ_70 = directional_quantile(curves_, quantiles = c(0.15, 0.85))
DQ_60 = directional_quantile(curves_, quantiles = c(0.2, 0.8))

# Save functional depth scores
write.table(data.frame(DQ_60, DQ_70, DQ_80, DQ_90), 
            paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names = FALSE, 
            sep = ',')

print('...end running')
