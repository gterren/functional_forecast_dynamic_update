#!/usr/bin/Rscript

print('Running fDepth_LD.R ... ')

library(fdaoutlier)
#library(ddalpha)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# L-inf norm 
LD_ = linfinity_depth(curves_)

# Save functional depth scores
write.table(LD_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')
print('...end running')
