#!/usr/bin/Rscript

print('Running fDetph_ID.R ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Integrated Depth
ID = depth.FM(fdata(curves_))$dep

# Save functional depth scores
write.table(data.frame(ID), paste(path_to_fDepth, 'fDetph.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

print('...end running')
