#!/usr/bin/Rscript

print('Running fDepth_MBD.csv ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, 'curves.csv', sep = ''), 
                    sep = ',', 
                    header = FALSE)

# Modified Band Depth
MBD_ = modified_band_depth(curves_)

# Save functional depth scores
write.table(MBD_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')

print('...end running')
