#!/usr/bin/Rscript

print('Running fDepth_Outliergram.csv ... ')

library(fdaoutlier)
library(roahd)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/functional_forecast_dynamic_update/fDepth/'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, 'curves.csv', sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Modified Band Depth
MBD = modified_band_depth(curves_)
MEI = MEI(curves_)

# Save functional depth scores

write.table(data.frame(MBD, MEI), paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            sep = ',')

print('...end running')
