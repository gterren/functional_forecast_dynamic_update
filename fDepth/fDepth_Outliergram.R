#!/usr/bin/Rscript

print('Running fDepth_Outliergram.csv ... ')

library(fdaoutlier)
library(roahd)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/software/fDepth/'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, 'curves.csv', sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Modified Band Depth
MBD_ = modified_band_depth(curves_)
MEI_ = MEI(curves_)

# Save functional depth scores
X_ = as.data.frame(list(MBD_, MEI_))
write.table(X_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')

print('...end running')
