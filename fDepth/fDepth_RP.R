#!/usr/bin/Rscript

print('Running fDetph_RP.R ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/software/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Random Projection depth
RP_ = depth.RP(fdata(curves_))$dep

# Save functional depth scores
write.table(RP_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')
print('...end running')
