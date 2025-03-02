#!/usr/bin/Rscript

print('Running fDepth_BD.R ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/software/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)
# Band Depth
BD_ = band_depth(curves_)

# Save functional depth scores
write.table(BD_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')
print('...end running')
