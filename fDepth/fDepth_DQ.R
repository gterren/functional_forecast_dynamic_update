#!/usr/bin/Rscript

print('Running fDepth_DQ.R ... ')

library(fdaoutlier)
#library(ddalpha)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/software/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Directional Quantile
DQ_ = directional_quantile(curves_, quantiles = c(0.025, 0.975))

# Save functional depth scores
write.table(DQ_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')
print('...end running')
