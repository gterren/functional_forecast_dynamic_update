#!/usr/bin/Rscript

print('Running fDepth_ERL.R ... ')

library(fdaoutlier)
library(fda.usc)

path_to_fDepth = '/Users/Guille/Desktop/dynamic_update/software/fDepth/'

file_name = 'curves.csv'

# Load actual data for a given source
curves_ = read.csv(paste(path_to_fDepth, file_name, sep = ''), 
                   sep = ',', 
                   header = FALSE)

# Extreme Rank Length
ERL_= extreme_rank_length(curves_, type = "two_sided")

# Save functional depth scores
write.table(ERL_, paste(path_to_fDepth, 'fDepth.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')
print('...end running')
