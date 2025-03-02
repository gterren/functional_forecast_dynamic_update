#!/usr/bin/Rscript

print('Running fPCA_train.R ... ')

library(fdapace)
library(data.table)

path_to_fPCA = '/Users/Guille/Desktop/dynamic_update/software/fPCA/'

# Load actual data for a given source
data_tr_ = read.csv(paste(path_to_fPCA, 'curves_train.csv', sep = ''), sep = ',', header = FALSE)

n = dim(data_tr_)[1]
t = dim(data_tr_)[2]

# Define the continuum
t_    = seq(1, t, length.out = t)
IDs_  = rep(1:n, each = t)
tVec_ = rep(t_, n)

# Perform fPCA
train = MakeFPCAInputs(IDs = IDs_, 
                       tVec = tVec_, t(data_tr_))

fPCA  = FPCA(train$Ly, train$Lt)

# Save fPCA Mean, Components and Scores
write.table(fPCA$mu, paste(path_to_fPCA, 'mu.csv', sep = ''), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ',')

write.table(fPCA$phi, paste(path_to_fPCA, 'factor.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')

write.table(fPCA$xiEst, paste(path_to_fPCA, 'loadings.csv', sep = ''), 
            row.names=FALSE, 
            col.names=FALSE, 
            sep = ',')
print('...end running')