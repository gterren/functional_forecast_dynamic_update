#!/usr/bin/Rscript

print('Running fPCA_test.R ... ')

library(fdapace)
library(data.table)

path_to_fPCA = '/Users/Guille/Desktop/dynamic_update/software/fPCA/'


# Load actual data for a given source
data_tr_ = read.csv(paste(path_to_fPCA, 'curves_train.csv', sep = ''), 
                    sep = ',', 
                    header = FALSE)

n_tr = dim(data_tr_)[1]
t_tr = dim(data_tr_)[2]

# Define the continuum
t_tr_    = seq(1, t_tr, length.out = t_tr)
IDs_tr_  = rep(1:n_tr, each = t_tr)
tVec_tr_ = rep(t_tr_, n_tr)

# Perform fPCA
train = MakeFPCAInputs(IDs = IDs_tr_, 
                       tVec = tVec_tr_, t(data_tr_))

fPCA  = FPCA(train$Ly, train$Lt)

# Save fPCA Mean, Components and Scores
write.table(fPCA$mu, paste(path_to_fPCA, 'mu.csv', sep = ''), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ',')

write.table(fPCA$phi, paste(path_to_fPCA, 'factors.csv', sep = ''), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ',')

write.table(fPCA$xiEst, paste(path_to_fPCA, 'loadings.csv', sep = ''), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ',')

data_ts_ = read.csv(paste(path_to_fPCA, 'curves_test.csv', sep = ''), 
                    sep = ',', 
                    header = FALSE)

n_ts = dim(data_ts_)[1]
t_ts = dim(data_ts_)[2]

# Define the continuum
t_ts_    = seq(1, t_ts, length.out = t_ts)
IDs_ts_  = rep(1:n_ts, each = t_ts)
tVec_ts_ = rep(t_tr_, n_ts)


test = MakeFPCAInputs(IDs  = IDs_ts_, 
                      tVec = tVec_ts_, t(data_ts_))

pred = predict(fPCA, test$Ly, test$Lt)

write.table(pred$scores, paste(path_to_fPCA, 'pred_loadings.csv', sep = ''), 
            row.names = FALSE, 
            col.names = FALSE, 
            sep = ',')
print('... end running')