library(fdapace)
library(reticulate)
library(tidyverse)
library(dplyr)
library(pracma)

pickle = import("pickle")

path_to_data   = '/Users/Guille/Dropbox/HighLevel-Data/'
path_to_output = '/Users/Guille/Desktop/dynamic_update/data/actuals'

file_name = '_actual_5min_site_'
sources_  = c("load", "solar", "wind")

# Define source and year source in {load, solar, wind} and year in {2017, 2018}
i_source = 1
t_event  = 12

# Load actual data for a given source
source       = sources_[[i_source]]
path_to_file = paste(path_to_data, source, file_name, 2017, '.csv', sep = '')
data_2017_   = read.csv(path_to_file, sep = ',', header = TRUE)
path_to_file = paste(path_to_data, source, file_name, 2018, '.csv', sep = '')
data_2018_   = read.csv(path_to_file, sep = ',', header = TRUE)
data_        = rbind(data_2017_, data_2018_)
names_       = gsub("\\.", "_", colnames(data_))
print(source)

# Define path to output files
path_to_folder = paste(path_to_output, paste('fPCA_ac_', t_event, 'hr', sep = ''), sep = '/')
dir.create(path_to_folder)
path_to_subfolder = paste(path_to_folder, source, sep = '/')
dir.create(path_to_subfolder)

# Pre-processing data
y_     = data_%>%select(-Time)
x_     = matrix(data_$Time)
N_days = dim(y_)[1]/(24*12)

# Define forecasting event hour
T_fc_event = t_event*12
N_assets   = dim(data_)[2] - 1
print(T_fc_event)

print(dim(y_))
print(dim(x_))

# Select random samples without replacement
set.seed(220)
idx_    = 1:N_days
idx_tr_ = sample(x = idx_, size = N_days*.9)
idx_ts_ = idx_[-idx_tr_]
N_tr_samples = length(idx_tr_)
N_ts_samples = length(idx_ts_)
print(N_tr_samples)
print(N_ts_samples)

# Pre-processing timestamps 
X_ = matrix(nrow = N_days, ncol = 12*24)
for (i_day in 1:N_days) {
  X_[i_day,] = x_[(1 + (i_day - 1)*12*24):(i_day*12*24)]
}
dim(X_)

# Save auxiliary data common to all files
aux_ = list('list', 3)
aux_[[1]] = X_
aux_[[2]] = idx_tr_
aux_[[3]] = idx_ts_
path_to_file = paste(path_to_folder, '/aux.pkl', sep = '')
py_save_object(aux_, path_to_file, pickle = 'pickle')
print(path_to_file)

# Loop over available assets
for (i_asset in 1:N_assets) {# Get assets values in matrix form
  Y_ = matrix(nrow = N_days, ncol = 12*24)
  
  for (i_day in 1:N_days) {
    Y_[i_day,] = y_[(1 + (i_day - 1)*12*24):(i_day*12*24), i_asset]
  }
  
  Y_tr_ = Y_[idx_tr_, 1:T_fc_event]
  Y_ts_ = Y_[idx_ts_, 1:T_fc_event]
  X_tr_ = Y_[idx_tr_, ]
  X_ts_ = Y_[idx_ts_, ]
  dim(Y_tr_)
  dim(Y_ts_)
  dim(X_tr_)
  dim(X_ts_)
  
  # Define the continuum
  t_ = seq(1, T_fc_event, length.out = T_fc_event)
  length(t_)
  
  # Perform fPCA
  train = MakeFPCAInputs(IDs  = rep(1:N_tr_samples, each = T_fc_event), 
                         tVec = rep(t_, N_tr_samples), t(Y_tr_))
  test  = MakeFPCAInputs(IDs  = rep(1:N_ts_samples, each = T_fc_event), 
                         tVec = rep(t_, N_ts_samples), t(Y_ts_))
  
  fPCA = FPCA(train$Ly, train$Lt)
  pred = predict(fPCA, test$Ly, test$Lt)

  # Get Results
  res_      = list('list', 5)
  res_[[1]] = fPCA$mu
  res_[[2]] = fPCA$phi
  res_[[3]] = fPCA$xiEst
  res_[[4]] = pred$scores
  res_[[5]] = pred$predCurves
  
  # Save fPCA Mean, Components and Scores
  path_to_file = paste(path_to_subfolder, '/', names_[i_asset + 1], '_ac', '.pkl', sep = '')
  py_save_object(res_, path_to_file, pickle = 'pickle')
  print(path_to_file)
}
