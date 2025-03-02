library(fdapace)
library(reticulate)

pickle = import("pickle")

path_to_data = '/Users/Guille/Desktop/dynamic_update/data'

sources_ = c("load", "solar", "wind")

# SimDat_20180613
# SimDat_20180911
# SimDat_20181126
# SimDat_20180702
# SimDat_20180722
# SimDat_20180718

simulation = 'SimDat_20180718'

T_fc_event = 12

load_files = function(main_dir, date) {
  dir_list   = list.dirs(main_dir, recursive = FALSE) 
  list_files = list.files(path       = main_dir,
                          pattern    = "*.csv", 
                          full.names = T) 
  N = length(list_files)
  y_ac_ = list("list", N)
  y_fc_ = list("list", N)
  Y_sc_ = list("list", N)
  name_ = list("list", N)
  date = sub(date, pattern = "SimDat_", replacement = "")
  for (i in 1:N) {
    data_ = read.csv(list_files[i], sep = ',', header = FALSE)
    name_[[i]] = sub(sub(".*/", "", list_files[i]), pattern = ".csv$", replacement = "")
    #name_[[i]] = gsub(name, pattern = "_", replacement = " ")
    y_ac_[[i]] = matrix(as.numeric(unlist(data_[2, -c(1:2)])), ncol = 1, nrow = 24)
    y_fc_[[i]] = matrix(as.numeric(unlist(data_[3, -c(1:2)])), ncol = 1, nrow = 24)
    Y_sc_[[i]] = matrix(as.numeric(unlist(data_[-c(1:3),-c(1:2)])), ncol = 24, nrow = 1000)
  }
  return(list(y_ac_, y_fc_, Y_sc_, name_, date))
}

N_sources  = length(sources_) 

for (i in 1:N_sources) {
  
  source = sources_[[i]]
  dir    = paste(path_to_data, simulation, sep = '/')
  file   = paste(dir, source, sep = '/')
  print(file)
  
  # Load Data from Assets in a folder
  data_ = load_files(file, simulation)
  
  # unpack Data
  y_ac_ = data_[[1]]
  y_fc_ = data_[[2]]
  Y_sc_ = data_[[3]]
  name_ = data_[[4]]
  date  = data_[[5]]
  #print(name_)
  #print(date))
  
  N_assets   = length(Y_sc_) 
  N_features = T_fc_event
  N_samples_tr  = dim(Y_sc_[[1]])[1]
  N_samples_ts  = dim(y_ac_[[1]])[2]
  print(N_assets)
  print(N_features)
  print(N_samples_tr)
  print(N_samples_ts)
  
  # Define the continuum
  t_ = seq(1, N_features, length.out = N_features)

  for (j in 1:N_assets) {
    print(rep(1:N_samples_ts, each = N_features))
    print(rep(t_, N_samples_tr))
    exit()
    
    # Perform fPCA
    train = MakeFPCAInputs(IDs  = rep(1:N_samples_tr, each = N_features), 
                           tVec = rep(t_, N_samples_tr), t(Y_sc_[[j]][,1:T_fc_event]))
    
    fPCA  = FPCA(train$Ly, train$Lt)
    test = MakeFPCAInputs(IDs  = rep(1:N_samples_ts, each = N_features), 
                          tVec = rep(t_, N_samples_ts), y_ac_[[j]][1:T_fc_event, ])
    
    pred = predict(fPCA, test$Ly, test$Lt)
    
    # Get Results
    res_      = list("list", 5)
    res_[[1]] = fPCA$mu
    res_[[2]] = fPCA$phi
    res_[[3]] = fPCA$xiEst
    res_[[4]] = pred$scores
    res_[[5]] = pred$predCurves
    
    # Save fPCA Mean, Components and Scores
    path_to_folder = paste(dir, paste('fPCA_sc_', T_fc_event, 'hr', sep = ''), sep = '/')
    dir.create(path_to_folder)
    path_to_subfolder = paste(path_to_folder, source, sep = '/')
    dir.create(path_to_subfolder)
    file_name = paste(path_to_subfolder, paste(name_[[j]], '_sc', '.pkl', sep = ''), sep = '/')
    py_save_object(res_, file_name, pickle = "pickle")
    print(file_name)
  }
}

