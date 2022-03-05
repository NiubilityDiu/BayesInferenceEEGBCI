# rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T')

library(ggplot2)
require(gridExtra)
library(R.matlab)

if (local_use) {
  parent_path = ''
  r_fun_path = 
  data_type = 'test'  # train or test
  design_num = 1
  num_rep_fit = 5
  num_rep_pred = 5
  dataset_num = 5
  scenario_name = 'TrueGen'
  zeta_0 = 0.5
  
} else {
  parent_path = ''
  r_parent_path = ''
  r_fun_path = ''
  data_type = args[2]
  design_num = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID')) + 1
  num_rep_fit = 5
  num_rep_pred = 5
  dataset_num = 100
  scenario_name = args[3]
  zeta_0 = as.numeric(args[4])
}
source(paste(r_fun_path, '/source_files/global_constant.R', sep=""))
source(paste(r_fun_path, '/source_files/sim_fun.R', sep=""))

mean_ids = data.frame(
  TrueGen = c(1, 1, 2, 2, 1, 1, 2, 2),
  LatencyLen25 = c(3, 3, 4, 4, 3, 3, 4, 4),
  LatencyLen35 = c(5, 5, 6, 6, 5, 5, 6, 6)
)
zeta_true_ls = lapply(1:6, function(x) 1 * (mean_fn_tar_ls[[x]] - mean_fn_ntar_ls[[x]] > zeta_0))

if (scenario_name == 'MisLatencyLen25') {
  fn_id = mean_ids$LatencyLen25[design_num]
  mean_fn_tar = mean_fn_tar_ls[[paste('tar_', fn_id, sep="")]]
  mean_fn_ntar = mean_fn_ntar_ls[[paste('ntar_', fn_id, sep="")]]
  mean_fn_tar = c(mean_fn_tar, rep(0, 5))
  mean_fn_ntar = c(mean_fn_ntar, rep(0, 5))
  zeta_true_0 = c(zeta_true_ls[[mean_ids$LatencyLen25[design_num]]], rep(0, 5))
} else if (scenario_name == 'MisLatencyLen35') {
  fn_id = mean_ids$LatencyLen35[design_num]
  mean_fn_tar = mean_fn_tar_ls[[paste('tar_', fn_id, sep="")]][1:n_length_fit]
  mean_fn_ntar = mean_fn_ntar_ls[[paste('ntar_', fn_id, sep="")]][1:n_length_fit]
  zeta_true_0 = zeta_true_ls[[mean_ids$LatencyLen35[design_num]]][1:n_length_fit]
} else {
  fn_id = mean_ids$TrueGen[design_num]
  mean_fn_tar = mean_fn_tar_ls[[paste('tar_', fn_id, sep="")]]
  mean_fn_ntar = mean_fn_ntar_ls[[paste('ntar_', fn_id, sep="")]]
  zeta_true_0 = zeta_true_ls[[mean_ids$TrueGen[design_num]]]
}

parent_path_plot = Sys.glob(
  file.path(parent_path, '*data*', '*SIM_files*', paste('*sim_', design_num, sep=''))
)
dir.create(file.path(parent_path_plot, 'Plots'))

zeta_smgp_binary = zeta_swlda_binary = matrix(0, nrow=dataset_num, ncol=n_length_fit)
iswr_smgp = imwr_smgp = iswr_swlda = imwr_swlda = NULL
for (dataset_id in 1:dataset_num) {
  print(dataset_id)
  # SMGP
  data_path = paste(parent_path_plot, '/sim_', design_num, '_dataset_', dataset_id, sep="")
  smgp_dat_dir_opt = Sys.glob(
    file.path(data_path, '*q2*', scenario_name, paste('*zeta_', zeta_0, '.mat*', sep=""))
  )
  mcmc_dat_opt = readMat(smgp_dat_dir_opt)
  zeta_smgp_id = mcmc_dat_opt$zeta
  zeta_smgp_binary[dataset_id,] = zeta_smgp_id
  
  # swLDA
  swlda_dat_dir = Sys.glob(
    file.path(data_path, "swLDA", scenario_name, 
              paste('*', num_rep_fit, '_down_zeta_', zeta_0, '.mat*', sep=""))
  )
  swlda_dat = readMat(swlda_dat_dir)
  zeta_swlda_id = swlda_dat$inmodel
  zeta_swlda_binary[dataset_id,] = zeta_swlda_id
  
  # Truth, zeta_true (borrow the function from global_constant.R)
  zeta_smgp_2_by_2_id = table(zeta_true_0, zeta_smgp_id)
  iswr_smgp_id = zeta_smgp_2_by_2_id[2, 2] / (zeta_smgp_2_by_2_id[2, 1] + zeta_smgp_2_by_2_id[2, 2])
  imwr_smgp_id = zeta_smgp_2_by_2_id[1, 1] / (zeta_smgp_2_by_2_id[1, 1] + zeta_smgp_2_by_2_id[1, 2])
  iswr_smgp = c(iswr_smgp, iswr_smgp_id)
  imwr_smgp = c(imwr_smgp, imwr_smgp_id)
  
  zeta_swlda_2_by_2_id = table(zeta_true_0, zeta_swlda_id)
  iswr_swlda_id = zeta_swlda_2_by_2_id[2, 2] / (zeta_swlda_2_by_2_id[2, 1] + zeta_swlda_2_by_2_id[2, 2])
  imwr_swlda_id = zeta_swlda_2_by_2_id[1, 1] / (zeta_swlda_2_by_2_id[1, 1] + zeta_swlda_2_by_2_id[1, 2])
  iswr_swlda = c(iswr_swlda, iswr_swlda_id)
  imwr_swlda = c(imwr_swlda, imwr_swlda_id)
}

# Compute ISWR and ISMR
sim_single_smgp_prior_dir = file.path(
  parent_path, 'EEG_MATLAB_data', 'SIM_summary',
  paste('sim_', design_num, '_train_', num_rep_fit, '_test_', num_rep_pred, 
        '_smgp_prior_', scenario_name, '_zeta_', zeta_0, '.RDS', sep="")
)
smgp_prior_rds = list(
  true_value = zeta_true_0,
  smgp = zeta_smgp_binary,
  smgp_iswr = iswr_smgp,
  smgp_imwr = imwr_smgp,
  swlda = zeta_swlda_binary,
  swlda_iswr = iswr_swlda,
  swlda_imwr = imwr_swlda
)

# Compute summary statistics of iswr/imwr
smgp_prior_summary_rds = list(
  smgp_iswr_mean = mean(smgp_prior_rds$smgp_iswr),
  smgp_iswr_sd = sd(smgp_prior_rds$smgp_iswr),
  smgp_imwr_mean = mean(smgp_prior_rds$smgp_imwr),
  smgp_imwr_sd = sd(smgp_prior_rds$smgp_imwr),
  
  swlda_iswr_mean = mean(smgp_prior_rds$swlda_iswr),
  swlda_iswr_sd = sd(smgp_prior_rds$swlda_iswr),
  swlda_imwr_mean = mean(smgp_prior_rds$swlda_imwr),
  swlda_imwr_sd = sd(smgp_prior_rds$swlda_imwr)
)

print(smgp_prior_summary_rds)
saveRDS(smgp_prior_rds, file=sim_single_smgp_prior_dir)