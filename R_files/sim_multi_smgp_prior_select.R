# rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T')

library(R.matlab)
library(ggplot2)
library(gridExtra)

if (local_use) {
  parent_path = ''
  r_fun_path = ''
  data_type = 'test'  # train or test
  design_num = 11
  dataset_num = 5
  num_rep_fit = 5
  num_rep_pred = 5
  scenario_name = 'TrueGen'
  zeta_0 = 0.5
} else {
  parent_path = ''
  r_parent_path = ''
  r_fun_path = paste(r_parent_path, 'Chapter_1/R_files', sep="")
  data_type = args[2]
  design_num = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID')) + 1
  dataset_num = 100
  # make sure the environment variable starts from 0 in the slurm script.
  num_rep_fit = 5
  num_rep_pred = 5
  scenario_name = args[3]
  zeta_0 = as.numeric(args[4])
}
source(paste(r_fun_path, '/source_files/global_constant.R', sep=""))

parent_path_plot = Sys.glob(
  file.path(parent_path, '*data*', '*SIM_files*', paste('*sim_', design_num, sep=''))
)
dir.create(file.path(parent_path_plot, 'Plots'))

num_electrode = 3  # fixed for multi-channel dimension
zeta_smgp_binary = array(0, dim=c(n_length_fit, num_electrode, dataset_num))
zeta_swlda_binary = array(0, dim=c(n_length_fit, num_electrode, dataset_num))
iswr_smgp = imwr_smgp = iswr_swlda = imwr_swlda = matrix(0, nrow=dataset_num, ncol=num_electrode)
zeta_true_0 = NULL

for (dataset_id in 1:dataset_num) {
  print(dataset_id)
  # SMGP
  data_path = paste(parent_path_plot, '/sim_', design_num, '_dataset_', dataset_id, sep="")
  smgp_dat_dir_opt = Sys.glob(
    file.path(data_path, '*q2*', scenario_name, paste('*zeta_', zeta_0, '.mat*', sep=""))
  )

  mcmc_dat_opt = readMat(smgp_dat_dir_opt)
  zeta_smgp_id = (apply(mcmc_dat_opt$zeta, c(3, 2), median) > zeta_0) * 1
  zeta_smgp_binary[,,dataset_id] = zeta_smgp_id
  
  zeta_true_0 = t(mcmc_dat_opt$zeta.true[,,1])

  # swLDA
  swlda_dat_dir_opt = Sys.glob(
    file.path(data_path, 'swLDA', scenario_name, '*zeta_0.7.mat*')
  )
  swlda_dat_opt = readMat(swlda_dat_dir_opt)
  zeta_swlda_id = matrix(swlda_dat_opt$inmodel, ncol=3)
  zeta_swlda_binary[,, dataset_id] = zeta_swlda_id
  
  # Truth
  for (e in 1:num_electrode) {
    zeta_smgp_2_by_2_id_e = table(zeta_true_0[, e], zeta_smgp_id[, e])
    iswr_smgp_id_e = zeta_smgp_2_by_2_id_e[2, 2] / (zeta_smgp_2_by_2_id_e[2, 1] + zeta_smgp_2_by_2_id_e[2, 2])
    imwr_smgp_id_e = zeta_smgp_2_by_2_id_e[1, 1] / (zeta_smgp_2_by_2_id_e[1, 1] + zeta_smgp_2_by_2_id_e[1, 2])
    iswr_smgp[dataset_id, e] = iswr_smgp_id_e
    imwr_smgp[dataset_id, e] = imwr_smgp_id_e
    
    zeta_swlda_2_by_2_id_e = table(zeta_true_0[, e], zeta_swlda_id[, e])
    iswr_swlda_id_e = zeta_swlda_2_by_2_id_e[2, 2] / (zeta_swlda_2_by_2_id_e[2, 1] + zeta_swlda_2_by_2_id_e[2, 2])
    imwr_swlda_id_e = zeta_swlda_2_by_2_id_e[1, 1] / (zeta_swlda_2_by_2_id_e[1, 1] + zeta_swlda_2_by_2_id_e[1, 2])
    iswr_swlda[dataset_id, e] = iswr_swlda_id_e
    imwr_swlda[dataset_id, e] = imwr_swlda_id_e
  }
}

sim_multi_smgp_prior_dir = file.path(
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
  smgp_iswr_mean = apply(smgp_prior_rds$smgp_iswr, 2, mean),
  smgp_iswr_sd = apply(smgp_prior_rds$smgp_iswr, 2, sd),
  smgp_imwr_mean = apply(smgp_prior_rds$smgp_imwr, 2, mean),
  smgp_imwr_sd = apply(smgp_prior_rds$smgp_imwr, 2, sd),
  
  swlda_iswr_mean = apply(smgp_prior_rds$swlda_iswr, 2, mean),
  swlda_iswr_sd = apply(smgp_prior_rds$swlda_iswr, 2, sd),
  swlda_imwr_mean = apply(smgp_prior_rds$swlda_imwr, 2, mean),
  swlda_imwr_sd = apply(smgp_prior_rds$swlda_imwr, 2, sd)
)

print(smgp_prior_summary_rds)
saveRDS(smgp_prior_rds, file=sim_multi_smgp_prior_dir)