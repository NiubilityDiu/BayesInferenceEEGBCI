# Produce estimated ERP functions and prediction accuracy plot 
# for single-channel simulation study under five scenarios
# rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T')
library(ggplot2)
require(gridExtra)
library(R.matlab)

if (local_use) {
  parent_path = ''
  r_fun_path = ''
  data_type = 'test'  # train or test
  design_num = 5
  num_rep_fit = 5
  num_rep_pred = 5
  zeta = 0.5
  
} else {
  parent_path = ''
  r_parent_path = ''
  r_fun_path = paste(r_parent_path, 'Chapter_1/R_files', sep="")
  data_type = args[2]
  design_num = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID')) + 1
  num_rep_fit = 5
  num_rep_pred = 5
  dataset_num = 100
  scenario_name = args[3]
  zeta = as.numeric(args[4])
}
source(paste(r_fun_path, '/source_files/global_constant.R', sep=""))
source(paste(r_fun_path, '/source_files/sim_fun.R', sep=""))

parent_sim_data_path = file.path(parent_path, 'EEG_MATLAB_data', 'SIM_summary')
dataset_id = 8
prob_vec = c(0.025, 0.975)

scenario_names = c('TrueGen', 'MisNoiseDist', 'MisLatencyLen25',
                   'MisLatencyLen35', 'MisSignalDist')
scenario_names_full = c('True Generative', 'Mis-specified Noise',
                        'Shorter Response Window', 'Longer Response Window',
                        'Mis-specified Signal Distribution')
true_gen_mcmc_est = import_extract_sim_est_fun(
  parent_sim_data_path, design_num, dataset_id, scenario_names[1], prob_vec
)
mis_noise_mcmc_est = import_extract_sim_est_fun(
  parent_sim_data_path, design_num, dataset_id, scenario_names[2], prob_vec
)
mis_latency_25_mcmc_est = import_extract_sim_est_fun(
  parent_sim_data_path, design_num, dataset_id, scenario_names[3], prob_vec
)
mis_latency_35_mcmc_est = import_extract_sim_est_fun(
  parent_sim_data_path, design_num, dataset_id, scenario_names[4], prob_vec
)
mis_signal_mcmc_est = import_extract_sim_est_fun(
  parent_sim_data_path, design_num, dataset_id, scenario_names[5], prob_vec
)
index_time = seq(0, n_length_fit/32*1000, length.out=n_length_fit)
sim_est_dat = data.frame(
  time = rep(index_time, 2*5),
  mean = c(true_gen_mcmc_est$tar_mean, true_gen_mcmc_est$ntar_mean,
           mis_noise_mcmc_est$tar_mean, mis_noise_mcmc_est$ntar_mean,
           mis_latency_25_mcmc_est$tar_mean, mis_latency_25_mcmc_est$ntar_mean,
           mis_latency_35_mcmc_est$tar_mean, mis_latency_35_mcmc_est$ntar_mean,
           mis_signal_mcmc_est$tar_mean, mis_signal_mcmc_est$ntar_mean),
  low = c(true_gen_mcmc_est$tar_ci[1,], true_gen_mcmc_est$ntar_ci[1,],
          mis_noise_mcmc_est$tar_ci[1,], mis_noise_mcmc_est$ntar_ci[1,],
          mis_latency_25_mcmc_est$tar_ci[1,], mis_latency_25_mcmc_est$ntar_ci[1,],
          mis_latency_35_mcmc_est$tar_ci[1,], mis_latency_35_mcmc_est$ntar_ci[1,],
          mis_signal_mcmc_est$tar_ci[1,], mis_signal_mcmc_est$ntar_ci[1,]),
  upp = c(true_gen_mcmc_est$tar_ci[2,], true_gen_mcmc_est$ntar_ci[2,],
          mis_noise_mcmc_est$tar_ci[2,], mis_noise_mcmc_est$ntar_ci[2,],
          mis_latency_25_mcmc_est$tar_ci[2,], mis_latency_25_mcmc_est$ntar_ci[2,],
          mis_latency_35_mcmc_est$tar_ci[2,], mis_latency_35_mcmc_est$ntar_ci[2,],
          mis_signal_mcmc_est$tar_ci[2,], mis_signal_mcmc_est$ntar_ci[2,]),
  true = c(mean_fn_tar_1, mean_fn_ntar_1,
           mean_fn_tar_1, mean_fn_ntar_1,
           mean_fn_tar_3, rep(0, 5), mean_fn_ntar_3, rep(0, 5),
           mean_fn_tar_5[1:n_length_fit], mean_fn_ntar_5[1:n_length_fit],
           mean_fn_tar_1, mean_fn_ntar_1),
  type = rep(rep(c('Target', 'Non-target'), each=n_length_fit), 5),
  scenario_name = rep(scenario_names_full, each=n_length_fit*2)
)
sim_est_dat$type = factor(sim_est_dat$type, levels=c('Target', 'Non-target'))
sim_est_dat$scenario_name = factor(
  sim_est_dat$scenario_name,
  levels=scenario_names_full
)

p_est = ggplot() + 
  geom_ribbon(data=sim_est_dat, aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.3) +
  geom_line(data=sim_est_dat, size=0.75, aes(x=time, y=true, color=type), linetype=1) +
  geom_point(data=sim_est_dat, size=2, aes(x=time, y=true, color=type), shape=1) +
  xlab('Time (ms)') + ylab('Amplitude (muV)') + 
  facet_wrap(~scenario_name, nrow=1) +
  scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, by=300)) +
  scale_y_continuous(limits=c(-2, 6), breaks=seq(-2, 6, by=1)) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = 'bottom',
        legend.title = element_blank(),
        legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
        plot.margin = margin(.5, .5, .5, .5, "cm"))
print(p_est)

# import .RDS files for 5 scenarios of the same simulation design number.
true_gen_dir = file.path(parent_sim_data_path, 
                         paste('sim_', design_num, '_train_', num_rep_fit, '_test_', num_rep_pred,
                               '_summary_', scenario_names[1], '_zeta_', zeta, '.RDS', sep="")) 
mis_noise_dir = file.path(parent_sim_data_path, 
                          paste('sim_', design_num, '_train_', num_rep_fit, '_test_', num_rep_pred,
                                '_summary_', scenario_names[2], '_zeta_', zeta, '.RDS', sep="")) 
mis_latency_25_dir = file.path(parent_sim_data_path, 
                               paste('sim_', design_num, '_train_', num_rep_fit, '_test_', num_rep_pred,
                                     '_summary_', scenario_names[3], '_zeta_', zeta, '.RDS', sep="")) 
mis_latency_35_dir = file.path(parent_sim_data_path, 
                               paste('sim_', design_num, '_train_', num_rep_fit, '_test_', num_rep_pred,
                                     '_summary_', scenario_names[4], '_zeta_', zeta, '.RDS', sep="")) 
mis_signal_dir = file.path(parent_sim_data_path, 
                               paste('sim_', design_num, '_train_', num_rep_fit, '_test_', num_rep_pred,
                                     '_summary_', scenario_names[5], '_zeta_', zeta, '.RDS', sep="")) 
true_gen_dat = readRDS(Sys.glob(true_gen_dir))
mis_noise_dat = readRDS(Sys.glob(mis_noise_dir))
mis_latency_25_dat = readRDS(Sys.glob(mis_latency_25_dir))
mis_latency_35_dat = readRDS(Sys.glob(mis_latency_35_dir))
mis_signal_dat = readRDS(Sys.glob(mis_signal_dir))

sim_pred_total = data.frame(
  method = rep(true_gen_dat$method_table$method, 5),
  rep_id = rep(true_gen_dat$method_table$rep_id, 5),
  mean = c(true_gen_dat$method_table$mean, 
           mis_noise_dat$method_table$mean,
           mis_latency_25_dat$method_table$mean,
           mis_latency_35_dat$method_table$mean,
           mis_signal_dat$method_table$mean),
  sd = c(true_gen_dat$method_table$sd, 
          mis_noise_dat$method_table$sd,
          mis_latency_25_dat$method_table$sd,
          mis_latency_35_dat$method_table$sd,
          mis_signal_dat$method_table$sd),
  scenario = rep(scenario_names_full, each=5*8)
)
sim_pred_total$scenario = factor(sim_pred_total$scenario, levels=scenario_names_full)

sim_pred_total_sub = sim_pred_total[sim_pred_total$method != 'ADABoosting' &
                                      sim_pred_total$method != 'Bagging',]
print(head(sim_pred_total_sub))

p_pred = ggplot(sim_pred_total_sub, aes(x=rep_id, y=mean)) +
  geom_point(size=1.5, aes(x=rep_id, y=mean, shape=method)) +
  geom_line(size=0.5, aes(x=rep_id, y=mean, linetype=method)) +
  geom_errorbar(aes(ymin=pmax(0, mean-sd), ymax=pmin(1, mean+sd), linetype=method),
                width=0.1, size=0.5) +
  ylim(c(-0.05, 1.05)) +
  xlab('Sequence Replications') + ylab("Prediction Accuracy") +
  scale_x_continuous(breaks=as.integer(seq(1, num_rep_pred, length.out=min(num_rep_pred, 5)))) +
  facet_wrap(~scenario, ncol=5) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
        # panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.background = element_rect(fill='transparent', size=0.25, 
                                         color='white', linetype='solid'),
        plot.margin = margin(.5, .5, .5, .5, "cm"))
print(p_pred)

p_sim_comb = grid.arrange(p_est, p_pred, ncol=1)
