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
  design_num = 12
  num_rep_fit = 5
  num_rep_pred = 5
  # dataset_num = 10
  scenario_name = 'TrueGen'
  zeta_bayes = 0.1
  
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
  zeta_bayes = as.numeric(args[4])
}
source(paste(r_fun_path, '/source_files/global_constant.R', sep=""))
source(paste(r_fun_path, '/source_files/sim_fun.R', sep=""))
parent_sim_data_path = file.path(parent_path, 'EEG_MATLAB_data', 'SIM_summary')


# Produce True Simulation Functions
index_time_true = seq(0, n_length_fit/32*1000, length.out=n_length_fit)
sim_true_dat = data.frame(
  time = rep(index_time_true, 2*3),
  value = c(mean_fn_tar_chan_1, mean_fn_ntar_chan_1,
            mean_fn_tar_chan_2, mean_fn_ntar_chan_2,
            mean_fn_tar_chan_3, mean_fn_ntar_chan_3),
  type = rep(rep(c('Target', 'Non-target'), each=n_length_fit),3),
  channel = rep(paste('Channel', 1:3, sep=" "), each=n_length_fit * 2)
)

p_true = ggplot() + 
  geom_line(data=sim_true_dat, size=1.2, aes(x=time, y=value, color=type), linetype=1) +
  geom_point(data=sim_true_dat, size=2, aes(x=time, y=value, color=type), shape=19) +
  xlab('Time (ms)') + ylab('Amplitude (muV)') + 
  facet_wrap(~channel, ncol=3) +
  scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, by=200)) +
  scale_y_continuous(limits=c(-1, 5.5), breaks=seq(-1, 5, by=1)) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = 'bottom',
        legend.title = element_blank(),
        legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
        plot.margin = margin(.5, .5, .5, .5, "cm"))
print(p_true)
ggsave(file.path(parent_sim_data_path,
                 paste('simulation_multi_mean_function_type_1.png', sep="")),
       plot=p_true, height=120, width=240, units='mm', dpi=500)




# import mcmc files to produce plots
# extract true response function to target/non-target stimuli from global_constant.R
# dataset_id = 1
prob_vec = c(0.025, 0.975)

mcmc_id_1 = Sys.glob(
  file.path(parent_sim_data_path,
  'multi_channel/case11/sim_11_dataset_1/BayesGenq2/TrueGen/', '*zeta_0.1.mat*')
)

true_gen_mcmc_est_1 = import_extract_sim_est_fun_multi(mcmc_id_1, prob_vec)
index_time = seq(0, n_length_fit/32*1000, length.out=n_length_fit)

sim_est_dat_1 = data.frame(
  time = rep(index_time, 2*3),
  mean = c(true_gen_mcmc_est_1$tar_mean[1,], true_gen_mcmc_est_1$ntar_mean[1,],
           true_gen_mcmc_est_1$tar_mean[2,], true_gen_mcmc_est_1$ntar_mean[2,],
           true_gen_mcmc_est_1$tar_mean[3,], true_gen_mcmc_est_1$ntar_mean[3,]),
  low = c(true_gen_mcmc_est_1$tar_ci[1,1,], true_gen_mcmc_est_1$ntar_ci[1,1,],
          true_gen_mcmc_est_1$tar_ci[1,2,], true_gen_mcmc_est_1$ntar_ci[1,2,],
          true_gen_mcmc_est_1$tar_ci[1,3,], true_gen_mcmc_est_1$ntar_ci[1,3,]),
  upp = c(true_gen_mcmc_est_1$tar_ci[2,1,], true_gen_mcmc_est_1$ntar_ci[2,1,],
          true_gen_mcmc_est_1$tar_ci[2,2,], true_gen_mcmc_est_1$ntar_ci[2,2,],
          true_gen_mcmc_est_1$tar_ci[2,3,], true_gen_mcmc_est_1$ntar_ci[2,3,]),
  true = c(mean_fn_tar_chan_1, mean_fn_ntar_chan_1,
           mean_fn_tar_chan_2, mean_fn_ntar_chan_2,
           mean_fn_tar_chan_3, mean_fn_ntar_chan_3),
  type = rep(rep(c('Target', 'Non-target'), each=n_length_fit), 3),
  scenario_name = scenario_name,
  channel = rep(1:3, each=2*n_length_fit),
  channel_std = paste('Channel ', rep(1:3, each=2*n_length_fit), sep="")
)
sim_est_dat_1$type = factor(sim_est_dat_1$type, levels=c('Target', 'Non-target'))
# sim_est_dat$channel_std = factor(sim_est_dat$channel, levels=c('Channel 1', 'Channel 2', 'Channel 3'))

p_est_1 = ggplot() + 
  geom_ribbon(data=sim_est_dat_1, aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.3) +
  geom_line(data=sim_est_dat_1, size=0.75, aes(x=time, y=true, color=type), linetype=1) +
  geom_point(data=sim_est_dat_1, size=2, aes(x=time, y=true, color=type), shape=1) +
  # geom_hline(yintercept = 0, linetype=2) + 
  xlab('Time (ms)') + ylab('Amplitude (muV)') + 
  facet_wrap(~channel_std, nrow=1) +
  scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, by=300)) +
  scale_y_continuous(limits=c(-2, 6.5), breaks=seq(-2, 6, by=1)) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = 'bottom',
        legend.title = element_blank(),
        legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
        plot.margin = margin(.5, .5, .5, .5, "cm"))
print(p_est_1)


mcmc_id_2 = Sys.glob(
  file.path(parent_sim_data_path,
            'multi_channel/case12/sim_12_dataset_5/BayesGenq2/TrueGen/', '*zeta_0.1.mat*')
)

true_gen_mcmc_est_2 = import_extract_sim_est_fun_multi(mcmc_id_2, prob_vec)
# index_time = seq(0, n_length_fit/32*1000, length.out=n_length_fit)

sim_est_dat_2 = data.frame(
  time = rep(index_time, 2*3),
  mean = c(true_gen_mcmc_est_2$tar_mean[1,], true_gen_mcmc_est_2$ntar_mean[1,],
           true_gen_mcmc_est_2$tar_mean[2,], true_gen_mcmc_est_2$ntar_mean[2,],
           true_gen_mcmc_est_2$tar_mean[3,], true_gen_mcmc_est_2$ntar_mean[3,]),
  low = c(true_gen_mcmc_est_2$tar_ci[1,1,], true_gen_mcmc_est_2$ntar_ci[1,1,],
          true_gen_mcmc_est_2$tar_ci[1,2,], true_gen_mcmc_est_2$ntar_ci[1,2,],
          true_gen_mcmc_est_2$tar_ci[1,3,], true_gen_mcmc_est_2$ntar_ci[1,3,]),
  upp = c(true_gen_mcmc_est_2$tar_ci[2,1,], true_gen_mcmc_est_2$ntar_ci[2,1,],
          true_gen_mcmc_est_2$tar_ci[2,2,], true_gen_mcmc_est_2$ntar_ci[2,2,],
          true_gen_mcmc_est_2$tar_ci[2,3,], true_gen_mcmc_est_2$ntar_ci[2,3,]),
  true = c(mean_fn_tar_chan_1, mean_fn_ntar_chan_1,
           mean_fn_tar_chan_2, mean_fn_ntar_chan_2,
           mean_fn_tar_chan_3, mean_fn_ntar_chan_3),
  type = rep(rep(c('Target', 'Non-target'), each=n_length_fit), 3),
  scenario_name = scenario_name,
  channel = rep(1:3, each=2*n_length_fit),
  channel_std = paste('Channel ', rep(1:3, each=2*n_length_fit), sep="")
)
sim_est_dat_2$type = factor(sim_est_dat_2$type, levels=c('Target', 'Non-target'))
# sim_est_dat$channel_std = factor(sim_est_dat$channel, levels=c('Channel 1', 'Channel 2', 'Channel 3'))

p_est_2 = ggplot() + 
  geom_ribbon(data=sim_est_dat_2, aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.3) +
  geom_line(data=sim_est_dat_2, size=0.75, aes(x=time, y=true, color=type), linetype=1) +
  geom_point(data=sim_est_dat_2, size=2, aes(x=time, y=true, color=type), shape=1) +
  # geom_hline(yintercept = 0, linetype=2) + 
  xlab('Time (ms)') + ylab('Amplitude (muV)') + 
  facet_wrap(~channel_std, nrow=1) +
  scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, by=300)) +
  scale_y_continuous(limits=c(-2, 6.5), breaks=seq(-2, 6, by=1)) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = 'bottom',
        legend.title = element_blank(),
        legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
        plot.margin = margin(.5, .5, .5, .5, "cm"))
print(p_est_2)

p_est = grid.arrange(p_est_1, p_est_2, nrow=2)
