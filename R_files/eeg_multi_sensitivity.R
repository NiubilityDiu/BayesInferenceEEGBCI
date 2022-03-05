# rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T')

library(ggplot2)
library(gridExtra)
library(R.matlab)

if (local_use) {
  parent_path = ''
  r_fun_path = ''
  data_type = 'test'  # train or test
  design_num = 114
  num_rep_fit = 7
  num_rep_pred = 7
  dec_factor = 4
  n_length_fit = ifelse(dec_factor == 4, 50, 25)
  bp_low = 0.5
  bp_upp = 6
  time_length = 25/32*1000
  zeta = 0.4
} else {
  parent_path = ''
  r_parent_path = ''
  r_fun_path = ''
  data_type = args[2]
  design_num = as.integer(args[3])
  # make sure the environment variable starts from 0 in the slurm script.
  num_rep_fit = as.integer(args[4])
  num_rep_pred = as.integer(args[5])
  dec_factor = as.integer(args[6])
  n_length_fit = ifelse(dec_factor == 4, 50, 25)
  bp_low = 0.5
  bp_upp = as.numeric(args[7])
  time_length = 25 / 32 * 1000
  zeta = as.numeric(args[8])
}
source(file.path(r_fun_path, 'source_files', 'eeg_global_constant.R'))
source(file.path(r_fun_path, 'source_files', 'sim_fun.R'))

# We still use the ranking based on single-channel model fitting
multi_channel_ls = list(
  K114=c(6, 7, 8, 10, 12)
)

multi_channel_num = multi_channel_ls[[paste('K', design_num, sep="")]]
multi_channel_output = multi_channel_names(multi_channel_num)
multi_channel_simple = multi_channel_output$num
multi_channel_official = multi_channel_output$official
# multi_channel = multi_channel  # exclude single channel

parent_path_plot = Sys.glob(
  file.path(parent_path, '*data*', '*TRN_files*', paste('*', design_num, '*', sep=''))
)
dir.create(file.path(parent_path_plot, 'fit_plots'), showWarnings = F)

# Sensitivity analysis
beta_vec_mean = beta_vec_upp = beta_vec_low = zeta_vec_median = NULL
channel_vec = channel_vec_multi = type_vec = time_vec = NULL
kernel_param = NULL
scale_vec = c(0.25, 0.5, 0.75)
gamma_vec = c(1.2, 1.8, 1.99)

e_temp = c(4)

for (scale in scale_vec) {
  for (gamma in gamma_vec) {
    hyper_param = paste('scale=', scale, ', gamma=', gamma, sep="")
    print(hyper_param)
    for (e in e_temp) {
      channel_name_num_e = multi_channel_simple[e]
      print(channel_name_num_e)
      mcmc_e_dir = Sys.glob(file.path(
        parent_path_plot, '*q2*', hyper_param, channel_name_num_e,
        paste('*binary_down_', dec_factor, '_raw_bp_', 
              bp_low, '_', bp_upp, '_zeta_', zeta, '.mat*', sep=""))
      )
      mcmc_e_mat = readMat(mcmc_e_dir)
      beta_e_tar = mcmc_e_mat$beta.tar[, , , 1]
      beta_e_ntar = mcmc_e_mat$beta.ntar[, , , 1]
      beta_e_tar_mean = apply(beta_e_tar, c(2, 3), mean)
      beta_e_ntar_mean = apply(beta_e_ntar, c(2, 3), mean)
      
      beta_e_tar_upp = apply(beta_e_tar, c(2, 3), quantile, 0.95)
      beta_e_tar_low = apply(beta_e_tar, c(2, 3), quantile, 0.05)
      
      beta_e_ntar_upp = apply(beta_e_ntar, c(2, 3), quantile, 0.95)
      beta_e_ntar_low = apply(beta_e_ntar, c(2, 3), quantile, 0.05)
      
      beta_vec_mean = c(beta_vec_mean, 
                        as.vector(t(beta_e_tar_mean)), 
                        as.vector(t(beta_e_ntar_mean)))
      beta_vec_low = c(beta_vec_low, 
                       as.vector(t(beta_e_tar_low)), 
                       as.vector(t(beta_e_ntar_low)))
      beta_vec_upp = c(beta_vec_upp, 
                       as.vector(t(beta_e_tar_upp)), 
                       as.vector(t(beta_e_ntar_upp)))
      zeta_e = mcmc_e_mat$zeta
      zeta_e_median = apply(zeta_e, c(2, 3), median)
      zeta_vec_median = c(zeta_vec_median, rep(as.vector(t(zeta_e_median)), 2))
      channel_vec = c(channel_vec, rep(rep(multi_channel_num[1:e], each=n_length_fit), 2))
      channel_vec_multi = c(channel_vec_multi, rep(multi_channel_simple[e], n_length_fit*e*2))
      type_vec = c(type_vec, rep(c('Target', 'Non-target'), each=n_length_fit*e))
      time_vec = c(time_vec, rep(seq(0, time_length, length.out=n_length_fit), 2*e))
      kernel_param = c(kernel_param, rep(hyper_param, n_length_fit*e*2))
    }
  }
}

mcmc_multi_dat = data.frame(
  mean = beta_vec_mean,
  low = beta_vec_low,
  upp = beta_vec_upp,
  zeta = zeta_vec_median,
  channel = channel_vec,
  channel_multi = channel_vec_multi,
  type = type_vec,
  time = time_vec,
  kernel_param = kernel_param
)
mcmc_multi_dat$type = factor(mcmc_multi_dat$type, levels=c('Target', 'Non-target'))
mcmc_multi_dat$channel = factor(mcmc_multi_dat$channel, levels=1:16, 
                                labels=channel_name_short)

time_upper_lim = 800
y_upper_lim = ceiling(max(mcmc_multi_dat$upp))
y_lower_lim = floor(min(mcmc_multi_dat$low))
# y_lower_lim = -2
# y_upper_lim = 5
p_mcmc_multi_dat = ggplot(data=mcmc_multi_dat[mcmc_multi_dat$channel=='Cz',], aes(x=time, y=mean)) +
  geom_line(size=1.0, aes(x=time, y=mean, color=type)) +
  # geom_hline(yintercept=0, linetype=2, alpha=0.5) + 
  geom_ribbon(aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.5) +
  geom_point(size=1.0, aes(x=time, y=mean, color=type), shape=1) +
  scale_x_continuous(limits=c(0, time_upper_lim),
                     breaks=seq(0, time_upper_lim, length.out=5)) +
  scale_y_continuous(limits=c(y_lower_lim, y_upper_lim), breaks=seq(y_lower_lim, y_upper_lim, by=3)) +
  xlab('Time (ms)') + ylab('Amplitude (muV)') +
  ggtitle('Estimated P300−ERPs') +
  facet_wrap(~kernel_param, nrow=3) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background=element_rect(fill = "white",
                                      colour = "black",
                                      size = 0.5, linetype = "solid"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='bottom',
        legend.title=element_blank(),
        legend.background=element_rect(fill='transparent', size=0.2,
                                       color='white', linetype='solid'),
        plot.margin=margin(.25, .25, .25, .25, 'cm'))






mcmc_multi_zeta_dat = mcmc_multi_dat[
  mcmc_multi_dat$type=='Target' & mcmc_multi_dat$channel == 'Cz', 
  c('zeta', 'channel', 'channel_multi', 'type', 'time', 'kernel_param')
]
zeta_binary = c(ifelse(mcmc_multi_zeta_dat$zeta > 0.6, 0.6, -1),
                ifelse(mcmc_multi_zeta_dat$zeta >= 0.75, 0.75, -1),
                ifelse(mcmc_multi_zeta_dat$zeta >= 0.9, 0.9, -1))
mcmc_multi_zeta_dat = rbind.data.frame(
  mcmc_multi_zeta_dat, mcmc_multi_zeta_dat, mcmc_multi_zeta_dat
)
mcmc_multi_zeta_dat$zeta_binary = zeta_binary
mcmc_multi_zeta_dat$threshold = rep(c(0.6, 0.75, 0.9), each=n_length_fit * 9)
mcmc_multi_zeta_dat$threshold = factor(
  mcmc_multi_zeta_dat$threshold, levels = c(0.6, 0.75, 0.9),
  labels=c('60% Confidence', '75% Confidence', '90% Confidence')
)

p_mcmc_multi_zeta_dat = ggplot(
  data=mcmc_multi_zeta_dat, aes(x=time, y=zeta_binary)) +
  geom_line(size=2.0, aes(x=time, y=zeta_binary, color=threshold)) +
  # geom_hline(yintercept=0.5, linetype=2, alpha=0.5) + 
  # geom_point(size=1.0, aes(x=time, y=zeta), shape=1) +
  scale_x_continuous(limits=c(0, time_upper_lim),
                     breaks=seq(0, time_upper_lim, length.out=5)) +
  scale_y_continuous(limits=c(0.5, 1), breaks=seq(0.5, 1, by=0.1)) +
  xlab('Time (ms)') + ylab('Probability') + ggtitle('Split Probabilities') +
  scale_color_brewer(palette='Dark2') +
  facet_wrap(~kernel_param, nrow=3) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background=element_rect(fill = "white",
                                      colour = "black",
                                      size = 0.5, linetype = "solid"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='bottom',
        legend.title=element_blank(),
        legend.background=element_rect(fill='transparent', size=0.2,
                                       color='white', linetype='solid'),
        plot.margin=margin(.25, .25, .25, .25, 'cm'))
print(p_mcmc_multi_zeta_dat)

p_mcmc_multi_inference = grid.arrange(p_mcmc_multi_dat, p_mcmc_multi_zeta_dat, nrow=1)
output_dir = paste('K', design_num, '_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
                   '_zeta_', zeta, '_multi_channel_Cz_sensitivity_inference', sep="")
ggsave(file.path(parent_path_plot, 'fit_plots', paste(output_dir, '.png', sep="")),
       p_mcmc_multi_inference, width=300, height=150, units='mm', dpi=400)



# Prediction comparsion
# Prediction Accuracy Comparison
parent_path_bayes = Sys.glob(
  file.path(parent_path, '*data*', '*BayesGenq2Pred*', 
            paste('*', design_num, '*', sep=''))
)
predict_bayes = channel_vec = seq_vec = hyper_param_vec = NULL

for (scale in scale_vec) {
  for (gamma in gamma_vec) {
    hyper_param = paste('scale=', scale, ', gamma=', gamma, sep="")
    hyper_param_vec = c(hyper_param_vec, rep(hyper_param, 7*1))
    for (e in e_temp) {
      channel_name_e = multi_channel_simple[e]
      channel_name_official_e = multi_channel_official[e]
      print(channel_name_e)
      
      bayes_e_dir = Sys.glob(file.path(
        parent_path_bayes, hyper_param, channel_name_e,
        paste('*', data_type, '_train_', num_rep_fit, '_pred_test_', num_rep_pred,
              '_binary_down_', dec_factor, '_raw_bp_', 
              bp_low, '_', bp_upp, '_zeta_', zeta, '.csv*', sep=""))
      )
      
      predict_bayes_e = read.csv(bayes_e_dir, header=F)
      predict_bayes_e_sub = predict_bayes_e[182, 1:7]
      predict_bayes = c(predict_bayes, as.numeric(as.matrix(predict_bayes_e_sub)))
      channel_vec = c(channel_vec, rep(channel_name_official_e, 7))
      seq_vec = c(seq_vec, 1:7)
    }
  }
}

predict_multi_dat = data.frame(
  bayes = signif(predict_bayes / num_letter, digits=2),
  # swlda = signif(predict_swlda / num_letter, digits=2),
  channel = channel_vec,
  seq = seq_vec,
  kernel_param = hyper_param_vec
)

p_predict_multi_dat = ggplot(data=predict_multi_dat[predict_multi_dat$channel=='Cz,C4,T8,CP4'
                                                    & predict_multi_dat$seq>=4,], aes(x=seq, y=bayes, color=kernel_param)) +
  geom_line(size=1, aes(x=seq, y=bayes, color=kernel_param)) +
  geom_hline(yintercept=0.6, linetype=2) + 
  geom_hline(yintercept=0.8, linetype=2) + 
  geom_point(size=2, aes(x=seq, y=bayes, color=kernel_param), shape=1) +
  scale_x_continuous(limits=c(4, 7), breaks=4:7) + 
  scale_y_continuous(limits=c(0.5, 1), breaks=seq(0.5, 1, by=0.1)) +
  xlab('Sequence Number') + ylab('Proportion') +
  ggtitle('Prediction Accuracy') +
  # facet_wrap(~channel, ncol=3) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background=element_rect(fill = "white",
                                      colour = "black",
                                      size = 0.5, linetype = "solid"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='bottom',
        legend.title=element_blank(),
        legend.background=element_rect(fill='transparent', size=0.2,
                                       color='white', linetype='solid'),
        plot.margin=margin(.5, .5, .5, .5, 'cm'))
print(p_predict_multi_dat)
