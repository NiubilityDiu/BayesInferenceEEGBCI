# Produce the channel-specific estimated ERPs function for single channel and multi-channels.
# rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T')
library(ggplot2)
library(gridExtra)
library(R.matlab)
library(dplyr)
library(patchwork)

if (local_use) {
  parent_path = ''
  r_fun_path = ''
  data_type = 'test'  # train or test
  design_num = 117
  num_rep_fit = 7
  num_rep_pred = 7
  dec_factor = 8
  n_length_fit = ifelse(dec_factor == 4, 50, 25)
  bp_low = 0.5
  bp_upp = 6
  time_length = 25/32*1000
  zeta_0 = 0.4
  } else {
  parent_path = ''
  r_parent_path = ''
  r_fun_path = file.path(r_parent_path, 'R_files')
  data_type = args[2]
  design_num = as.integer(args[3])
  num_rep_fit = as.integer(args[4])
  num_rep_pred = as.integer(args[5])
  dec_factor = as.integer(args[6])
  n_length_fit = ifelse(dec_factor == 4, 50, 25)
  bp_low = 0.5
  bp_upp = as.numeric(args[7])
  time_length = 25 / 32 * 1000
  zeta_0 = as.numeric(args[8])
}
source(file.path(r_fun_path, 'source_files', 'eeg_global_constant.R'))
source(file.path(r_fun_path, 'source_files', 'sim_fun.R'))


parent_path_plot = Sys.glob(
  file.path(parent_path, '*data*', '*TRN_files*', paste('*', design_num, '*', sep=''))
)
dir.create(file.path(parent_path_plot, 'fit_plots'), showWarnings = F)
multi_channel_num = 1:16


if (F) {
  # Only include 16-channel model fitting results
  e = 16
  beta_vec_mean = beta_vec_upp = beta_vec_low = channel_vec = type_vec = time_vec =NULL
  zeta_vec_median = NULL
  channel_name_num_e = multi_channel_simple[e]
  mcmc_e_dir = Sys.glob(
    file.path(
      parent_path_plot, '*q2*',
      'all_channels', 
      paste('*continuous_down_', dec_factor, '_raw_bp_', 
            bp_low, '_', bp_upp, '.mat*', sep=""))
  )
  mcmc_e_mat = readMat(mcmc_e_dir)
  zeta_e = mcmc_e_mat$zeta
  zeta_e_median = apply(zeta_e, c(2, 3), median)
  zeta_vec_median = c(zeta_vec_median, rep(as.vector(t(zeta_e_median)), 2))
  
  beta_e_tar = mcmc_e_mat$beta.tar[, , , 1]
  beta_e_ntar = mcmc_e_mat$beta.ntar[, , , 1]
  # merge raw beta estimates based on zeta_e_median
  for (ee in 1:e) {
    for (ii in 1:n_length_fit) {
      if (zeta_e_median[ee, ii] <= zeta_0) {
        beta_e_temp = (beta_e_tar[, ee, ii] + 5 * beta_e_ntar[, ee, ii]) / 6
        beta_e_tar[, ee, ii] = beta_e_temp
        beta_e_ntar[, ee, ii] = beta_e_temp
      }
    }
  }
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
  
  channel_vec = c(channel_vec, rep(rep(multi_channel_num[1:e], each=n_length_fit), 2))
  type_vec = c(type_vec, rep(c('Target', 'Non-target'), each=n_length_fit*e))
  time_vec = c(time_vec, rep(seq(0, time_length, length.out=n_length_fit), 2*e))
  
  mcmc_multi_dat = data.frame(
    mean = beta_vec_mean,
    low = beta_vec_low,
    upp = beta_vec_upp,
    zeta = zeta_vec_median,
    channel = channel_vec,
    type = type_vec,
    time = time_vec
  )
  mcmc_multi_dat$type = factor(mcmc_multi_dat$type, levels=c('Target', 'Non-target'))
  mcmc_multi_dat$channel = factor(mcmc_multi_dat$channel, levels=1:16, 
                                  labels=channel_name_short)
  mcmc_multi_dat$zeta_60 = ifelse(mcmc_multi_dat$zeta >= 0.6, 0.6, -1)
  mcmc_multi_dat$zeta_75 = ifelse(mcmc_multi_dat$zeta >= 0.75, 0.75, -1)
  mcmc_multi_dat$zeta_90 = ifelse(mcmc_multi_dat$zeta >= 0.90, 0.90, -1)
  
  time_upper_lim = 800
  y_upper_lim = ceiling(max(mcmc_multi_dat$upp))
  y_lower_lim = floor(min(mcmc_multi_dat$low))
  
  
  p_multi_spatial_ls = lapply(1:e, function(x) NULL)
  names(p_multi_spatial_ls) = channel_name_short
  for(ee in 1:e) {
    cee = channel_name_short[ee]
    p_multi_spatial_ls[[ee]] = ggplot(
      data=mcmc_multi_dat %>% filter(channel==cee), aes(x=time, y=mean)) +
      geom_line(size=1.0, aes(x=time, y=mean, color=type)) +
      # geom_hline(yintercept=0, linetype=2, alpha=0.5) + 
      geom_ribbon(aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.5) +
      geom_point(size=1.0, aes(x=time, y=mean, color=type), shape=1) +
      scale_x_continuous(limits=c(0, time_upper_lim),
                         breaks=seq(0, time_upper_lim, length.out=5)) +
      scale_y_continuous(limits=c(y_lower_lim, y_upper_lim), 
                         breaks=seq(y_lower_lim, y_upper_lim, by=2)) +
      xlab('') + ylab('') + ggtitle(cee) +
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
  }
  # Rearrange the plots by the spatial distribution of EEG channels
  layout_eeg = '
#ABC#
DEFGH
#I#J#
#KLM#
#NOP#
'
  p_multi_dat = 
    p_multi_spatial_ls$F3 + p_multi_spatial_ls$Fz + 
    p_multi_spatial_ls$F4 + p_multi_spatial_ls$T7 + 
    p_multi_spatial_ls$C3 + p_multi_spatial_ls$Cz + 
    p_multi_spatial_ls$C4 + p_multi_spatial_ls$T8 + 
    p_multi_spatial_ls$CP3 + p_multi_spatial_ls$CP4 + 
    p_multi_spatial_ls$P3 + p_multi_spatial_ls$Pz + 
    p_multi_spatial_ls$P4 + p_multi_spatial_ls$PO7 + 
    p_multi_spatial_ls$Oz + p_multi_spatial_ls$PO8 +
    plot_layout(design=layout_eeg, guides='collect') & theme(legend.position = 'bottom')
  output_dir = paste('K', design_num, '_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
                     '_zeta_', zeta_0, '_multi_channel_', e, '_erp_estimate', sep="")
  ggsave(file.path(parent_path_plot, 'fit_plots', paste(output_dir, '.png', sep="")),
         p_multi_dat, width=300, height=250, units='mm', dpi=400)
  
  
  
  p_multi_spatial_zeta_ls = lapply(1:e, function(x) NULL)
  names(p_multi_spatial_zeta_ls) = channel_name_short
  for (ee in 1:e) {
    cee = channel_name_short[ee]
    mcmc_multi_zeta_dat = data.frame(
      channel = cee,
      time = rep(seq(0, 25/32*1000, length.out=n_length_fit), 3),
      zeta_binary = c(mcmc_multi_dat$zeta_60[mcmc_multi_dat$type=='Target' & 
                                               mcmc_multi_dat$channel == cee],
                      mcmc_multi_dat$zeta_75[mcmc_multi_dat$type=='Target' & 
                                               mcmc_multi_dat$channel == cee],
                      mcmc_multi_dat$zeta_90[mcmc_multi_dat$type=='Target' & 
                                               mcmc_multi_dat$channel == cee]),
      threshold = rep(c(0.6, 0.75, 0.9), each=n_length_fit)
    )
    mcmc_multi_zeta_dat$threshold = factor(
      mcmc_multi_zeta_dat$threshold, levels=c(0.6, 0.75, 0.9),
      labels=c('60% Confidence', '75% Confidence', '90% Confidence')
    )
    p_multi_spatial_zeta_ls[[ee]] = ggplot(
      data=mcmc_multi_zeta_dat, aes(x=time, y=zeta_binary)) +
      geom_line(size=2.0, aes(x=time, y=zeta_binary, color=threshold)) +
      scale_x_continuous(limits=c(0, time_upper_lim),
                         breaks=seq(0, time_upper_lim, length.out=5)) +
      scale_y_continuous(limits=c(0.5, 1), breaks=seq(0.5, 1, by=0.1)) +
      xlab('') + ylab('') + ggtitle(cee) +
      scale_color_brewer(palette="Accent") +
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
  }
  
  # Rearrange the plots by the spatial distribution of EEG channels
  layout_eeg = '
  #ABC#
  DEFGH
  #I#J#
  #KLM#
  #NOP#'
  
  p_multi_zeta_dat = 
    p_multi_spatial_zeta_ls$F3 + p_multi_spatial_zeta_ls$Fz + 
    p_multi_spatial_zeta_ls$F4 + p_multi_spatial_zeta_ls$T7 + 
    p_multi_spatial_zeta_ls$C3 + p_multi_spatial_zeta_ls$Cz + 
    p_multi_spatial_zeta_ls$C4 + p_multi_spatial_zeta_ls$T8 + 
    p_multi_spatial_zeta_ls$CP3 + p_multi_spatial_zeta_ls$CP4 + 
    p_multi_spatial_zeta_ls$P3 + p_multi_spatial_zeta_ls$Pz + 
    p_multi_spatial_zeta_ls$P4 + p_multi_spatial_zeta_ls$PO7 + 
    p_multi_spatial_zeta_ls$Oz + p_multi_spatial_zeta_ls$PO8 +
    plot_layout(design=layout_eeg, guides='collect') & theme(legend.position = 'bottom')
  output_dir = paste('K', design_num, '_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
                     '_zeta_', zeta_0, '_multi_channel_', e, '_zeta_estimate', sep="")
  ggsave(file.path(parent_path_plot, 'fit_plots', paste(output_dir, '.png', sep="")),
         p_multi_zeta_dat, width=300, height=250, units='mm', dpi=400)
  
}






# Prediction Accuracy Comparison
parent_path_bayes = Sys.glob(
  file.path(parent_path, '*data*', '*BayesGenq2Pred*',
            paste('*', design_num, '*', sep=''))
)
parent_path_cnn = Sys.glob(
  file.path(parent_path, '*data*', '*CNN*',
            paste('*', design_num, '*', sep=''))
)
parent_path_svc = Sys.glob(
  file.path(parent_path, '*data*', '*SVC*',
            paste('*', design_num, '*', sep=''))
)
parent_path_logistic = Sys.glob(
  file.path(parent_path, '*data*', '*LR*',
            paste('*', design_num, '*', sep=''))
)
parent_path_rf = Sys.glob(
  file.path(parent_path, '*data*', '*RF*',
            paste('*', design_num, '*', sep=''))
)
parent_path_swlda = Sys.glob(
  file.path(parent_path, '*data*', '*EEGswLDA*',
            paste('*', design_num, '*', sep=''))
)

predict_bayes = predict_cnn = predict_svc = predict_logistic = predict_rf = 
  predict_swlda = channel_vec = seq_vec = NULL
es = c(2, 3, 4, 5, 16)
for (e in es) {
  if (e == num_electrode) {
    channel_name_e = 'all_channels'
  } else {
    channel_name_e = multi_channel_simple[e]
  }
  channel_name_official_e = multi_channel_official[e]
  print(channel_name_e)

  bayes_e_dir = Sys.glob(file.path(
    parent_path_bayes, channel_name_e,
    paste('*', data_type, '_train_', num_rep_fit, '_pred_test_', num_rep_pred,
          '_binary_down_', dec_factor, '_raw_bp_',
          bp_low, '_', bp_upp, '_zeta_', zeta_0, '.csv*', sep=""))
  )

  predict_bayes_e = read.csv(bayes_e_dir, header=F)
  predict_bayes_e_sub = predict_bayes_e[182, 1:7]
  predict_bayes = c(predict_bayes, as.numeric(as.matrix(predict_bayes_e_sub)))
  
  cnn_e_dir = Sys.glob(file.path(
    parent_path_cnn, channel_name_e,
    paste('*train_', num_rep_fit, '_pred_', data_type, '_', num_rep_pred,
          '_ML_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
          '.csv*', sep=""))
  )
  predict_cnn_e = read.csv(cnn_e_dir, header = F)
  predict_cnn_e_sub = predict_cnn_e[25:43, 2:8]
  predict_cnn_e_compare = apply(predict_cnn_e_sub == target_letters, 2, sum)
  predict_cnn = c(predict_cnn, as.numeric(predict_cnn_e_compare))
  
  svc_e_dir = Sys.glob(file.path(
    parent_path_svc, channel_name_e,
    paste('*train_', num_rep_fit, '_pred_', data_type, '_', num_rep_pred,
          '_ML_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
          '.csv*', sep=""))
  )
  predict_svc_e = read.csv(svc_e_dir, header = F)
  predict_svc_e_sub = predict_svc_e[25:43, 2:8]
  predict_svc_e_compare = apply(predict_svc_e_sub == target_letters, 2, sum)
  predict_svc = c(predict_svc, as.numeric(predict_svc_e_compare))
  
  logistic_e_dir = Sys.glob(file.path(
    parent_path_logistic, channel_name_e,
    paste('*train_', num_rep_fit, '_pred_', data_type, '_', num_rep_pred,
          '_ML_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
          '.csv*', sep=""))
  )
  predict_logistic_e = read.csv(logistic_e_dir, header = F)
  predict_logistic_e_sub = predict_logistic_e[25:43, 2:8]
  predict_logistic_e_compare = apply(predict_logistic_e_sub == target_letters, 2, sum)
  predict_logistic = c(predict_logistic, as.numeric(predict_logistic_e_compare))
  
  rf_e_dir = Sys.glob(file.path(
    parent_path_rf, channel_name_e,
    paste('*train_', num_rep_fit, '_pred_', data_type, '_', num_rep_pred,
          '_ML_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
          '.csv*', sep=""))
  )
  predict_rf_e = read.csv(rf_e_dir, header = F)
  predict_rf_e_sub = predict_rf_e[25:43, 2:8]
  predict_rf_e_compare = apply(predict_rf_e_sub == target_letters, 2, sum)
  predict_rf = c(predict_rf, as.numeric(predict_rf_e_compare))
  
  swlda_e_dir = Sys.glob(file.path(
    parent_path_swlda, channel_name_e,
    paste('*train_', num_rep_fit, '_pred_', data_type, '_', num_rep_pred,
          '_ML_down_', dec_factor, '_from_raw_bp_', bp_low, '_', bp_upp,
          '_zeta_0.7.csv*', sep=""))
  )
  predict_swlda_e = read.csv(swlda_e_dir, header = F)
  predict_swlda_e_sub = predict_swlda_e[25:43, 2:8]
  predict_swlda_e_compare = apply(predict_swlda_e_sub == target_letters, 2, sum)
  predict_swlda = c(predict_swlda, as.numeric(predict_swlda_e_compare))
  
  channel_vec = c(channel_vec, rep(channel_name_official_e, 7))
  seq_vec = c(seq_vec, 1:7)
}

predict_multi_dat = data.frame(
  bayes = signif(predict_bayes / num_letter, digits=2),
  cnn = signif(predict_cnn /num_letter, digits=2),
  svc = signif(predict_svc / num_letter, digits=2),
  logistic = signif(predict_logistic / num_letter, digits=2),
  rf = signif(predict_rf / num_letter, digits=2),
  swlda = signif(predict_swlda / num_letter, digits=2),
  channel = channel_vec,
  seq = seq_vec
)

predict_multi_dat_2 = data.frame(
  value = c(predict_multi_dat$bayes, predict_multi_dat$cnn,
            predict_multi_dat$svc, predict_multi_dat$logistic,
            predict_multi_dat$rf, predict_multi_dat$swlda),
  channel = rep(predict_multi_dat$channel, 6),
  seq = rep(predict_multi_dat$seq, 6),
  method = rep(c('Bayes', 'Neural Network',
                 'Support Vector Machine', 'Logistic Regression',
                 'Random Forest', 'swLDA'), each=nrow(predict_multi_dat))
)

print(predict_multi_dat_2[predict_multi_dat_2$seq==6,])
output_dir_2 = paste('K', design_num, '_down_', dec_factor, '_raw_bp_', bp_low, '_', bp_upp,
                     '_zeta_', zeta_0, '_multi_channel_test_prediction', sep="")

write.table(
  x=predict_multi_dat,
  file=file.path(parent_path_plot, 'fit_plots',
                 paste(output_dir_2, '.txt', sep="")),
  row.names=F
)

write.csv(
  x=predict_multi_dat,
  file=file.path(parent_path_plot, 'fit_plots',
                 paste(output_dir_2, '.csv', sep="")),
  row.names=F
)
