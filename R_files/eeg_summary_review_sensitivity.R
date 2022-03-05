rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
local_use = T
# local_use = (args[1] == 'T')

library(readxl)
library(ggplot2)
library(gridExtra)
library(R.matlab)
library(dplyr)


parent_path = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation'
source(file.path(parent_path, 'Dataset and Rcode', 'Chapter_1', 'EEGConvolution2', 
                 'R_files', 'source_files', 'eeg_global_constant.R'))
source(file.path(parent_path, 'Dataset and Rcode', 'Chapter_1', 'EEGConvolution2', 
                 'R_files', 'source_files', 'additional_fun.R'))

# import channel selection results based on different bandpass filters.
file_dir = file.path(parent_path, 'manuscript', 'Chapter_1', 
                     'First-round review 2021-08-06', 'channel_selection.xlsx')
channel_select_df = read_excel(Sys.glob(file_dir), sheet='channel_selection')
channel_select_df = channel_select_df[,1:19]

output_rank_top_n(channel_select_df, 0.6, 6.5, 5)


# import channel selection results based on different kernel hyper-parameters.
file_dir_2 = file.path(parent_path, 'manuscript', 'Chapter_1', 
                     'First-round review 2021-08-06', 'channel_selection_2.xlsx')
channel_select_2_df = read_excel(Sys.glob(file_dir_2), col_names=F, sheet='channel_selection_2')
channel_select_2_df = channel_select_2_df[,1:19]
colnames(channel_select_2_df) = c('scale', 'gamma', 'subject', 1:16)
output_rank_top_n_2(channel_select_2_df, 0.6, 1.9, 5)

# ERP function estimates and SMTW plots for channels Cz and PO8.
beta_df_hyper = data.frame(mean=NULL, low=NULL, upp=NULL, zeta=NULL, channel=NULL,
                           type=NULL, time=NULL, hyper=NULL)
channel_id_name = 'PO8'
for (scale_iid in c(0.4, 0.5, 0.6)) {
  for (gamma_iid in c(1.7, 1.8, 1.9)) {
    beta_df_hyper = rbind.data.frame(
      beta_df_hyper,
      output_summary_df('K114', scale_iid, gamma_iid, 1)
    )
  }
}

beta_df_hyper$type = factor(beta_df_hyper$type, levels=c('Target', 'Non-target'))
beta_df_hyper$zeta_60 = ifelse(beta_df_hyper$zeta >= 0.6, 0.6, -1)
beta_df_hyper$zeta_75 = ifelse(beta_df_hyper$zeta >= 0.75, 0.75, -1)
beta_df_hyper$zeta_90 = ifelse(beta_df_hyper$zeta >= 0.90, 0.90, -1)

time_upper_lim = 800
y_upper_lim = ceiling(max(beta_df_hyper$upp))
y_lower_lim = floor(min(beta_df_hyper$low))

p_beta_df_hyper = ggplot(
  data=beta_df_hyper, aes(x=time, y=mean)) +
  geom_line(size=1.0, aes(x=time, y=mean, color=type)) +
  facet_wrap(~hyper, nrow=3) +
  geom_ribbon(aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.5) +
  geom_point(size=1.0, aes(x=time, y=mean, color=type), shape=1) +
  scale_x_continuous(limits=c(0, time_upper_lim),
                     breaks=seq(0, time_upper_lim, length.out=5)) +
  scale_y_continuous(limits=c(y_lower_lim, y_upper_lim), 
                     breaks=seq(y_lower_lim, y_upper_lim, by=2)) +
  xlab('Time (ms)') + ylab('Amplitude (muV)') + ggtitle('Estimated P300-ERPs') +
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
p_beta_df_hyper


zeta_df_hyper = data.frame(
  zeta=NULL, channel=NULL, type=NULL, time=NULL, 
  hyper=NULL, category=NULL)
for (hyper_iid in unique(beta_df_hyper$hyper)) {
  zeta_df_hyper = rbind.data.frame(
    zeta_df_hyper, data.frame(
      zeta = c(beta_df_hyper[beta_df_hyper$hyper==hyper_iid & beta_df_hyper$type=='Target', 'zeta_60'],
               beta_df_hyper[beta_df_hyper$hyper==hyper_iid & beta_df_hyper$type=='Target', 'zeta_75'],
               beta_df_hyper[beta_df_hyper$hyper==hyper_iid & beta_df_hyper$type=='Target', 'zeta_90']),
      channel = unique(beta_df_hyper$channel),
      type = 'Target',
      time = rep(beta_df_hyper$time[1:50], 3),
      hyper = hyper_iid,
      category = rep(c('zeta_60', 'zeta_75', 'zeta_90'), each=50)
    )
  )
}
zeta_df_hyper$category = factor(zeta_df_hyper$category, levels=c('zeta_60', 'zeta_75', 'zeta_90'),
                                labels=paste(c(60, 75, 90), '% Confidence', sep=''))

p_zeta_df_hyper = ggplot(
  data=zeta_df_hyper, aes(x=time, y=zeta)) +
  geom_line(size=2.0, aes(x=time, y=zeta, color=category)) +
  facet_wrap(~hyper, nrow=3) +
  scale_x_continuous(limits=c(0, time_upper_lim),
                     breaks=seq(0, time_upper_lim, length.out=5)) +
  scale_y_continuous(limits=c(0.5, 1), breaks=seq(0.5, 1, by=0.1)) +
  xlab('Time (ms)') + ylab('Probability') + ggtitle('Split Probabilities') +
  theme(plot.title=element_text(hjust=0.5),
        panel.background=element_rect(fill = "white",
                                      colour = "black",
                                      size = 0.5, linetype = "solid"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        legend.position='bottom',
        legend.background=element_rect(fill='transparent', size=0.2,
                                       color='white', linetype='solid'),
        plot.margin=margin(.25, .25, .25, .25, 'cm'))
p_zeta_df_hyper

p_hyper_inference = grid.arrange(p_beta_df_hyper, p_zeta_df_hyper, nrow=1)
output_dir = paste('sensitivity_inference_channel_', channel_id_name, '_plot', sep='')
ggsave(file.path(parent_path, 'manuscript', 'Chapter_1', 
                 'First-round review 2021-08-06', paste(output_dir, '.png', sep="")),
       p_hyper_inference, width=300, height=150, units='mm', dpi=400)


# prediction accuracy comparison
predict_df_hyper = data.frame(
  scale=NULL, gamma=NULL, accuracy=NULL
)
predict_path = file.path(parent_path, 'Dataset and Rcode', 'EEG_MATLAB_data', 
                         'BayesGenq2Pred', 'K114', 'channel_15_14_16_13_6')
for (scale_iid in c(0.4, 0.5, 0.6)) {
  for (gamma_iid in c(1.7, 1.8, 1.9)) {
    predict_dir = file.path(
      predict_path, paste('scale=', scale_iid, ', gamma=', gamma_iid, sep=''),
      '*test_train_7_pred_test_7_binary_down_4_raw_bp_0.5_6_zeta_0.4.csv*'
    )
    predict_csv = read.csv(Sys.glob(predict_dir), header=F)
    predict_df_hyper = rbind.data.frame(
      predict_df_hyper, 
      data.frame(scale=scale_iid, gamma=gamma_iid, 
                 accuracy=as.numeric(as.character(predict_csv[176, 6]))) 
    )
  }
}
predict_df_hyper$accuracy = predict_df_hyper$accuracy / 19
predict_df_hyper

