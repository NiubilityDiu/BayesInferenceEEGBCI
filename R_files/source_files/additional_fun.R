parent_path_2 = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/Dataset and Rcode/EEG_MATLAB_data/TRN_files/'
channel_name_short = c('F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz')
library(R.matlab)

output_rank_top_n = function(input_df, low, upp, top_n) {
  input_df_sub = input_df %>% filter(Low==low, Upp==upp)
  input_mat_sub = as.matrix(input_df_sub[,4:(4+top_n-1)], nrow=10*top_n, ncol=1)
  input_mat_sub = factor(input_mat_sub, levels=1:16, labels=channel_name_short)
  return (sort(table(input_mat_sub), decreasing = T))
}

output_rank_top_n_2 = function(input_df, scale_val, gamma_val, top_n) {
  input_df_sub = input_df %>% filter(scale==scale_val, gamma==gamma_val)
  input_mat_sub = as.matrix(input_df_sub[,4:(4+top_n-1)], nrow=10*top_n, ncol=1)
  input_mat_sub = factor(input_mat_sub, levels=1:16, labels=channel_name_short)
  return (sort(table(input_mat_sub), decreasing = T))
}


output_summary_df = function(k_sub_name, scale_val, gamma_val, channel_id) {
  time_upper_lim = 800
  channel_names_sub = c(15, 14, 16, 13, 6)
  iid = which(channel_names_sub == channel_id)
  file_dir = file.path(
    parent_path_2, k_sub_name, 'BayesGenq2', 'channel_15_14_16_13_6',
    paste('scale=', scale_val, ', gamma=', gamma_val, sep=''),
    '*continuous_down_4_raw_bp_0.5_6.mat*'
  )
  mcmc_df = readMat(Sys.glob(file_dir))
  # print(names(mcmc_df))
  # print(dim(mcmc_df$beta.tar))
  beta_tar_ee = mcmc_df$beta.tar[, iid, , 1]
  beta_ntar_ee = mcmc_df$beta.ntar[, iid, , 1]
  zeta_ee = mcmc_df$zeta[, iid, ]
  time_vec = seq(0, 1000/32*25, length.out=50)
  beta_df = data.frame(
    mean = c(apply(beta_tar_ee, 2, mean), apply(beta_ntar_ee, 2, mean)),
    low = c(apply(beta_tar_ee, 2, quantile, 0.05), 
            apply(beta_ntar_ee, 2, quantile, 0.05)),
    upp = c(apply(beta_tar_ee, 2, quantile, 0.95), 
            apply(beta_ntar_ee, 2, quantile, 0.95)),
    zeta = rep(apply(zeta_ee, 2, median), 2),
    channel = channel_name_short[channel_names_sub[iid]],
    type = rep(c('Target', 'Non-target'), each=50),
    time = rep(time_vec, 2),
    hyper = paste('scale=', scale_val, ', gamma=', gamma_val, sep="")
  )
  return (beta_df)
}
