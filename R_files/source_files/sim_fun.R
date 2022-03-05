library(reticulate)
# library(readxl)

# global constants
candidate_letters = c(toupper(letters), 1:5, 'SPEAK', '.', 'BS', '!', '_')
channel_name_official = c(
  'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
  'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz')


# source functions
create_design_mat = function(n_multiple, flash_and_pause_length, repet_num) {
  n_length = n_multiple * flash_and_pause_length
  dm_row = (repet_num * NUM_REP + n_multiple - 1) * flash_and_pause_length
  dm_col = repet_num * NUM_REP * n_length
  dm = matrix(0, nrow=dm_row, ncol=dm_col)
  for (trial_id in 0:(repet_num * NUM_REP-1)) {
    row_id_low = trial_id * flash_and_pause_length + 1
    row_id_upp = row_id_low + n_length - 1
    col_id_low = trial_id * n_length + 1
    col_id_upp = col_id_low + n_length - 1
    dm[row_id_low:row_id_upp, col_id_low:col_id_upp] = diag(x=1, nrow=n_length)
  }
  return (dm)
}


create_transform_mat = function(eeg_type, letter_dim, repet_num, n_length) {
  left_zero_mat = matrix(0, nrow=NUM_REP * n_length * repet_num * letter_dim,
                         ncol=n_length)
  right_diag_mat = do.call(rbind, replicate(NUM_REP * repet_num * letter_dim, 
                                            diag(x=1, nrow=n_length), simplify=F))
  target_ids = which(eeg_type == 1)
  target_ids_len = length(eeg_type)
  for (id in 0:(target_ids_len-1)) {
    row_low_i = id * n_length + 1
    row_upp_i = row_low_i + n_length - 1
    left_zero_mat[row_low_i:row_upp_i,] = diag(nrow=n_length)
    right_diag_mat[row_low_i:row_upp_i,] = matrix(0, nrow=n_length, ncol=n_length)
  } 
  transform_mat = cbind(left_zero_mat, right_diag_mat)
  transform_mat = reticulate::array_reshape(
    transform_mat, dim=c(letter_dim, repet_num * NUM_REP * n_length, 2 * n_length), order='C'
  )
  return (transform_mat)
}


# compute frequency table for single dataset
bayes_freq_table_subset_id = function(file_dir, rep_num, num_letter) {
  
  dat_result = read.csv(file_dir, header=F)
  pred_result_table = matrix(0, nrow=num_letter, ncol=rep_num)
  
  for (l in 1:num_letter) {
    for (rep_id in 1:rep_num) {
      row_num = (rep_num + 2) * l - (rep_num - 2)  # depends on rep_num
      pred_result_table[l, rep_id] = as.character(dat_result[row_num+rep_id, 2])
    }
  }
  freq_pred_accuracy = sapply(1:rep_num, function(x) {id_bool = pred_result_table[,x] == target_letters 
    return (sum(id_bool))})
  return (freq_pred_accuracy)
}


bayes_freq_table_design_num = function(
  parent_path, method_name, design_num, dataset_num, data_type, 
  rep_num_fit, rep_num_pred, scenario_name, 
  zeta, num_letter, analysis_type
) {
  
  bayes_design = NULL
  
  if (analysis_type == 'simulation_single') {
    file_name_suffix = '_binary_down_gamma_exp_zeta_'
  } else {
    file_name_suffix = '_binary_down_1_fit_zeta_'
  } 
  for (subset_id in 1:dataset_num) {
    key_part = file.path(parent_path, "EEG_MATLAB_data", 
                     paste0('*', method_name, '*', sep=""), 
                     paste0('sim_', design_num, sep=""),
                     paste0('sim_', design_num, '_dataset_', subset_id, sep=""), 
                     paste0('*', scenario_name, '*', sep=""),
                     paste0('*', data_type, "_train_", rep_num_fit, '_pred_', data_type, '_',
                            rep_num_pred, file_name_suffix, zeta, '.csv*', sep=""))

    bayes_dat_id_dir = Sys.glob(key_part)
    if (analysis_type == 'simulation_single') {
      bayes_subset_id = bayes_freq_table_subset_id(bayes_dat_id_dir, num_rep_pred, num_letter)
    } else {
      # Use existing information from the csv file
      bayes_subset_id_csv = read.csv(bayes_dat_id_dir, header=F)
      # based on row and column criterion
      bayes_subset_id = as.numeric(as.matrix(bayes_subset_id_csv[144,1:rep_num_pred]))
    }
    bayes_design = rbind(bayes_design, bayes_subset_id / num_letter)
  } 
  bayes_design_mean = apply(bayes_design, 2, mean)
  bayes_design_sd = apply(bayes_design, 2, sd)
  
  return (list(prob = bayes_design,
               mean = bayes_design_mean,
               sd = bayes_design_sd))
}


exist_ml_freq_table_subset_id = function(file_dir, rep_num, num_letter) {
  
  dat_result = read.csv(file_dir, header=F)
  pred_result_table = dat_result[(num_letter+6):(2*num_letter+5), 2:(1+rep_num)]
  if (rep_num > 1) {
    freq_pred_accuracy = sapply(1:rep_num, function(x) 
    {id_bool = pred_result_table[,x] == target_letters 
      return (mean(id_bool))})
  } else {
    freq_pred_accuracy = (pred_result_table == target_letters) * 1
  }
  return (freq_pred_accuracy)
}


exist_ml_freq_table_design_num = function(
  parent_path, method_name, design_num, dataset_num, data_type, 
  num_rep_fit, num_rep_pred, scenario_name, zeta, num_letter
) {

  exist_ml_design = NULL
  for (subset_id in 1:dataset_num) {
    temp_id_dir = file.path(parent_path, '*EEG_MATLAB_data*', method_name,
                            paste0('sim_', design_num, sep=""),
                            paste0('sim_', design_num, '_dataset_', subset_id, sep=""),
                            paste0('*', scenario_name, '*', sep=""),
                            paste0('*train_', num_rep_fit, '_pred_', data_type, '_',
                                   num_rep_pred, sep=""))
    if (zeta > 0) {
      temp_id_dir_2 = paste(temp_id_dir, '_down_zeta_', zeta, '*', sep="")
    } else {
      temp_id_dir_2 = paste(temp_id_dir, '*', sep="")
    }
    ml_dat_id_dir = Sys.glob(temp_id_dir_2)
    exist_ml_subset_id = exist_ml_freq_table_subset_id(
      ml_dat_id_dir, num_rep_pred, num_letter
    )
    exist_ml_design = rbind(exist_ml_design, exist_ml_subset_id)
  }
  
  exist_ml_design_mean = apply(exist_ml_design, 2, mean)
  exist_ml_design_sd = apply(exist_ml_design, 2, sd)
  
  return (list(prob = exist_ml_design,
               mean = exist_ml_design_mean,
               sd = exist_ml_design_sd))
}


extract_sim_est_fun = function(mcmc_dat, prob_vec) {
  beta_tar_mcmc = mcmc_dat$beta.tar[, 1, , 1]
  beta_ntar_mcmc = mcmc_dat$beta.ntar[, 1, , 1]
  beta_tar_mean = apply(beta_tar_mcmc, 2, mean)
  beta_ntar_mean = apply(beta_ntar_mcmc, 2, mean)
  beta_tar_ci = apply(beta_tar_mcmc, 2, quantile, prob_vec)
  beta_ntar_ci = apply(beta_ntar_mcmc, 2, quantile, prob_vec)
  
  return (list(tar_mean = beta_tar_mean,
               ntar_mean = beta_ntar_mean,
               tar_ci = beta_tar_ci,
               ntar_ci = beta_ntar_ci))
}


import_extract_sim_est_fun = function(
  parent_sim_data_path, design_num, dataset_id, 
  scenario_name, prob_vec
) {
  mcmc_dir = Sys.glob(
    file.path(parent_sim_data_path, 'single_channel', paste('case', design_num, sep=""), 
              paste('*dataset_', dataset_id, '*', sep=""), '*q2*', scenario_name, '*mat*')
  )
  print(mcmc_dir)
  mcmc_dat = readMat(mcmc_dir)
  mcmc_est = extract_sim_est_fun(mcmc_dat, prob_vec)
  return (mcmc_est)
}


extract_sim_est_fun_multi = function(mcmc_dat, prob_vec) {
  beta_tar_mcmc = mcmc_dat$beta.tar[, , , 1]
  beta_ntar_mcmc = mcmc_dat$beta.ntar[, , , 1]
  beta_tar_mean = apply(beta_tar_mcmc, c(2, 3), mean)
  beta_ntar_mean = apply(beta_ntar_mcmc, c(2, 3), mean)
  beta_tar_ci = apply(beta_tar_mcmc, c(2, 3), quantile, prob_vec)
  beta_ntar_ci = apply(beta_ntar_mcmc, c(2, 3), quantile, prob_vec)
  
  return (list(tar_mean = beta_tar_mean,
               ntar_mean = beta_ntar_mean,
               tar_ci = beta_tar_ci,
               ntar_ci = beta_ntar_ci))
}


import_extract_sim_est_fun_multi = function(mcmc_dir, prob_vec) {
  # mcmc_dir = Sys.glob(
  #   file.path(parent_sim_data_path, paste('*', design_num, '*', sep=""),
  #             paste('*dataset_', dataset_id, sep=""), '*q2*', scenario_name,
  #             paste('*zeta_', zeta_bayes, '.mat*', sep=""))
  # )
  print(mcmc_dir)
  mcmc_dat = readMat(mcmc_dir)
  print(names(mcmc_dat))
  mcmc_est = extract_sim_est_fun_multi(mcmc_dat, prob_vec)
  return (mcmc_est)
}


produce_row_col_seq_dat = function(
  row_cum, col_cum, seq_id, channel_e, num_letter, mcmc_iter_num,
  target_letter_rows, target_letter_cols
) {
  
  row_cum_seq = as.vector(t(row_cum[, seq_id, , 1]))
  row_cum_dat = data.frame(
    value = row_cum_seq,
    letter_id = rep(1:num_letter, each=mcmc_iter_num),
    true_id = rep(target_letter_rows, each=mcmc_iter_num),
    channel = channel_e,
    seq_id = seq_id
  )
  row_cum_dat$value = factor(row_cum_dat$value, levels=1:6)
  
  col_cum_seq = as.vector(t(col_cum[, seq_id, , 1]))
  col_cum_dat = data.frame(
    value = col_cum_seq,
    letter_id = rep(1:num_letter, each=mcmc_iter_num),
    true_id = rep(target_letter_cols, each=mcmc_iter_num),
    channel = channel_e,
    seq_id = seq_id
  )
  col_cum_dat$value = factor(col_cum_dat$value, levels=7:12)
  
  return (list(row = row_cum_dat,
               col = col_cum_dat))
}


produce_row_col_plot_list = function(
  row_col_dat, num_letter, mcmc_iter_num, subject_id, target_letters, true_row_col_ids
) {
  p_row_col_cum_ls = lapply(1:num_letter, function(x) NULL)
  for (l_id in 1:num_letter) {
    p_row_col_cum_ls[[l_id]] = ggplot(
      data=row_col_dat[row_col_dat$letter_id == l_id,]) +
      geom_bar(aes(x=value)) + scale_x_discrete(drop=FALSE) +
      scale_y_continuous(limits=c(0, mcmc_iter_num), breaks=seq(0, mcmc_iter_num, by=20)) +
      ylab('Frequency') +
      ggtitle(paste(subject_id, ', target_letter: ', target_letters[l_id], 
                    ', true_id: ', true_row_col_ids[l_id], sep="")) +
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
  }
  return (p_list = p_row_col_cum_ls)
}


multi_channel_names = function(input_e_vec) {
  e_len = length(input_e_vec)
  output_e_vec = lapply(1:e_len, function(x) NULL)
  output_e_vec_official = lapply(1:e_len, function(x) NULL)
  for (i in 1:e_len) {
    output_e_vec[[i]] = paste('channel', paste(input_e_vec[1:i], collapse="_"), sep='_')
    output_e_vec_official[[i]] = paste(channel_name_official[input_e_vec[1:i]], collapse = ',')
  }
  return (list(
    num = as.character(output_e_vec),
    official = as.character(output_e_vec_official)
    ))
}


create_erp_dataset_subject = function(
  parent_path, design_num, multi_channel_num,
  multi_channel_simple, dec_factor, bp_low, bp_upp, n_length_fit
) {
  e = 16
  beta_vec_mean = beta_vec_upp = beta_vec_low = channel_vec = type_vec = time_vec =NULL
  zeta_vec_median = NULL
  channel_name_num_e = multi_channel_simple[e]
  
  parent_path_plot = Sys.glob(
    file.path(parent_path, '*data*', '*TRN_files*', paste('*', design_num, '*', sep=''))
  )
  dir.create(file.path(parent_path_plot, 'fit_plots'), showWarnings = F)
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
  
  return (mcmc_multi_dat)
}


produce_erp_plot_ls = function(
  mcmc_multi_dat, design_num, time_upper_lim, y_upper_lim, y_lower_lim, 
  channel_name_short
) {
  p_multi_spatial_ls = lapply(1:16, function(x) NULL)
  names(p_multi_spatial_ls) = channel_name_short
  for(e in 1:16) {
    cee = channel_name_short[e]
    p_multi_spatial_ls[[e]] = ggplot(
      data=mcmc_multi_dat %>% filter(channel==cee), aes(x=time, y=mean)) +
      geom_line(size=1.0, aes(x=time, y=mean, color=type)) +
      # geom_hline(yintercept=0, linetype=2, alpha=0.5) + 
      geom_ribbon(aes(x=time, ymin=low, ymax=upp, fill=type), alpha=0.5) +
      geom_point(size=1.0, aes(x=time, y=mean, color=type), shape=1) +
      scale_x_continuous(limits=c(0, time_upper_lim),
                         breaks=seq(0, time_upper_lim, length.out=5)) +
      scale_y_continuous(limits=c(y_lower_lim, y_upper_lim), 
                         breaks=seq(y_lower_lim, y_upper_lim, length.out=5)) +
      xlab('') + ylab('') + ggtitle(paste('Participant', design_num, sep=' ')) +
      theme(plot.title=element_text(hjust=0.5, size=10),
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
  
  return (p_multi_spatial_ls)
}
