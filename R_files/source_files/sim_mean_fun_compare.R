# beta_tar_opt_est = mcmc_dat_opt$beta.tar[, 1, , 1]
# beta_ntar_opt_est = mcmc_dat_opt$beta.ntar[, 1, , 1]
# beta_tar_opt_mean = apply(beta_tar_opt_est, 2, mean)
# beta_ntar_opt_mean = apply(beta_ntar_opt_est, 2, mean)
# beta_tar_opt_low = apply(beta_tar_opt_est, 2, quantile, 0.025)
# beta_tar_opt_upp = apply(beta_tar_opt_est, 2, quantile, 0.975)
# beta_ntar_opt_low = apply(beta_ntar_opt_est, 2, quantile, 0.025)
# beta_ntar_opt_upp = apply(beta_ntar_opt_est, 2, quantile, 0.975)
# 
# mean_fn_dat_opt = data.frame(
#   time = rep(seq(0, time_length, length.out=n_length_fit), 2),
#   true_value = c(mean_fn_tar, mean_fn_ntar),
#   mean_value = c(beta_tar_opt_mean, beta_ntar_opt_mean),
#   low_value = c(beta_tar_opt_low, beta_ntar_opt_low),
#   upp_value = c(beta_tar_opt_upp, beta_ntar_opt_upp),
#   category = rep(c('target', 'non-target'), each=n_length_fit)
# )
# 
# plot_title = paste('Sim', scenario_name, 'Case', design_num, 'Dataset', dataset_id, 'Train', num_rep_fit, 'Opt', sep='-')
# p1 = ggplot(mean_fn_dat_opt, aes(time, true_value))
# p1 = p1 + geom_point(size=1.5, aes(color=factor(category))) +
#   geom_line(size=1, aes(color=factor(category))) +
#   geom_ribbon(aes(ymin=low_value, ymax=upp_value, fill=factor(category)), alpha=0.25) +
#   xlab('Time (ms)') + ylab("Amplitude (muV)") +
#   # ylim(c(-2, 10)) +
#   scale_x_continuous(breaks=as.integer(seq(0, time_length, length.out=5))) +
#   ggtitle('True Curves and Credible Bands') +
#   theme(plot.title=element_text(hjust=0.5),
#         panel.background = element_rect(fill = "white",
#                                         colour = "black",
#                                         size = 0.5, linetype = "solid"),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         legend.position = c(0.8, 0.8),
#         legend.title = element_blank(),
#         legend.background = element_rect(fill="transparent",
#                                          size=0.25,
#                                          color='white', linetype='solid'),
#         plot.margin = margin(1, 1, 1, 1, "cm"))
# 
# mean_fn_dat_opt_2 = data.frame(
#   time = rep(seq(0, time_length, length.out=n_length_fit), 4),
#   value = c(mean_fn_tar, mean_fn_ntar, beta_tar_opt_mean, beta_ntar_opt_mean),
#   category = rep(c('target-true', 'non-target-true',
#                    'target-est', 'non-target-est'), each=n_length_fit)
# )
# 
# p2 = ggplot(mean_fn_dat_opt_2, aes(time, value))
# p2 = p2 + geom_point(size=1.5, aes(time, value, color=factor(category))) +
#   geom_line(size=1, aes(time, value, color=factor(category))) +
#   xlab('Time (ms)') + ylab("Amplitude (muV)") +
#   scale_x_continuous(breaks=as.integer(seq(0, time_length, length.out=5))) +
#   ggtitle('True Curves and Posterior Means') +
#   theme(plot.title=element_text(hjust=0.5),
#         panel.background = element_rect(fill = "white",
#                                         colour = "black",
#                                         size = 0.5, linetype = "solid"),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         legend.position = c(0.8, 0.8),
#         legend.title = element_blank(),
#         legend.background = element_rect(fill="transparent",
#                                          size=0.25,
#                                          color='white', linetype='solid'),
#         plot.margin = margin(1, 1, 1, 1, "cm"))
# print(p2)
# 
# p = grid.arrange(p1, p2, ncol=2, top=plot_title)
# print(p)
# dir.create(file.path(parent_path_plot, 'Plots', scenario_name))
# ggsave(paste(parent_path_plot, '/Plots/', scenario_name, '/', plot_title, '.png', sep=""), p,
#        width=300, height=200, units='mm', dpi=300)