library(ggplot2)
library(gridExtra)
target_letters = c('T', 'H', 'E', '_',
                   'Q', 'U', 'I', 'C', 'K', '_',
                   'B', 'R', 'O', 'W', 'N', '_',
                   'F', 'O', 'X')
# target_letters = rep(target_letters, 2)
NUM_REP = 12
num_letter = length(target_letters)
# num_repetition = 15
n_length_fit = 30
# n_length_fit = 25
# dec_factor = 8
# time_length = n_length_fit / 32 * 1000  # down-sampling latency time length

display_bool = F

# True Gen
mean_fn_tar_1 = c(
  0, -0.2465, -0.5469, -0.4695, -0.1083,  
  0.4509,  1.1296,  1.8580,  2.5746, 3.2267,  
  3.7708, 4.1731, 4.4092,  4.4651,  4.3372,  
  4.0326,  3.5694,  2.9772, 2.2972,  1.5827, 
  0.8992,  0.3251, 0.05, 0.001,  0,
  0, 0, 0, 0, 0
)
mean_fn_ntar_1 = c(
  0, -0.04226, -0.125, -0.1,  0.0315,  
  0.1351,  0.2595,  0.3927,  0.5243, 0.6448,  
  0.7463,  0.8223,  0.8680,  0.8804,  0.8582,  
  0.8021,  0.7151,  0.6023, 0.4711,  0.3314,  
  0.1958,  0.0795,  0.0004, -0.0203,  -0.012,
  0, 0, 0, 0, 0
)
if (display_bool) {
  plot(1:length(mean_fn_tar_1), mean_fn_tar_1, type='l',
       xlab='Time', ylab="Magnitude", main='mean_fn_1', col="red")
  lines(1:length(mean_fn_ntar_1), mean_fn_ntar_1,
         xlab='Time', ylab="Magnitude", main='mean_fn_1', col="blue")
}


mean_fn_tar_2 = c(
  0., -0.4269, -1.1153, -1.7062, -1.9339,
  -1.6727, -0.9505,  0.1063,  1.3401, 2.6107, 
  3.8171,  4.8971,  5.8132,  6.5359, 7.0345, 
  7.2775,  7.2434,  6.9336, 6.3816,  5.4, 
  4.0,  2.8, 1.8,  1.0,  0.4, 
  0.2, 0, 0, 0, 0
)
mean_fn_ntar_2 = c(
  0, -0.42, -1.1, -1.7, -1.93, 
  -1.67, -1.1, 0, 1.34, 2.0,
  2.3, 2.5, 2.2, 1.5, 1.0,
  0.5, 0.25, 0.1, 0.05, 0,
  0, 0, 0, 0, 0,
  0, 0, 0, 0, 0
)
if (display_bool) {
  plot(1:length(mean_fn_tar_2), mean_fn_tar_2,
       xlab='Time', ylab="Magnitude", main='mean_fn_2', col="red")
  points(1:length(mean_fn_ntar_2), mean_fn_ntar_2,
         xlab='Time', ylab="Magnitude", main='mean_fn_2', col="blue")
}


# LatencyLen25
mean_fn_tar_3 = c(
  0, -0.388, -0.446, -0.020,  0.586,  
  1.428,  2.327,  3.176,  3.877,  4.354,
  4.550, 4.435, 4.007,  3.300,  2.382,  
  1.367,  0.6,  0.275,  0.150,  0.050,
  0.000,  0.000, 0.000,  0.000,  0.000
  )
mean_fn_ntar_3 = c(
  0, -0.05, -0.06,  -0.02,  0.12,
  0.3178,  0.4827,  0.6376, 0.7658,  0.8534,
  0.8904,  0.8709,  0.7944,  0.6661,  0.4975,
  0.3076, 0.1233, 0.06, 0.03,  0.01,
  0, 0, 0, 0, 0
  )
if (display_bool) {
  plot(1:length(mean_fn_tar_3), mean_fn_tar_3,
       xlab='Time', ylab="Magnitude", main='mean_fn_3', col="red")
  points(1:length(mean_fn_ntar_3), mean_fn_ntar_3,
         xlab='Time', ylab="Magnitude", main='mean_fn_3', col="blue")
}


mean_fn_tar_4 = c(
  0, -0.7361, -1.5, -1.9183, -1.4918,
  -0.4558, 0.8,  2.5851, 4.1239,  5.4274,  
  6.3923,  6.9847,  7.2221, 7.1381, 6.7296,  
  5.6,  4.6,  3.4, 2.0, 1.0,
  0.5, 0.1, 0, 0, 0
)
mean_fn_ntar_4 =c(
  0, -0.73, -1.5, -1.9, -1.5, 
  -0.5, 0.8, 1.6, 2.2, 2.4, 
  2.5, 2.4, 2.0, 1.4, 0.9,
  0.5, 0.1, 0, 0, 0,
  0, 0, 0, 0, 0 
)
if (display_bool) {
  plot(1:length(mean_fn_tar_4), mean_fn_tar_4,
       xlab='Time', ylab="Magnitude", main='mean_fn_4', col="red")
  points(1:length(mean_fn_ntar_4), mean_fn_ntar_4,
         xlab='Time', ylab="Magnitude", main='mean_fn_4', col="blue")
}


# LatencyLen35
mean_fn_tar_5 = c(
  0, -0.2, -0.3952, -0.4680, -0.3558, 
  -0.0831,  0.3274,  0.8443,  1.4281, 2.0370, 
  2.6324,  3.1825,  3.6623,  4.0533,  4.3405,  
  4.5104,  4.5504,  4.4496, 4.2017,  3.8096,  
  3.2899,  2.6762,  2.0202,  1.3869,  0.8455,  
  0.4536,  0.2572, 0.1703,  0.1,  0.0474, 
  0, 0, 0, 0, 0
)
mean_fn_ntar_5 = c(
  0,  -0.04, -0.08, -0.09, -0.08,  
  0.0354,  0.1141,  0.2103,  0.3174,  0.4289,  
  0.5380,  0.6390,  0.7272,  0.7990,  0.8514, 
  0.8823,  0.8897,  0.8723, 0.8290,  0.7599, 
  0.6669,  0.5541,  0.4291,  0.3025,  0.1877,  
  0.0989,  0.0468, 0.0326,  0.02,  0.0093,
  0, 0, 0, 0, 0
)
if (display_bool) {
  plot(1:length(mean_fn_tar_5), mean_fn_tar_5,
       xlab='Time', ylab="Magnitude", main='mean_fn_5', col="red")
  points(1:length(mean_fn_ntar_5), mean_fn_ntar_5,
         xlab='Time', ylab="Magnitude", main='mean_fn_5', col="blue")
}


mean_fn_tar_6 = c(
  0, -0.7544, -1.3544, -1.7876, -1.9299,
  -1.7358, -1.2238, -0.4538, 0.4941,  1.5370, 
  2.6007,  3.6256,  4.5675,  5.3958, 6.0896, 
  6.6339,  7.0171, 7.2302,  7.2669,  7.1258,  
  6.8122,  6.3391,  5.7274,  5.0034,  4.1960, 
  3.3355, 2.4592,  1.6321,  0.9944, 0.4,
  0, 0, 0, 0, 0
)
mean_fn_ntar_6 = c(
  0,  -0.75, -1.35, -1.8, -1.9, 
  -1.73, -1.22, -0.45, 0.5, 1.0, 1.5, 1.9, 2.2,
  2.4, 2.5, 2.4, 2.2, 1.9, 1.6, 1.2, 0.9,
  0.6, 0.25, 0.1, 0.05,
  0, 0, 0, 0, 0,
  0, 0, 0, 0, 0
) 
if (display_bool) {
  plot(1:length(mean_fn_tar_6), mean_fn_tar_6,
       xlab='Time', ylab="Magnitude", main='mean_fn_6', col="red")
  points(1:length(mean_fn_ntar_6), mean_fn_ntar_6,
         xlab='Time', ylab="Magnitude", main='mean_fn_6', col="blue")
}


mean_fn_tar_ls = list(
  tar_1 = mean_fn_tar_1,
  tar_2 = mean_fn_tar_2,
  tar_3 = mean_fn_tar_3,
  tar_4 = mean_fn_tar_4,
  tar_5 = mean_fn_tar_5,
  tar_6 = mean_fn_tar_6
)

mean_fn_ntar_ls = list(
  ntar_1 = mean_fn_ntar_1,
  ntar_2 = mean_fn_ntar_2,
  ntar_3 = mean_fn_ntar_3,
  ntar_4 = mean_fn_ntar_4,
  ntar_5 = mean_fn_ntar_5,
  ntar_6 = mean_fn_ntar_6
)

# # https://davetang.org/muse/2013/05/09/on-curve-fitting/
# # Use the curve to construct the new ones under affine transform
# # For MisLatencyLen25
# x = 1:20
# tar1_fit5 = lm(mean_fn_tar$mean_fn_tar_3[x] ~ poly(x, 5, raw=T))
# ntar1_fit5 = lm(mean_fn_ntar$mean_fn_ntar_3[x] ~ poly(x, 5, raw=T))
# xx = 1:25
# # plot(xx, mean_fn_tar$mean_fn_tar_1[xx], pch=19, ylim=c(-2,8))
# plot(xx, predict(tar1_fit5, data.frame(x=xx*4/5)))
# tar1_fit5_pred = as.vector(predict(tar1_fit5, data.frame(x=xx*4/5)))
# 
# # plot(xx, mean_fn_ntar$mean_fn_ntar_1[xx], pch=19, ylim=c(-1,1.5))
# plot(xx, predict(ntar1_fit5, data.frame(x=xx*4/5)), col="black")
# ntar1_fit5_pred = as.vector(predict(ntar1_fit5, data.frame(x=xx*4/5)))
# 
# # For MisLatencyLen35
# x = 1:20
# tar5_fit5 = lm(mean_fn_tar$mean_fn_tar_3[x] ~ poly(x, 10, raw=T))
# ntar5_fit5 = lm(mean_fn_ntar$mean_fn_ntar_3[x] ~ poly(x, 10, raw=T))
# xx = 1:30
# # plot(xx, mean_fn_tar$mean_fn_tar_1, pch=19, ylim=c(-2,8))
# plot(xx, predict(tar5_fit5, data.frame(x=xx * 2/3)))
# tar5_fit5_pred = as.vector(predict(tar5_fit5, data.frame(x=xx*2/3)))
# 
# # plot(xx, mean_fn_ntar$mean_fn_ntar_3[xx], pch=19, ylim=c(-1,1.5))
# plot(xx, predict(ntar5_fit5, data.frame(x=xx*2/3)), col="black")
# ntar5_fit5_pred = as.vector(predict(ntar5_fit5, data.frame(x=xx*2/3)))


# # For MisLatencyLen25
# x = 1:20
# tar2_fit5 = lm(mean_fn_tar$mean_fn_tar_4[x] ~ poly(x, 10, raw=T))
# ntar2_fit5 = lm(mean_fn_ntar$mean_fn_ntar_4[x] ~ poly(x, 10, raw=T))
# xx = 1:25
# # plot(xx, mean_fn_tar$mean_fn_tar_2[xx], pch=19, ylim=c(-2,8))
# plot(xx, predict(tar2_fit5, data.frame(x=xx*4/5)))
# tar2_fit5_pred = as.vector(predict(tar2_fit5, data.frame(x=xx*4/5)))
#
# plot(xx, mean_fn_ntar$mean_fn_ntar_1[xx], pch=19, ylim=c(-1,1.5))
# plot(xx, predict(ntar2_fit5, data.frame(x=xx*4/5)), col="black")
# ntar2_fit5_pred = as.vector(predict(ntar2_fit5, data.frame(x=xx*4/5)))
#
# # For MisLatencyLen35
# x = 1:20
# tar6_fit5 = lm(mean_fn_tar$mean_fn_tar_4[x] ~ poly(x, 10, raw=T))
# ntar6_fit5 = lm(mean_fn_ntar$mean_fn_ntar_4[x] ~ poly(x, 10, raw=T))
# xx = 1:30
# # plot(xx, mean_fn_tar$mean_fn_tar_1, pch=19, ylim=c(-2,8))
# plot(xx, predict(tar6_fit5, data.frame(x=xx * 2/3)))
# tar6_fit5_pred = as.vector(predict(tar6_fit5, data.frame(x=xx*2/3)))
#
# # plot(xx, mean_fn_ntar$mean_fn_ntar_3[xx], pch=19, ylim=c(-1,1.5))
# plot(xx, predict(ntar6_fit5, data.frame(x=xx*2/3)), col="black")
# ntar6_fit5_pred = as.vector(predict(ntar6_fit5, data.frame(x=xx*2/3)))
#

if (display_bool) {
  # plot three variations together
  x_lim = c(0, 1200); y_lim = c(-1, 5)
  dat_1_true_gen = data.frame(
    time = rep(seq(0, 30/32*1000, length.out=30), 2),
    value = c(mean_fn_tar_ls$tar_1, mean_fn_ntar_ls$ntar_1),
    category = rep(c('Target', 'Non-target'), each=30)
  )
  p1_true_gen = ggplot(dat_1_true_gen, aes(time, value, colour=factor(category)))
  p1_true_gen = p1_true_gen + geom_point(size=1.5) +
    geom_line(size=1, aes(colour=factor(category))) +
    xlab('Time (ms)') + ylab("Amplitude (muV)") + ggtitle('True Generative') +
    scale_x_continuous(limits=x_lim, breaks=as.integer(seq(x_lim[1], x_lim[2], by=200))) +
    scale_y_continuous(limits=y_lim, breaks=seq(y_lim[1], y_lim[2], by=1)) +
    theme(plot.title=element_text(hjust=0.5),
          panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = c(0.8, 0.8),
          legend.title = element_blank(),
          legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
          plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))
  print(p1_true_gen)
  
  
  dat_1_len_25 = data.frame(
    time = rep(seq(0, 25/32*1000, length.out=25), 2),
    value = c(mean_fn_tar_ls$tar_3, mean_fn_ntar_ls$ntar_3),
    category = rep(c('Target', 'Non-target'), each=25)
  )
  p1_len_25 = ggplot(dat_1_len_25, aes(time, value, colour=factor(category)))
  p1_len_25 = p1_len_25 + geom_point(size=1.5) +
    geom_line(size=1, aes(colour=factor(category))) +
    xlab('Time (ms)') + ylab("Amplitude (muV)") + ggtitle('Shorter Latency Length') +
    scale_x_continuous(limits=x_lim, breaks=as.integer(seq(x_lim[1], x_lim[2], by=200))) +
    scale_y_continuous(limits=y_lim, breaks=seq(y_lim[1], y_lim[2], by=1)) +
    theme(plot.title=element_text(hjust=0.5),
          panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = c(0.8, 0.8),
          legend.title = element_blank(),
          legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
          plot.margin = margin(.5, .5, .5, .5, "cm"))
  print(p1_len_25)
  
  
  dat_1_len_35 = data.frame(
    time = rep(seq(0, 35/32*1000, length.out=35), 2),
    value = c(mean_fn_tar_ls$tar_5, mean_fn_ntar_ls$ntar_5),
    category = rep(c('Target', 'Non-target'), each=35)
  )
  p1_len_35 = ggplot(dat_1_len_35, aes(time, value, colour=factor(category)))
  p1_len_35 = p1_len_35 + geom_point(size=1.5) +
    geom_line(size=1, aes(colour=factor(category))) +
    xlab('Time (ms)') + ylab("Amplitude (muV)") + ggtitle('Longer Latency Length') +
    scale_x_continuous(limits=x_lim, breaks=as.integer(seq(x_lim[1], x_lim[2], by=200))) +
    scale_y_continuous(limits=y_lim, breaks=seq(y_lim[1], y_lim[2], by=1)) +
    theme(plot.title=element_text(hjust=0.5),
          panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = c(0.8, 0.8),
          legend.title = element_blank(),
          legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
          plot.margin = margin(.5, .5, .5, .5, "cm"))
  print(p1_len_35)
  
  p1 = grid.arrange(p1_true_gen, p1_len_25, p1_len_35, ncol=3,
                    top='')
  print(p1)
  
  parent_dir_2 = '/Users/niubilitydiu/Box\ Sync/Dissertation/Dataset\ and\ Rcode/EEG_MATLAB_data/SIM_summary/'
  ggsave(paste(parent_dir_2, 'simulation_mean_function_type_1.png', sep=""),
         plot=p1, height=100, width=250, units='mm', dpi=500)
  ggsave(paste(parent_dir_2, 'simulation_mean_function_type_1.pdf', sep=""),
         plot=p1, height=100, width=250, units='mm', dpi=500)
  
  # dat_2_true_gen = data.frame(
  #   time = rep(seq(0, 30/32*1000, length.out=30), 2),
  #   value = c(mean_fn_tar_ls$tar_2, mean_fn_ntar_ls$ntar_2),
  #   category = rep(c('Target', 'Non-target'), each=30)
  # )
  # p2_true_gen = ggplot(dat_2_true_gen, aes(time, value, colour=factor(category)))
  # p2_true_gen = p2_true_gen + geom_point(size=1.5) +
  #   geom_line(size=1, aes(colour=factor(category))) +
  #   xlim(c(0, 1100)) + ylim(c(-2, 8)) +
  #   xlab('Time (ms)') + ylab("Amplitude (muV)") +
  #   # scale_x_continuous(breaks=as.integer(seq(0, 1200, length.out=13))) +
  #   theme(plot.title=element_text(hjust=0.5),
  #         panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
  #         panel.grid.major = element_blank(),
  #         panel.grid.minor = element_blank(),
  #         legend.position = c(0.8, 0.8),
  #         legend.title = element_blank(),
  #         legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
  #         plot.margin = margin(1, 1, 1, 1, "cm"))
  # print(p2_true_gen)
  #
  #
  # dat_2_len_25 = data.frame(
  #   time = rep(seq(0, 25/32*1000, length.out=25), 2),
  #   value = c(mean_fn_tar_ls$tar_4, mean_fn_ntar_ls$ntar_4),
  #   category = rep(c('Target', 'Non-target'), each=25)
  # )
  # p2_len_25 = ggplot(dat_2_len_25, aes(time, value, colour=factor(category)))
  # p2_len_25 = p2_len_25 + geom_point(size=1.5) +
  #   geom_line(size=1, aes(colour=factor(category))) +
  #   xlim(c(0, 1100)) + ylim(c(-2, 8)) +
  #   xlab('Time (ms)') + ylab("Amplitude (muV)") +
  #   # scale_x_continuous(breaks=as.integer(seq(0, 1200, length.out=13))) +
  #   theme(plot.title=element_text(hjust=0.5),
  #         panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
  #         panel.grid.major = element_blank(),
  #         panel.grid.minor = element_blank(),
  #         legend.position = c(0.8, 0.8),
  #         legend.title = element_blank(),
  #         legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
  #         plot.margin = margin(1, 1, 1, 1, "cm"))
  # print(p2_len_25)
  #
  #
  # dat_2_len_35 = data.frame(
  #   time = rep(seq(0, 35/32*1000, length.out=35), 2),
  #   value = c(mean_fn_tar_ls$tar_6, mean_fn_ntar_ls$ntar_6),
  #   category = rep(c('Target', 'Non-target'), each=35)
  # )
  # p2_len_35 = ggplot(dat_2_len_35, aes(time, value, colour=factor(category)))
  # p2_len_35 = p2_len_35 + geom_point(size=1.5) +
  #   geom_line(size=1, aes(colour=factor(category))) +
  #   xlim(c(0, 1100)) + ylim(c(-2, 8)) +
  #   xlab('Time (ms)') + ylab("Amplitude (muV)") +
  #   # scale_x_continuous(breaks=as.integer(seq(0, 1200, length.out=13))) +
  #   theme(plot.title=element_text(hjust=0.5),
  #         panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
  #         panel.grid.major = element_blank(),
  #         panel.grid.minor = element_blank(),
  #         legend.position = c(0.8, 0.8),
  #         legend.title = element_blank(),
  #         legend.background = element_rect(fill='transparent', size=0.25, color='white', linetype='solid'),
  #         plot.margin = margin(1, 1, 1, 1, "cm"))
  # print(p2_len_35)
  #
  # p2 = grid.arrange(p2_true_gen, p2_len_25, p2_len_35, ncol=3,
  #                   top='Type 2 for TrueGen, LatencyLen25, LatencyLen35')
  # print(p2)
  #
  # parent_dir_2 = '/Users/niubilitydiu/Box\ Sync/Dissertation/Dataset\ and\ Rcode/EEG_MATLAB_data/SIM_summary/'
  # ggsave(paste(parent_dir_2, 'mean_fn_type_1_true_len25_len35.png', sep=""),
  #        plot=p1, height=120, width=300, units='mm', dpi=300)
  # ggsave(paste(parent_dir_2, 'mean_fn_type_2_true_len25_len35.png', sep=""),
  #        plot=p2, height=120, width=300, units='mm', dpi=300)
}





# Multi-channel mean function true values
# Only for TrueGen scenario
mean_fn_multi_channel_1 = data.frame(
  channel = rep(1:3, each=2 * n_length_fit),
  type = rep(rep(c('Target', 'Non-target'), each=n_length_fit), 3),
  time = rep(seq(0, n_length_fit/32*1000, length.out=n_length_fit), 2*3),
  value = c(0.000000e+00,  8.229730e-01,  1.623497e+00,  2.379737e+00,  3.071064e+00,
                         3.678620e+00,  4.185832e+00,  4.578867e+00,  4.847001e+00,  4.982922e+00,
                         4.982922e+00,  4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                         3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,  6.123234e-16,
                         -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01, -2.449294e-16,
                         0, 0, 0, 0, 0,
            0.000000e+00,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                          7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                          9.965845e-01,  9.694003e-01,  9.157733e-01, 8.371665e-01,  7.357239e-01,
                          6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  1.224647e-16,
                          -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01, -4.898587e-17,
                          0, 0, 0, 0, 0,
            -2.449294e-16, -5.877853e-01, -9.510565e-01, -9.510565e-01,
                         -5.877853e-01,  6.123234e-16,  8.229730e-01,  1.623497e+00,
                         2.379737e+00,  3.071064e+00,  3.678620e+00,  4.185832e+00,
                         4.578867e+00,  4.847001e+00,  4.982922e+00,  4.982922e+00,
                         4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                         3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,
                         0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
                         0.000000e+00,  0.000000e+00,
            -4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01,
                          -1.175571e-01,  1.224647e-16,  1.645946e-01,  3.246995e-01,
                          4.759474e-01,  6.142127e-01,  7.357239e-01,  8.371665e-01,
                          9.157733e-01,  9.694003e-01,  9.965845e-01,  9.965845e-01,
                          9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                          6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,
                          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
                          0.000000e+00,  0.000000e+00,
            0, 0, 0, 0.000000e+00,  8.229730e-01,  1.623497e+00,  2.379737e+00,  3.071064e+00,
                         3.678620e+00,  4.185832e+00,  4.578867e+00,  4.847001e+00,  4.982922e+00,
                         4.982922e+00,  4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                         3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,  6.123234e-16,
                         -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01, -2.449294e-16,
                         0, 0,
            0, 0, 0, 0.000000e+00,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                          7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                          9.965845e-01,  9.694003e-01,  9.157733e-01, 8.371665e-01,  7.357239e-01,
                          6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  1.224647e-16,
                          -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01, -4.898587e-17,
                          0, 0)
)
mean_fn_multi_channel_1$type = factor(mean_fn_multi_channel_1$type, levels=c('Target', 'Non-target'))
mean_fn_multi_channel_1$channel = factor(mean_fn_multi_channel_1$channel,
                                         levels=c(1, 2, 3),
                                         labels=c('Channel 1', 'Channel 2', 'Channel 3'))
if (display_bool) {
  p_mean_fn_multi_channel_1 = ggplot(data=mean_fn_multi_channel_1) + 
    geom_line(aes(x=time, y=value, color=type), size=2) +
    geom_point(aes(x=time, y=value, color=type), size=3) +
    # geom_step(aes(x=time, y=value_binary, linetype=method), direction='h', size=1) +
    scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, by=200)) + 
    scale_y_continuous(limits=c(-2, 6), breaks=seq(-2, 6, by=2)) +
    xlab('Time (ms)') + ylab('Amplitude (muV)') + 
    facet_wrap(~channel, nrow=1) +
    theme(plot.title=element_text(hjust=0.5),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.25, 
                                         color='white', linetype='solid'),
          plot.margin=margin(.5, .5, .5, .5, 'cm'))
  
  # print(p_mean_fn_multi_channel_1)
  parent_dir = '/Users/niubilitydiu/Box\ Sync/Dissertation/Dataset\ and\ Rcode/EEG_MATLAB_data/SIM_summary/'
  ggsave(paste(parent_dir, 'simulation_multi_mean_function_type_1.png', sep=""),
         plot=p_mean_fn_multi_channel_1, height=100, width=200, units='mm', dpi=300)
}

mean_fn_multi_channel_2 = data.frame(
  channel = rep(1:3, each=2 * n_length_fit),
  type = rep(rep(c('Target', 'Non-target'), each=n_length_fit), 3),
  time = rep(seq(0, n_length_fit/32*1000, length.out=n_length_fit), 2*3),
  value = c(mean_fn_multi_channel_1$value[mean_fn_multi_channel_1$channel=='Channel 2'],
            0.6 * mean_fn_multi_channel_1$value[mean_fn_multi_channel_1$channel=='Channel 2' &
                                            mean_fn_multi_channel_1$type == 'Target'],
            mean_fn_multi_channel_1$value[mean_fn_multi_channel_1$channel=='Channel 2' &
                                            mean_fn_multi_channel_1$type == 'Non-target'],
            0.3 * mean_fn_multi_channel_1$value[mean_fn_multi_channel_1$channel=='Channel 2' &
                                                  mean_fn_multi_channel_1$type == 'Target'],
            mean_fn_multi_channel_1$value[mean_fn_multi_channel_1$channel=='Channel 2' &
                                            mean_fn_multi_channel_1$type == 'Non-target']
            )
)
mean_fn_multi_channel_2$type = factor(mean_fn_multi_channel_2$type, levels=c('Target', 'Non-target'))
mean_fn_multi_channel_2$channel = factor(mean_fn_multi_channel_2$channel,
                                         levels=c(1, 2, 3),
                                         labels=paste('Channel ', 1:3, '\n',
                                                      c('Ratio: 5:1', 'Ratio: 3:1', 'Ratio: 1.5:1'), 
                                                      sep=""))

if (display_bool) {
  p_mean_fn_multi_channel_2 = ggplot(data=mean_fn_multi_channel_2) + 
    geom_line(aes(x=time, y=value, color=type), size=2) +
    geom_point(aes(x=time, y=value, color=type), size=3) +
    # geom_step(aes(x=time, y=value_binary, linetype=method), direction='h', size=1) +
    scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, by=200)) + 
    scale_y_continuous(limits=c(-2, 6), breaks=seq(-2, 6, by=2)) +
    xlab('Time (ms)') + ylab('Amplitude (muV)') + 
    facet_wrap(~channel, nrow=1) +
    theme(plot.title=element_text(hjust=0.5),
          panel.background=element_rect(fill = "white",
                                        colour = "black",
                                        size = 0.5, linetype = "solid"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position='bottom',
          legend.title=element_blank(),
          legend.background=element_rect(fill='transparent', size=0.25, 
                                         color='white', linetype='solid'),
          plot.margin=margin(.5, .5, .5, .5, 'cm'))
  
  print(p_mean_fn_multi_channel_2)
  parent_dir = '/Users/niubilitydiu/Box\ Sync/Dissertation/Dataset\ and\ Rcode/EEG_MATLAB_data/SIM_summary/'
  ggsave(paste(parent_dir, 'simulation_multi_mean_function_type_2.png', sep=""),
         plot=p_mean_fn_multi_channel_2, height=100, width=200, units='mm', dpi=300)
}
