parent_path = '/Users/niubilitydiu/Box\ Sync/Dissertation/Dataset\ and\ Rcode/EEG_MATLAB_data/EEGBayesLDA'
setwd(parent_path)
library(ggplot2)
# library(hrbrthemes)
# import data file
# For simlation studies, we need to specify additional setups
# For real data, we only need trn_repetition number as input argument
target_letters = c('T', 'H', 'E', '_', 
                   'Q', 'U', 'I', 'C', 'K', '_', 
                   'B', 'R', 'O', 'W', 'N', '_', 
                   'F', 'O', 'X')
num_letter = 19
num_repetition = 15
real_data = T

K_num_ids = c(106:108, 111:123, 143, 145:147,
              151, 152, 154, 155, 156, 158, 159, 160,
              166, 167, 171, 172, 177, 178, 179, 
              183, 184, 185, 190, 191, 212, 223)

K_ids_num = length(K_num_ids)

if (real_data) {
  K_num = 223
  serial_num = '001'
  trn_id = 15
  exp_type = 'BCI'
  data_type = 'TRN'
  method_name = 'lda_bayes_pred_select'
  file_path = paste('./K_protocol/K', K_num, '_', serial_num, '_', exp_type, 
                    '_', data_type, '_', method_name, '_trn_', trn_id, '.csv', sep="")
} else {
  mean_ratio = 5
  rho = 0.8
  channel_dim = 3
  sim_id = 2020010230
  method_name = 'lda_bayes_pred_select'  # or 'convol_lda_bayes_pred_select'
  trn_id = 15
  file_path = paste('./mean_ratio_', mean_ratio, '_1/rho_', rho, '_', channel_dim, 'd/sim_',
                    sim_id, '_', method_name, '_trn_', trn_id, '.csv', sep="")
}
print(file_path)
file_path_char = strsplit(file_path, "")
file_path_char_length = length(file_path_char[[1]])
pdf_file_path = paste(file_path_char[[1]][1:(file_path_char_length-4)], collapse="")
pdf_file_path = paste(pdf_file_path, '_plots.pdf', sep="")
print(pdf_file_path)

pred_mat = read.csv(file_path, header=F)
# Record the plots
pdf(pdf_file_path)

# Task 1, compute the freq prediction accuracy
# Extract max prob letter
pred_table = NULL
pred_table_binary = NULL

for (i in 1:num_letter) {
  pred_table = rbind(pred_table, pred_mat[(i-1)*4+6, 2:16])
}
for (i in 1:num_repetition) {
  pred_table_binary = cbind(pred_table_binary, (pred_table[,i] == target_letters)*1)
}
freq_pred_accuracy = apply(pred_table_binary, 2, mean)

if (real_data) {
  plot(1:num_repetition, freq_pred_accuracy, 
       xlab="sequences", ylab="fred pred accuracy", main=paste("K", K_num, sep=""), type='b', ylim=c(0, 1))
} else {
  plot(1:num_repetition, freq_pred_accuracy, 
       xlab="sequences", ylab="fred pred accuracy", main=paste("sim_id_", sim_id, sep=""), type='b', ylim=c(0, 1))
}
print(freq_pred_accuracy)

# Task 2, produce the heat map of the uncertainty measure over sequences.
pred_prob_table = NULL

for (i in 1:num_letter) {
  pred_prob_table = rbind(pred_prob_table, pred_mat[(i-1)*4+4, 2:16])
}

seqs = 1:num_repetition
# y_letter = NULL
# for(i in 1:num_letter) {
#   y_letter = c(y_letter, paste(i, target_letters[i], sep=":"))
# }
# y_letter
y_letter = 1:num_letter
pred_prob_3d = expand.grid(X=seqs, Y=y_letter)
pred_prob_3d$certainty = as.numeric(t(as.matrix(pred_prob_table)))

# Heatmap
if (real_data) {
  main_title = paste('K', K_num, '_train_seq_', trn_id, sep="")
} else {
  main_title = paste('sim_id_', sim_id, '_train_seq_', trn_id, sep="")
}

ggplot(pred_prob_3d, aes(X, Y, fill=certainty)) + geom_tile() +
  # scale_fill_gradient(low='white', high='blue') + 
  scale_fill_distiller(palette = "Blues", direction=1) +
  scale_x_discrete(name='sequences', limits=1:num_repetition, expand=c(0,0)) +
  scale_y_reverse(name='target letters', breaks=1:19, labels=target_letters, expand=c(0,0)) +
  ggtitle(main_title) + 
  theme(plot.title = element_text(hjust=0.5))

dev.off()