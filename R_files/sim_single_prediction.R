rm(list=ls(all.names = T))
args <- commandArgs(trailingOnly = TRUE)
# local_use = T
local_use = (args[1] == 'T')

library(ggplot2)
require(gridExtra)

if (local_use) {
  parent_path = ''
  r_fun_path = ''
  data_type = 'test'  # train or test
  design_num = 2
  num_rep_fit = 5
  num_rep_pred = 5
  dataset_num = 100
  scenario_name = 'TrueGen'
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

# Show the combination of parameter setup
mean_fn_type = c(1, 2)
s_x_sq = c(10, 20)
rho = c(1, 2)
t_df = 5
para_list_mat = data.frame(
  mean_fn_type = rep(rep(mean_fn_type, each=2), 2),
  s_x_sq = rep(s_x_sq, each=4),
  rho = rep(rho, 4)
)

bayes_sseq = bayes_freq_table_design_num(
  parent_path, 'BayesGenq2Pred', design_num, dataset_num, 
  data_type, num_rep_fit, num_rep_pred, 
  scenario_name, zeta, num_letter, 'simulation_single'
)

zeta_swlda = 0.7
swlda = exist_ml_freq_table_design_num(
  parent_path, 'EEGswLDA', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, zeta_swlda, num_letter
)

ada = exist_ml_freq_table_design_num(
  parent_path, 'EEGADA', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, 0, num_letter
)

bag = exist_ml_freq_table_design_num(
  parent_path, 'EEGBAG', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, 0, num_letter
)

logistic_reg = exist_ml_freq_table_design_num(
  parent_path, 'EEGLR', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, 0, num_letter
)

random_forest = exist_ml_freq_table_design_num(
  parent_path, 'EEGRF', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, 0, num_letter
)

svm = exist_ml_freq_table_design_num(
  parent_path, 'EEGSVC', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, 0, num_letter
)

nn_ord = exist_ml_freq_table_design_num(
  parent_path, 'NNOrd', design_num, dataset_num, data_type,
  num_rep_fit, num_rep_pred, scenario_name, 0, num_letter
)

method_names = c('Bayes', 'swLDA', 'ADABoosting',
                 'Bagging', 'Logistic', 'Ramdom Forest',
                 'Support Vector Machine', 'Neural Network')
method_table = data.frame(
  method = rep(method_names, each=num_rep_pred),
  rep_id = rep(1:num_rep_pred, length(method_names)),
  mean = c(bayes_sseq$mean, swlda$mean, ada$mean,
           bag$mean, logistic_reg$mean, random_forest$mean,
           svm$mean, nn_ord$mean),
  sd = c(bayes_sseq$sd, swlda$sd, ada$sd,
         bag$sd, logistic_reg$sd, random_forest$sd,
         svm$sd, nn_ord$sd)
)
print(method_table)

p1 = ggplot(method_table, aes(rep_id, mean))
p1 = p1 + geom_point(size=1.5) +
  geom_line(size=0.5) +
  geom_errorbar(aes(ymin=pmax(0, mean-sd), ymax=pmin(1, mean+sd)),
                width=0.1, size=0.5) +
  ylim(c(-0.05, 1.05)) +
  xlab('Sequence Replications') + ylab("Prediction Accuracy") +
  scale_x_continuous(breaks=as.integer(seq(1, num_rep_pred, length.out=min(num_rep_pred, 5)))) +
  facet_wrap(~method, ncol=4) +
  theme(plot.title=element_text(hjust=0.5),
        panel.background = element_rect(fill='white', color='black', size=0.5, linetype='solid'),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        plot.margin = margin(1, 1, 1, 1, "cm"))
print(p1)

method_summary = list(
  bayes=bayes_sseq,
  swlda=swlda,
  ada=ada,
  bag=bag,
  logistic_regression=logistic_reg,
  random_forest=random_forest,
  support_vector_machine=svm,
  neural_network=nn_ord,
  method_table=method_table
)
output_file_dir = paste(parent_path, '/EEG_MATLAB_data/SIM_summary/sim_',
                        design_num, '_train_', num_rep_fit, '_', data_type, '_',
                        num_rep_pred,  '_summary_', scenario_name, '_zeta_', zeta, sep="")
saveRDS(method_summary, file=paste(output_file_dir, '.RDS', sep=""))
ggsave(paste(output_file_dir, '.png', sep=""), plot=p1, height=200, width=400, units='mm', dpi=300)
