import os
import numpy as np
import sys

local_use = (sys.argv[1] == 'True')
# local_use = True

date_base_num = 2020010200

sim_ids = date_base_num + np.arange(30) + 1
if local_use:
    # sim_name_id = sys.argv[2]
    sim_name_id = date_base_num + 6
else:
    sim_name_id = sim_ids[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]

data_type = 'SIM_files'
letters = list('THE_QUICK_BROWN_FOX')
DAT_TYPE = 'float32'
letter_dim = len(letters)
num_repetition = 15
if sim_name_id <= date_base_num + 10:
    num_electrode = 1
elif sim_name_id <= date_base_num + 20:
    num_electrode = 2
else:
    num_electrode = 3

num_electrode_generate = 2

n_multiple = 5
flash_and_pause_length = 5
n_length = int(n_multiple * flash_and_pause_length)
num_rep = 12
total_stm_num = num_repetition * num_rep

# Bayes LDA global/hyper-parameters
u = 8
alpha_s = 5.0
beta_s = 5.0
zeta_lambda = 1e-3 * np.ones([num_electrode])
zeta_s = 1e-1 * np.ones([num_electrode])
zeta_rho = 1e-4 * np.ones([num_electrode])
ki = 0.4
scale_1 = 0.2
scale_2 = 0.15
std_bool = False
display_plot_bool = False
beta_ising = 0.1
gamma_neighbor = 2
plot_threshold = 0.5
a = 1  # weight for target
b = 5  # weight for non-target
kappa = 3200
NUM_INTERVAL = 200

# trn_repetition = 10
# n_samples, n_burn_in = 200, 2000
# num_steps_between_results, step_size_init = 1, 5e-4
# target_accept_prob, num_leapfrog_steps = 0.2, 5
#
# trn_repetition = 8
# trn_seq_len = int((trn_repetition*num_rep+n_multiple-1)*flash_and_pause_length)
# test_repetition = num_repetition - trn_repetition
#
# alpha = 10
# beta = 1
# lambda_num = 1000
