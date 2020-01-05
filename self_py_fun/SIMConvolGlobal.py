import os
import numpy as np
import sys

local_use = (sys.argv[1] == 'True')
# local_use = True

date_base_num = 2020010200

sim_ids = date_base_num + np.arange(10) + 21
if local_use:
    # sim_name_id = sys.argv[2]
    sim_name_id = date_base_num + 18
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

num_electrode_generate = 3

n_multiple = 5
flash_and_pause_length = 5
n_length = int(n_multiple * flash_and_pause_length)
num_rep = 12
total_stm_num = num_repetition * num_rep
kappa = 5200
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
