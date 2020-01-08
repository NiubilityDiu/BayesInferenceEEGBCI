import os
import numpy as np
import sys
# Define the global constants:

local_use = (sys.argv[1] == 'True')
# local_use = True
K_num_ids = np.array([106, 107, 108,
                      111, 112, 113, 114, 115, 117, 118, 119, 120,
                      121, 122, 123,
                      143, 145, 146, 147,
                      151, 152, 154, 155, 156, 158, 159, 160,
                      166, 167, 171, 172, 177, 178, 179,
                      183, 184, 185, 190,
                      191,
                      212,
                      223])

# 41 subjects with K protocols
if local_use:
    K_num = 108
else:
    K_num = K_num_ids[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
    # K_num = 106

letter_dim = 19

serial_num = '001'
exper_name = 'BCI'  # BCI, CMP...
data_type = 'TRN'  # Choose among TRN_files, FRT_files and NOC_files
n_multiple = 4
sub_file_name = 'K' + str(K_num) + '_' + \
                serial_num + '_' + \
                exper_name + '_' + data_type
data_type = data_type + '_files'
method_name = 'qda' + '_python'  # Choose among qda, convol, or lda/swlda (to be continued)
file_subscript = 'down'
# Choose among down or raw_trun
# When do EEG_pre.py from scratch, set it to "raw_trun" first.

sampling_rate = 256
num_letter = 19
# K154 and K190 have 20 num_repetitions with larger total_seq_length,
# and the mean functions are weird, too.
if sub_file_name == 'K154_001_BCI_TRN' or sub_file_name == 'K190_001_BCI_TRN':
    num_repetition = 20
else:
    num_repetition = 15

# test_repetition = num_repetition - trn_repetition
num_rep = 12
num_electrode = 16
p300_flash_strength = 1
p300_pause_strength = 0
non_p300_strength = 0

if file_subscript is not 'down':
    dec_factor = 1
else:
    dec_factor = 8
flash_and_pause_length = int(40 / dec_factor)
n_length = n_multiple * flash_and_pause_length

# For Bayesian method:
# Hyper-params:
# message = 'using the first {} letters and the first {} repetitions'.format(letter_dim, trn_repetition)
# kernel_scale_1 = 0.1*np.ones([num_electrode])
# kernel_scale_0 = 0.2*np.ones([num_electrode])
# eps_kernel_scale = 0.5*np.ones([num_electrode])

# Gibbs sampling:
u = 8
alpha_s = 5.0
beta_s = 5.0
zeta_lambda = 1e-4 * np.ones([num_electrode])
zeta_s = 5e-3 * np.ones([num_electrode])
zeta_rho = 1e-4 * np.ones([num_electrode])
ki = 0.4
scale_1 = 0.2
scale_2 = 0.2
std_bool = False
beta_ising = 0.1
gamma_neighbor = 2
plot_threshold = 0.5
a = 1  # weight for target
b = 5  # weight for non-target
kappa = 3200
NUM_INTERVAL = 200

# HMC Inference
# step_size_init = 1e-4
# num_steps_between_results = 1
# num_leapfrog_steps = 5
# n_samples = 100
# n_burn_in = 2000
# target_accept_prob = 0.5

if method_name is 'convol_python':
    var_key = 'mcmc'
else:
    var_key = 'sample'

# total_seq_length = int((num_rep * num_repetition + 4) * flash_and_pause_length)
# trn_total_seq_length = int((num_rep * trn_repetition + 4) * flash_and_pause_length)
# lambda_p = 10
# # WLS use
# lambda_s = 100
# lambda_0 = 100
# iter_num = 31
# jitter = 10
# PLOT_INTERVAL = 5
# DAT_TYPE = 'float32'
# electrode_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int32) - 1
# electrode_dim = len(electrode_ids)

true_row_indices = [4, 2, 1, 6, 3, 4, 2, 1, 2, 6, 1, 3, 3, 4, 3, 6, 1, 3, 4]
true_col_indices = [8, 8, 11, 12, 11, 9, 9, 9, 11, 12, 8, 12, 9, 11, 8, 12, 12, 9, 12]
target_letters = 'THE_QUICK_BROWN_FOX'  # TRN set reads a logical/reasonable sentence.

# Job Array ID tutorial:
# https://rc.byu.edu/wiki/index.php?page=How+do+I+submit+a+large+number+of+very+similar+jobs%3F
if __name__ == '__main__':
    # print('trn_repetition={}\n'.format(trn_repetition))
    print('true_row_index={}\n'.format(true_row_indices))
    print('true_col_index={}\n'.format(true_col_indices))
    print('target_letter={}\n'.format(target_letters))

