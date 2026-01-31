import os
import numpy as np
import sys


local_use = (sys.argv[1].upper() == 'T')
# local_use = True
sim_type = sys.argv[2]
# sim_type = 'latency'
# sim_type = 'seq'
# sim_type = 'super_seq'
# sim_type = 'super_seq_trun_seq'
# sim_type = 'ML'
# repet_num_fit = 5
repet_num_fit = int(sys.argv[3])
data_type = 'SIM_files'

scenario_names = ['TrueGen', 'MisNoiseDist', 'MisLatencyLen25',
                  'MisLatencyLen35', 'MisSignalDist', 'MisTypeDist']
# scenario_name = scenario_names[1 - 1]
scenario_name = scenario_names[int(sys.argv[4]) - 1]
# print('scenario_name = {}'.format(scenario_name))
# Jian suggested that I use different N_MULTIPLE to generate data,
# but use the same N_MULTIPLE_FIT to fit the data with our method.

if '25' in scenario_name:
    N_MULTIPLE = 5
elif '35' in scenario_name:
    N_MULTIPLE = 7
else:
    N_MULTIPLE = 6

if 'Noise' in scenario_name:
    normal_bool = False
else:
    normal_bool = True
t_df = 5
NUM_ELECTRODE = 3

if local_use:
    seed_index = 1027
    design_num = 0  # start from 0
    subset_num = 0  # start from 0

    if NUM_ELECTRODE == 1:
        # mean_fn_type = 1
        # mean_fn_type = int(sys.argv[5])
        s_x_sq = 10
        rho = np.array([[0.5, 0]])
        # 0.5, 0 or 0.6, 0.2 for single channel
        s_x_sq = s_x_sq * np.ones([NUM_ELECTRODE])
        # rho always has the shape (num_electrode, q)
        q = rho.shape[1]
        rho = np.tile(rho, [NUM_ELECTRODE, 1])
    else:
        # mean_fn_type = 'multi_channel_2'
        mean_fn_type = sys.argv[5]
        s_x_sq = 20
        rho = np.array([0.5, 0])
        rho_s = 0.5  # fix

else:
    seed_index = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    design_num = int(seed_index / 100)
    subset_num = seed_index % 100

    # Single-channel simulation setup
    if NUM_ELECTRODE == 1:
        mean_fn_types = [1, 2]
        s_x_sqs = [10, 20]
        rho_ids = [1, 2]
        t_df = 5

        para_list = [mean_fn_types, s_x_sqs, rho_ids]
        level_num = 8
        para_list = np.meshgrid(*para_list)
        para_list_long = []
        for i in range(3):
            para_list_long.append(np.reshape(para_list[i], level_num))
        para_list_long = np.stack(para_list_long, axis=-1)
        print(para_list_long)

        _, s_x_sq, rho_id = para_list_long[design_num, :]
        mean_fn_type = int(sys.argv[5])

        if rho_id == 1:
            rho = np.array([[0.5, 0]])
        else:
            rho = np.array([[0.6, 0.2]])

        s_x_sq = s_x_sq * np.ones([NUM_ELECTRODE])
        # rho always has the shape (num_electrode, q)
        q = rho.shape[1]
        rho = np.tile(rho, [NUM_ELECTRODE, 1])

    else:
        # mean_fn_type = 'multi_channel'
        # multi_channel or multi_channel_2
        mean_fn_type = sys.argv[5]
        s_x_sqs = [20, 40]
        rho = np.array([0.5, 0])
        rho_s = 0.5
        t_df = 5
        if mean_fn_type == 'multi_channel':
            s_x_sq = s_x_sqs[design_num-10]
        else:
            s_x_sq = s_x_sqs[design_num-20]

# Change here!
sim_common = 'sim_' + str(design_num + 1) + '_dataset_' + str(subset_num + 1)
reshape_3d_bool = True
delta_comb_bool = True
std_bool = False
display_plot_bool = False
soft_bool = True
continuity_order = 2
window_length = 5

DEC_FACTOR = 1
LETTERS = []
LETTERS.extend('The_quick_brown_fox')
LETTER_DIM = len(LETTERS)
DAT_TYPE = 'float32'
NUM_REPETITION = 10  # Here it determines the training and testing set sizes.
REPETITION_TRN = 5
REPETITION_TEST = NUM_REPETITION - REPETITION_TRN
CHANNEL_IDS = np.arange(NUM_ELECTRODE)
FLASH_PAUSE_LENGTH = 5
N_LENGTH = int(N_MULTIPLE * FLASH_PAUSE_LENGTH)
NUM_REP = 12
TOTAL_STM_TRN_NUM = LETTER_DIM * REPETITION_TRN * NUM_REP
TOTAL_STM_TEST_NUM = LETTER_DIM * REPETITION_TEST * NUM_REP
TOTAL_STM_NUM = TOTAL_STM_TRN_NUM + TOTAL_STM_TEST_NUM

SEQ_LENGTH = (NUM_REP + N_MULTIPLE - 1) * FLASH_PAUSE_LENGTH
SUPER_SEQ_LENGTH_TRN = NUM_REP * REPETITION_TRN * FLASH_PAUSE_LENGTH + (N_MULTIPLE - 1) * FLASH_PAUSE_LENGTH
SUPER_SEQ_LENGTH_TEST = NUM_REP * REPETITION_TEST * FLASH_PAUSE_LENGTH + (N_MULTIPLE - 1) * FLASH_PAUSE_LENGTH

# This is only for testing only where we have 19 letters with 5 sequence replications.
# LETTERS_2 = list('THE_QUICK_BROWN_FOX')
ROW_INDICES = [4, 2, 1, 6, 3, 4, 2, 1, 2, 6, 1, 3, 3, 4, 3, 6, 1, 3, 4]
COL_INDICES = [8, 8, 11, 12, 11, 9, 9, 9, 11, 12, 8, 12, 9, 11, 8, 12, 12, 9, 12]

# We use consistent length for model fitting
# Test whether latency length has an influence on model fitting and prediction accuracy
N_MULTIPLE_FIT = 6
N_LENGTH_FIT = int(N_MULTIPLE_FIT * FLASH_PAUSE_LENGTH)
SEQ_LENGTH_FIT = (NUM_REP + N_MULTIPLE - 1) * FLASH_PAUSE_LENGTH
SUPER_SEQ_LENGTH_TRN_FIT = NUM_REP * REPETITION_TRN * FLASH_PAUSE_LENGTH + (N_MULTIPLE_FIT - 1) * FLASH_PAUSE_LENGTH
SUPER_SEQ_LENGTH_TEST_FIT = NUM_REP * REPETITION_TEST * FLASH_PAUSE_LENGTH + (N_MULTIPLE_FIT - 1) * FLASH_PAUSE_LENGTH

KAPPA = 2001
BURN_IN = KAPPA - 1001
NUM_INTERVAL = 100

# True parameter setup
eta = 0 * np.ones([NUM_ELECTRODE])
s_x_sq_alpha = 1.0
s_x_sq_beta = 1.0
lambda_stepsize = 1e-2 * np.ones([NUM_ELECTRODE])
s_stepsize = 1e-2 * np.ones([NUM_ELECTRODE])
