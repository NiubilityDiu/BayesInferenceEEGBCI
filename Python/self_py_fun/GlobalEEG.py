import sys, os
from self_py_fun.Miscellaneous import *

# Passing variable from outside input
# local_use = True
local_use = (sys.argv[1] == 'T')
# K_num = 114
K_num = int(sys.argv[2])
K_num_ids_2 = np.array([114, 117, 121, 146, 151, 158, 171, 172, 177, 183])
# K_num = K_num_ids_2[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]
# 41 subjects with K protocols

# DEC_FACTOR = 8
DEC_FACTOR = int(sys.argv[3])
FLASH_AND_PAUSE_LENGTH = int(40 / DEC_FACTOR)
N_MULTIPLE = 5
N_LENGTH = N_MULTIPLE * FLASH_AND_PAUSE_LENGTH
NUM_ELECTRODE = 16

file_subscript = 'down'
SERIAL_NUM = '001'
EXP_NAME = 'BCI'  # BCI, CMP...
DATA_TYPE = 'TRN'  # Choose among TRN_files, FRT_files, NOC_files, and SLO_files
seed_index = 1001

K_num_ids = np.array([106, 107, 108,
                      111, 112, 113, 114, 115, 117, 118, 119, 120,
                      121, 122, 123,
                      143, 145, 146, 147,
                      151, 152, 154, 155, 156, 158, 159, 160,
                      166, 167, 171, 172, 177, 178, 179,
                      183, 184, 185, 190,
                      191, 212, 223])

channel_ids_ls = {
    # 'K114': np.array([6, 7, 8, 10, 12]),  # single
    'K114': np.array([15, 6, 7, 12, 1]),  # 16-channel joint
    'K117': np.array([14, 16, 15,  8, 10]),  # 16-channel joint
    'K121': np.array([15, 14, 8, 7, 6]),  # 16-channel joint
    'K146': np.array([15, 14, 1, 5, 7]),  # 16-channel joint
    'K151': np.array([15, 6, 1, 12, 7]),  # 16-channel joint
    'K158': np.array([16, 3, 2, 1, 6]),  # 16-channel joint
    'K171': np.array([16, 6, 7, 2, 15]),  # 16-channel joint
    'K172': np.array([16, 1, 2, 12, 6]),  # 16-channel joint
    'K177': np.array([15, 10, 7, 8, 12]),  # 16-channel joint
    'K183': np.array([6, 2, 3, 1, 7])  # 16-channel joint
}

channel_ids_swlda_ls = {
    'K114': np.array([15, 14, 13, 4, 16]),
    'K117': np.array([14, 16, 15, 12, 11])
}

if local_use:
    # channel_id = 5
    # channel_id = np.array([6, 7, 8, 10]) - 1
    channel_id = np.arange(NUM_ELECTRODE)
    # channel_id = np.array([15, 14, 16, 6, 13]) - 1
else:
    # channel_id = 5
    channel_id = np.array([15, 14, 16, 6, 13]) - 1
    # channel_id = np.arange(NUM_ELECTRODE)
    # channel_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    # channel_ids = return_cumulative_array_in_list(channel_ids_ls['K' + str(K_num)] - 1)
    # channel_id = channel_ids[int(os.environ.get('SLURM_ARRAY_TASK_ID'))]

if 'SLO' in DATA_TYPE or 'REG' in DATA_TYPE:
    H_num = 101
    trn_id = 1
    sub_file_name = 'H' + str(H_num) + '_025_RC_' + DATA_TYPE + '_TRN_0' + str(trn_id)
else:
    sub_file_name = 'K' + str(K_num) + '_' + SERIAL_NUM + '_' + EXP_NAME + '_' + DATA_TYPE

# K154 and K190 have 20 num_repetitions with larger total_seq_length,
# and the mean functions are weird, too.
if '154' in sub_file_name or '190' in sub_file_name:
    NUM_REPETITION = 20
else:
    NUM_REPETITION = 15
NUM_REPETITION_TEST = 5

# This setup is only for first stage of real data fitting examination.
rep_odd_id = np.arange(3, NUM_REPETITION + 1, 2)
rep_even_id = np.arange(2, NUM_REPETITION + 1, 2)

# Global Parameter of the Generation Setting
TARGET_LETTERS = list('THE_QUICK_BROWN_FOX')  # TRN set reads a logical/reasonable sentence.
ROW_INDICES = [4, 2, 1, 6, 3, 4, 2, 1, 2, 6, 1, 3, 3, 4, 3, 6, 1, 3, 4]
COL_INDICES = [8, 8, 11, 12, 11, 9, 9, 9, 11, 12, 8, 12, 9, 11, 8, 12, 12, 9, 12]
LETTER_DIM = len(TARGET_LETTERS)

DATA_TYPE = DATA_TYPE + '_files'  # order fixed!
SAMPLING_RATE = 256
NUM_REP = 12
P300_FLASH_STRENGTH = 1
P300_PAUSE_STRENGTH = 0
NON_P300_STRENGTH = 0
TOTAL_STM_TRN_NUM = LETTER_DIM * NUM_REPETITION * NUM_REP
SEQ_LENGTH = (N_MULTIPLE - 1 + NUM_REP) * FLASH_AND_PAUSE_LENGTH

# display_plot_bool = False
# Bayesian MCMC setting
KAPPA = 2
BURN_IN = 1
NUM_INTERVAL = 100

# Job Array ID tutorial:
# https://rc.byu.edu/wiki/index.php?page=How+do+I+submit+a+large+number+of+very+similar+jobs%3F
if __name__ == '__main__':
    # print('trn_repetition={}\n'.format(trn_repetition))
    print('true_row_index={}\n'.format(ROW_INDICES))
    print('true_col_index={}\n'.format(COL_INDICES))
    print('target_letter={}\n'.format(TARGET_LETTERS))

