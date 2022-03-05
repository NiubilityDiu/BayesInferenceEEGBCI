import self_py_fun.GlobalEEG as gc
from self_py_fun.BayesGenFun import *
import csv

# global constants
np.random.seed(gc.seed_index)
reshape_1d_bool = False
# single_pred_bool = False
sim_dat_bool = False
eeg_dat_type = 'super_seq'
file_subscript = 'down'
q_mcmc = 2
thinning = 5
method_name = 'BayesGenq' + str(q_mcmc)
# bp_low = 0.5
bp_low = float(sys.argv[7])
# bp_upp = 7
bp_upp = float(sys.argv[4])
if float(bp_upp) == int(bp_upp):
    bp_upp = int(bp_upp)
else:
    bp_upp = float(bp_upp)

if bp_upp < 0:
    eeg_file_suffix = 'raw'
else:
    eeg_file_suffix = 'raw_bp_{}_{}'.format(bp_low, bp_upp)
# zeta_0 = 0.5
zeta_0 = float(sys.argv[5])
single_letter_dim = 1
single_rep_dim = 1
# level_num = 8
level_num = int(sys.argv[6])
# scale_val = float(sys.argv[8])
scale_val = 0.5
# gamma_val = float(sys.argv[9])
gamma_val = 1.8

BayesGenSeqObj = BayesGenSeq(
    data_type=gc.DATA_TYPE,
    sub_name_short=gc.sub_file_name[:4],
    sub_folder_name=gc.sub_file_name,
    num_repetition=gc.NUM_REPETITION,
    num_electrode=gc.NUM_ELECTRODE,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)

job_id_opt = 'binary_down_{}_{}_zeta_{}'.format(
    gc.DEC_FACTOR, eeg_file_suffix, zeta_0
)
b_distance = np.zeros([gc.NUM_ELECTRODE])
s_x_sq_median = np.zeros([gc.NUM_ELECTRODE])
z_convol = []
# rho_post = []
rho_set, _, _, _, _ = BayesGenSeqObj.produce_pre_compute_rhos(q_mcmc, 50, level_num)
d_mat = BayesGenSeqObj.create_design_mat_gen_bayes_seq(1)

gc_file_subscript = '{}_{}_from_{}'.format(
    gc.file_subscript, gc.DEC_FACTOR, eeg_file_suffix
)
# Import the training set without subsetting yet
[signal_x, code_3d, type_3d] = BayesGenSeqObj.import_eeg_processed_dat(
    gc_file_subscript, reshape_1d_bool
)


signal_x = np.transpose(signal_x, axes=(1, 0, 2, 3))
[signal_x_odd, eeg_type_odd, eeg_code_odd, signal_x_even, eeg_type_even, eeg_code_even,
 single_seq_len] = BayesGenSeqObj.split_convol_super_seq_by_seq_ids(
    signal_x, type_3d, code_3d, gc.rep_odd_id, gc.rep_even_id,
    odd_reshape=1, even_reshape=1
)

# signal_x_rank = np.concatenate([signal_x_odd, signal_x_even], axis=1)
# eeg_type_rank = np.concatenate([eeg_type_odd, eeg_type_even], axis=0)
# seq_num_rank = (len(gc.rep_odd_id) + len(gc.rep_even_id)) * gc.LETTER_DIM
signal_x_rank = np.copy(signal_x_odd)
eeg_type_rank = np.copy(eeg_type_odd)
seq_num_rank = len(gc.rep_odd_id) * gc.LETTER_DIM
# mcmc_num = 0
# _, pres_chky_t_set, _ = BayesGenSeqObj.generate_std_ar2_pres_candidate(
#     rho_set, single_seq_len
# )

channel_name = 'all_channels'
# print(channel_name)
[_, arg_rho_mcmc, _,
 beta_tar_mcmc, beta_ntar_mcmc,
 _, _, _] = BayesGenSeqObj.import_mcmc(
    eeg_dat_type, method_name, single_rep_dim, channel_name, job_id_opt, sim_dat_bool,
    scale=scale_val, gamma=gamma_val
)
mcmc_num = beta_tar_mcmc.shape[0]

# thinning process
thin_idx = np.arange(0, mcmc_num, thinning)
beta_tar_mcmc, beta_ntar_mcmc = beta_tar_mcmc[thin_idx, ...], beta_ntar_mcmc[thin_idx, ...]

for e in range(gc.NUM_ELECTRODE):
    b_distance[e] = BayesGenSeqObj.bhattacharyya_gaussian_distance(
        beta_tar_mcmc[:, e, :, 0], beta_ntar_mcmc[:, e, :, 0]
    )

    beta_mcmc_e = np.concatenate(
        [beta_tar_mcmc[:, e, np.newaxis, ...],
         beta_ntar_mcmc[:, e, np.newaxis, ...]], axis=2)  # (mcmc_num, 1, 100, 1)
    beta_mcmc_e = np.transpose(beta_mcmc_e, axes=(1, 0, 2, 3))  # (1, mcmc_num, 100, 1)
    for seq_id in range(seq_num_rank):
        t_mat_id = BayesGenSeqObj.create_transform_mat(
            eeg_type_rank[seq_id*gc.NUM_REP:(seq_id+1)*gc.NUM_REP], 1, 1, 'super_seq'
        )
        dt_mat_id = np.matmul(d_mat, t_mat_id)[np.newaxis, ...]  # (1, 1, 160, 100)
        z_e_odd_id = np.matmul(dt_mat_id, beta_mcmc_e)
        z_convol.append(z_e_odd_id)


# print('b_distance = {}'.format(np.round(b_distance, decimals=2)))
# print(np.argsort(b_distance)[::-1] + 1)
# print('s_x_sq_median = {}'.format(np.round(s_x_sq_median, decimals=2)))
# print(np.argsort(s_x_sq_median) + 1)

# Try Jian's idea of computing SNR to
# incorporate both ERPs and external noise activity
# For real data training set, we split the dataset by odd/even number and test on the entire training set
# No need to worry about the scaling factor right now, make sure the method works
# signal_x_rank = np.concatenate(signal_x_rank, axis=0)
# print('signal_x_rank has shape {}'.format(signal_x_rank.shape))
z_convol = np.reshape(
    np.stack(z_convol, axis=0),
    [gc.NUM_ELECTRODE, seq_num_rank, len(thin_idx), single_seq_len, 1]
)
# print('z_convol has shape {}'.format(z_convol.shape))
signal_var = np.mean(np.var(z_convol, axis=(-3, -2, -1)), axis=-1)
observe_var = np.mean(np.var(signal_x_rank, axis=(-2, -1)), axis=-1)
# print('signal_var = {}'.format(np.round(signal_var, decimals=2)))
# print('observe_var = {}'.format(np.round(observe_var, decimals=2)))
snr_modify = signal_var / observe_var * 100
print('Modified SNR % = {}'.format(np.round(snr_modify, decimals=2)))
arg_snr_modify = np.argsort(snr_modify)[::-1] + 1
print('Channel Rank by SNR = {}'.format(arg_snr_modify))

# Save the results to csv file
if gc.local_use:
    parent_path_dir = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation'
    channel_select_dir = '{}/manuscript/Chapter_1/First-round review 2021-08-06'.format(parent_path_dir)
    channel_select_dir = '{}/channel_selection_2.csv'.format(channel_select_dir)
else:
    channel_select_dir = '/home/mtianwen/EEG_MATLAB_data/BayesGenq2Pred/Chapter_1/channel_selection_2.csv'

with open(channel_select_dir, 'a') as channel_csv:
    channel_reader = csv.writer(channel_csv, delimiter=',',
                                # quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
    K_sub_info = [scale_val, gamma_val, 'K{}'.format(gc.K_num)]
    K_sub_info.extend(arg_snr_modify.tolist())
    K_sub_info.append('\n')
    print(K_sub_info)
    channel_reader.writerow(K_sub_info)

