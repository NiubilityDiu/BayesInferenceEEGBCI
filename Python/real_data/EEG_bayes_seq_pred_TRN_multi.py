import self_py_fun.GlobalEEG as gc
from self_py_fun.BayesGenFun import *

# global constants
np.random.seed(gc.seed_index)
reshape_1d_bool = False
# single_pred_bool = False
sim_dat_bool = False
# eeg_dat_type = 'single_seq'
file_subscript = 'down'
q_mcmc = 2
thinning = 10
method_name = 'BayesGenq' + str(q_mcmc)
# bp_low = 0.4
bp_low = float(sys.argv[7])
# bp_upp = 6
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
num_electrode = len(gc.channel_id)
letter_dim = gc.LETTER_DIM
# letter_dim = 3
target_letters = gc.TARGET_LETTERS[:letter_dim]
target_letter_rows = gc.ROW_INDICES[:letter_dim]
target_letter_cols = gc.COL_INDICES[:letter_dim]
repet_num_fit = len(gc.rep_odd_id)
repet_num_pred = len(gc.rep_even_id)
rho_level_num = int(sys.argv[6])
# rho_level_num = 8
n = 50
scale_val = float(sys.argv[8])
gamma_val = float(sys.argv[9])


BayesGenSeqObj = BayesGenSeq(
    data_type=gc.DATA_TYPE,
    sub_name_short=gc.sub_file_name[:4],
    sub_folder_name=gc.sub_file_name,
    num_repetition=gc.NUM_REPETITION,
    num_electrode=num_electrode,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)


print('We are using channel {} for prediction.'.format(gc.channel_id + 1))
channel_id_str = [str(e+1) for e in gc.channel_id]
if len(gc.channel_id) == 16:
    channel_name = 'all_channels'
else:
    channel_name = 'channel_' + '_'.join(channel_id_str)

gc_file_subscript = '{}_{}_from_{}'.format(
    gc.file_subscript, gc.DEC_FACTOR, eeg_file_suffix
)
job_id_opt = 'binary_down_{}_{}_zeta_{}'.format(
    gc.DEC_FACTOR, eeg_file_suffix, zeta_0
)

# Import the training set without subsetting yet
[signal_x, code_3d, type_3d] = BayesGenSeqObj.import_eeg_processed_dat(
    gc_file_subscript, reshape_1d_bool
)
signal_x_sub = np.transpose(signal_x[:, gc.channel_id, ...], [1, 0, 2, 3])
# For real data training set, we split the dataset by odd/even number and test on the entire training set
# No need to worry about the scaling factor right now, make sure the method works
[signal_x_odd, eeg_type_odd_3d, eeg_code_odd_3d,
 signal_x_even, eeg_type_even_3d, eeg_code_even_3d,
 single_seq_len] = BayesGenSeqObj.split_convol_super_seq_by_seq_ids(
    signal_x_sub, type_3d, code_3d, gc.rep_odd_id, gc.rep_even_id,
    odd_reshape=3, even_reshape=3
)

# Hyper-parameters
rho_set, rho_level, _, _, _ = BayesGenSeqObj.produce_pre_compute_rhos(
    q_mcmc, n, rho_level_num
)
cov_t_set = [BayesGenSeqObj.create_ar2_pres_mat(1, rho_set[i], single_seq_len)[0]
             for i in range(rho_level)]
d_mat_pred = BayesGenSeqObj.create_design_mat_gen_bayes_seq(single_rep_dim)

# Import MCMC samples
[cov_s_mcmc, arg_rho_mcmc, zeta_mcmc,
 beta_tar_mcmc, beta_ntar_mcmc,
 scale_opt, var_opt, _] = BayesGenSeqObj.import_mcmc(
    'super_seq', method_name, single_rep_dim, channel_name,
    job_id_opt, sim_dat_bool, scale=scale_val, gamma=gamma_val
)

# zeta_median = np.median(zeta_mcmc, axis=0)
# print('zeta_median = {}'.format(zeta_median))

arg_rho_mcmc = arg_rho_mcmc[0, :]
beta_comb_mcmc = np.concatenate([beta_tar_mcmc, beta_ntar_mcmc], axis=-2)
mcmc_num = beta_comb_mcmc.shape[0]


log_prior_prob = np.log(1 / 36 * np.ones([36]))
prob_dist_total = []
letter_dist_total = []
target_letter_rank = []


# Prediction section
print('Classification for odd sequence (training sequence).')
pred_odd_bool = True

BayesGenSeqObj.standard_compute_pred_single_seq_multi(
    beta_comb_mcmc, cov_s_mcmc, arg_rho_mcmc, cov_t_set,
    eeg_code_odd_3d, signal_x_odd, d_mat_pred,
    letter_dim, repet_num_fit, repet_num_fit, mcmc_num, thinning,
    'single_seq', method_name, channel_name,
    job_id_opt, pred_odd_bool, sim_dat_bool,
    scale=scale_val, gamma=gamma_val
)

# bayes_odd_no_first_dict = BayesGenSeqObj.standard_pred_cum_seq(
#     scale_opt, var_opt,
#     letter_dim, repet_num_fit, repet_num_fit, mcmc_num, thinning,
#     'single_seq', target_letters, target_letter_rows, target_letter_cols,
#     method_name, channel_name, job_id_opt, 'single_seq_log_lkd_multi',
#     True, pred_odd_bool, sim_dat_bool
# )
bayes_odd_dict = BayesGenSeqObj.standard_pred_cum_seq(
    scale_opt, var_opt,
    letter_dim, repet_num_fit, repet_num_fit, mcmc_num, thinning,
    'single_seq', target_letters, target_letter_rows, target_letter_cols,
    method_name, channel_name, job_id_opt, 'single_seq_log_lkd_multi',
    False, pred_odd_bool, sim_dat_bool,
    scale=scale_val, gamma=gamma_val
)

print('Classification for even sequence (testing sequence).')
pred_odd_bool = False

BayesGenSeqObj.standard_compute_pred_single_seq_multi(
    beta_comb_mcmc, cov_s_mcmc, arg_rho_mcmc, cov_t_set,
    eeg_code_even_3d, signal_x_even, d_mat_pred,
    letter_dim, repet_num_pred, repet_num_fit, mcmc_num, thinning,
    'single_seq', method_name, channel_name,
    job_id_opt, pred_odd_bool, sim_dat_bool,
    scale=scale_val, gamma=gamma_val
)

# bayes_even_no_first_dict = BayesGenSeqObj.standard_pred_cum_seq(
#     scale_opt, var_opt,
#     letter_dim, repet_num_pred, repet_num_fit, mcmc_num, thinning,
#     'single_seq', target_letters, target_letter_rows, target_letter_cols,
#     method_name, channel_name, job_id_opt, 'single_seq_log_lkd_multi',
#     True, pred_odd_bool, sim_dat_bool
# )
bayes_even_dict = BayesGenSeqObj.standard_pred_cum_seq(
    scale_opt, var_opt,
    letter_dim, repet_num_pred, repet_num_fit, mcmc_num, thinning,
    'single_seq', target_letters, target_letter_rows, target_letter_cols,
    method_name, channel_name, job_id_opt, 'single_seq_log_lkd_multi',
    False, pred_odd_bool, sim_dat_bool,
    scale=scale_val, gamma=gamma_val
)
