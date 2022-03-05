from self_py_fun.BayesGenFun import *
import self_py_fun.GlobalSIM as sg

np.random.seed(sg.seed_index)
q_mcmc = 2
method_name = 'BayesGenq' + str(q_mcmc)
file_subscript = '_'.join([sg.sim_type, sg.scenario_name, 'test'])
thinning = 10
channel_dim = sg.NUM_ELECTRODE
scenario_name = sg.scenario_name
letter_dim = sg.LETTER_DIM
# letter_dim = 5
letters_test = sg.LETTERS[:letter_dim]
letter_rows_test = sg.ROW_INDICES[:letter_dim]
letter_cols_test = sg.COL_INDICES[:letter_dim]
single_rep_dim = 1
# zeta_0 = 0.5
zeta_0 = float(sys.argv[6])
sim_dat_bool = True
# rho_level_num = 8
rho_level_num = int(sys.argv[7])


BayesGenSeqObj = BayesGenSeq(
    data_type=sg.data_type,
    sub_folder_name=sg.sim_common,
    sub_name_short='sim_' + str(sg.design_num + 1),
    num_repetition=sg.REPETITION_TRN,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=letter_dim,
    n_multiple=sg.N_MULTIPLE,
    local_bool=sg.local_use
)

job_id_dec_2 = 'binary_down_1_fit_zeta_{}'.format(zeta_0)

# Import testing dataset
[_, _, _, _, _,
 signals_x_test, eeg_code_test, eeg_type_test, _] = BayesGenSeqObj.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.REPETITION_TEST, file_subscript, 3
)
single_seq_len = sg.SEQ_LENGTH_FIT
letter_dim_2 = sg.LETTER_DIM
repet_num_pred = sg.REPETITION_TEST
signals_x_test_rs = np.zeros(
    [channel_dim, letter_dim_2, repet_num_pred, single_seq_len, 1]
)
for i in range(repet_num_pred):
    low_i = i * sg.NUM_REP * sg.window_length
    upp_i = low_i + single_seq_len
    signals_x_test_rs[:, :, i, :, :] = np.copy(signals_x_test[:, :, low_i:upp_i, :])

eeg_code_test = np.reshape(eeg_code_test, [letter_dim_2, repet_num_pred, sg.NUM_REP])
eeg_type_test = np.reshape(eeg_type_test, [letter_dim_2, repet_num_pred, sg.NUM_REP])

print('signals_x_test has shape {}'.format(signals_x_test.shape))
print('eeg_code_test has shape {}'.format(eeg_code_test.shape))
print('eeg_type_test has shape {}'.format(eeg_type_test.shape))

# Hyper-parameters
rho_set, rho_level, _, _, _ = BayesGenSeqObj.produce_pre_compute_rhos(
    q_mcmc, 10, rho_level_num
)
cov_t_set = [BayesGenSeqObj.create_ar2_pres_mat(1, rho_set[i], single_seq_len)[0]
             for i in range(rho_level)]
d_mat_pred = BayesGenSeqObj.create_design_mat_gen_bayes_seq(single_rep_dim)

# Import mcmc
[cov_s_mcmc, arg_rho_mcmc, zeta_mcmc, _,
 beta_tar_mcmc, beta_ntar_mcmc,
 scale_opt, var_opt, log_lkd_mcmc] = BayesGenSeqObj.import_mcmc(
    sg.sim_type, method_name, sg.repet_num_fit, scenario_name, job_id_dec_2, sim_dat_bool
)

'''
# Use true values
cov_s_mcmc = BayesGenSeqObj.create_compound_symmetry_cov_mat(20, 0.5, channel_dim)
cov_s_mcmc = np.tile(cov_s_mcmc[np.newaxis, ...], (101, 1, 1))
arg_rho_mcmc = np.zeros([101]) + 11
arg_rho_mcmc = arg_rho_mcmc.astype('int8')
'''

arg_rho_mcmc = np.squeeze(arg_rho_mcmc, axis=0)
beta_comb_mcmc = np.concatenate([beta_tar_mcmc, beta_ntar_mcmc], axis=-2)
mcmc_num = beta_comb_mcmc.shape[0]

log_prior_prob = np.log(1 / 36 * np.ones([36]))
prob_dist_total = []
letter_dist_total = []
target_letter_rank = []

print('Classification for testing sequence.')
pred_odd_bool = False

BayesGenSeqObj.standard_compute_pred_single_seq_multi(
    beta_comb_mcmc, cov_s_mcmc, arg_rho_mcmc, cov_t_set,
    eeg_code_test, signals_x_test_rs, d_mat_pred,
    letter_dim, repet_num_pred, sg.repet_num_fit, mcmc_num, thinning,
    'single_seq', method_name, scenario_name, job_id_dec_2, pred_odd_bool, sim_dat_bool
)

exclude_first_bool = False
bayes_even_dict = BayesGenSeqObj.standard_pred_cum_seq(
    scale_opt, var_opt,
    letter_dim, repet_num_pred, sg.repet_num_fit, mcmc_num, thinning,
    'single_seq', letters_test, letter_rows_test, letter_cols_test,
    method_name, scenario_name, job_id_dec_2, 'single_seq_log_lkd_multi',
    exclude_first_bool, pred_odd_bool, sim_dat_bool
)
