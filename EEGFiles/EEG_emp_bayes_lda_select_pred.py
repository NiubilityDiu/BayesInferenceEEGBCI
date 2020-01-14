import sys
sys.path.insert(0, './self_py_fun')
# from self_py_fun.MMAR_q_latent_z_multi import *
import self_py_fun.EEGConvolGlobal as gc
from self_py_fun.xDAFun import *
print(os.getcwd())
# tf.compat.v1.random.set_random_seed(612)
# np.random.seed(612)

sim_note = 'std_bool={}, kappa={}'.format(gc.std_bool, gc.kappa)
print(sim_note)
trn_repetition = 5
# channel_ids = np.array([5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]) - 1
channel_ids = None

LDAGibbsObj = XDAGibbs(
    # hyper-parameters:
    sigma_sq_delta=100,
    mu_1_delta=np.zeros([gc.num_electrode, gc.u, 1]),  # Later on, we need to change the mu_1_delta by OLS (done!)
    mu_0_delta=np.zeros([gc.num_electrode, gc.u, 1]),
    u=gc.u, a=gc.a, b=gc.b, kappa=gc.kappa, letter_dim=gc.letter_dim, trn_repetition=trn_repetition,
    # EEGPreFun
    data_type=gc.data_type,
    sub_folder_name=gc.sub_file_name,
    # EEGGeneralFun
    num_repetition=gc.num_repetition, num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    num_letter=gc.letter_dim, n_multiple=gc.n_multiple,
    local_bool=gc.local_use
)

phi_val, phi_fn = LDAGibbsObj.create_gaussian_kernel_fn(
    scale_1=gc.scale_1, u=gc.u, ki=gc.ki, scale_2=gc.scale_2, display_plot=False
)
phi_val = np.tile(phi_val[:, np.newaxis], [1, gc.num_electrode])
phi_fn = np.tile(phi_fn[np.newaxis, ...], [gc.num_electrode, 1, 1])
phi_fn_inner_prod = np.transpose(phi_fn, [0, 2, 1]) @ phi_fn
phi_fn_inner_prod_inv = LDAGibbsObj.compute_hermittan_matrix_inv(phi_fn_inner_prod)
phi_fn_ols_operator = phi_fn_inner_prod_inv @ np.transpose(phi_fn, [0, 2, 1])
# Import the training set without subsetting yet
[signals, eeg_code, eeg_type] = LDAGibbsObj.import_eeg_processed_dat(gc.file_subscript, reshape_to_1d=False)

eeg_code = np.reshape(eeg_code, [gc.letter_dim, gc.num_repetition * gc.num_rep])
eeg_type = np.reshape(eeg_type, [gc.letter_dim, gc.num_repetition * gc.num_rep])

print('trn_repetition = {}'.format(trn_repetition))

[signals_all, _,
 train_x_mat_tar, train_x_mat_ntar,
 train_tar_mean, train_ntar_mean,
 train_x_tar_sum, train_x_ntar_sum,
 train_x_tar_indices, train_x_ntar_indices,
 eeg_code_3d] = LDAGibbsObj.obtain_pre_processed_signals(
        signals, eeg_code, eeg_type, std_bool=gc.std_bool
)

# Use signals_train_t_mean and signals_train_nt_mean to compute mu_1/mu_0
# train_t_mean = phi_fn @ mu_1_delta, train_nt_mean = phi_fn @ mu_0_delta
# LDAGibbsObj.mu_1_delta = phi_fn_ols_operator @ train_tar_mean
# LDAGibbsObj.mu_0_delta = phi_fn_ols_operator @ train_ntar_mean
# Or without any prior information
LDAGibbsObj.mu_1_delta = phi_fn_ols_operator @ np.zeros_like(train_tar_mean)
LDAGibbsObj.mu_0_delta = phi_fn_ols_operator @ np.zeros_like(train_ntar_mean)

# Compute preliminary s_sq_est
# s_sq_est = np.var(np.concatenate([train_x_mat_tar, train_x_mat_ntar], axis=0), axis=(0, 2))

[delta_tar_mcmc, delta_ntar_mcmc,
 lambda_mcmc, gamma_mcmc,
 log_lhd_mcmc, pres_mat_band] = LDAGibbsObj.import_empirical_bayes_lda_mcmc()

delta_tar_mcmc_mean = np.mean(delta_tar_mcmc, axis=0)
delta_ntar_mcmc_mean = np.mean(delta_ntar_mcmc, axis=0)
lambda_mcmc_mean = np.mean(lambda_mcmc, axis=0)
gamma_mcmc_mean = np.mean(gamma_mcmc, axis=0)

message_eeg = gc.sub_file_name
# Truncated mean with standardization
LDAGibbsObj.save_lda_selection_indicator(
    train_tar_mean, train_ntar_mean,
    lambda_mcmc_mean, gamma_mcmc_mean,
    message_eeg, gc.sub_file_name[:4], phi_fn, method_name='emp_bayes_lda',
    threshold=gc.plot_threshold, mcmc=False
)

beta_tar_mcmc = phi_fn[np.newaxis, ...] @ (lambda_mcmc[..., np.newaxis, np.newaxis] * delta_tar_mcmc)
beta_ntar_mcmc = phi_fn[np.newaxis, ...] @ (lambda_mcmc[..., np.newaxis, np.newaxis] * delta_ntar_mcmc)

beta_tar_lower = np.quantile(beta_tar_mcmc, q=0.025, axis=0)
beta_tar_upper = np.quantile(beta_tar_mcmc, q=0.975, axis=0)
beta_ntar_lower = np.quantile(beta_ntar_mcmc, q=0.025, axis=0)
beta_ntar_upper = np.quantile(beta_ntar_mcmc, q=0.975, axis=0)

# posterior mode over mcmc iterations
LDAGibbsObj.save_lda_selection_indicator(
    delta_tar_mcmc_mean, delta_ntar_mcmc_mean,
    lambda_mcmc_mean, gamma_mcmc_mean,
    message_eeg, gc.sub_file_name[:4], phi_fn,
    method_name='emp_bayes_lda', threshold=gc.plot_threshold, mcmc=True,
    beta_tar_lower=beta_tar_lower, beta_tar_upper=beta_tar_upper,
    beta_ntar_lower=beta_ntar_lower, beta_ntar_upper=beta_ntar_upper
)

# Visualize log-likelihood over MCMC iterations (we don't know the truth for real data, so I use 0 instead.)
true_log_lhd = np.zeros([gc.num_electrode])
LDAGibbsObj.save_empirical_bayes_mcmc_trace_plot(
    lambda_mcmc, log_lhd_mcmc, gamma_mcmc_mean,
    true_log_lhd, gc.sub_file_name[:4]
)

# classification
if not gc.soft_bool:  # hard-threshold removal
    gamma_mat_hard = LDAGibbsObj.determine_selected_feature_matrix(gamma_mcmc_mean, gc.plot_threshold)
    if channel_ids is None:
        print('There are {} features selected with threshold {}'.format(np.sum(gamma_mat_hard), gc.plot_threshold))
    else:
        print('There are {} features selected with threshold {}'.format(np.sum(gamma_mat_hard[channel_ids, :]),
                                                                        gc.plot_threshold))
    mcmc_length = gamma_mcmc.shape[0]
    gamma_mcmc = np.tile(gamma_mat_hard[np.newaxis, ...], [mcmc_length, 1, 1])

lda_bayes_result = LDAGibbsObj.produce_lda_bayes_result_dict(
    signals_all, eeg_code_3d,
    delta_tar_mcmc, delta_ntar_mcmc,
    lambda_mcmc,
    gamma_mcmc, None, None, pres_mat_band,
    phi_fn, trn_repetition, gc.target_letters,
    channel_ids=channel_ids, soft_bool=gc.soft_bool
)

print('Proportion of correct prediction:')
mean_prop_pred = lda_bayes_result['mean']
max_prop_pred = lda_bayes_result['max']
letter_max_prop_pred = lda_bayes_result['letter_max']

for i, letter_i in enumerate(gc.target_letters):
    print('{}, Train:'.format(letter_i))
    print('Correctly pred: {}'.format(mean_prop_pred[i, :trn_repetition]))
    print('Max prob: {}'.format(max_prop_pred[i, :trn_repetition]))
    print('Max prob letter: {}'.format(letter_max_prop_pred[i, :trn_repetition]))
    print('{}, Test:'.format(letter_i))
    print('Correctly pred: {}'.format(mean_prop_pred[i, trn_repetition:]))
    print('Max prob: {}'.format(max_prop_pred[i, trn_repetition:]))
    print('Max prob letter: {}'.format(letter_max_prop_pred[i, trn_repetition:]))

LDAGibbsObj.save_lda_bayes_results(lda_bayes_result, message_eeg, gc.target_letters)