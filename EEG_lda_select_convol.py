import sys
sys.path.insert(0, './self_py_fun')
# from self_py_fun.MMAR_q_latent_z_multi import *
import self_py_fun.EEGConvolGlobal as gc
from self_py_fun.xDAFun import *
print(os.getcwd())
# tf.compat.v1.random.set_random_seed(612)
# np.random.seed(612)

u = 8
alpha_s = 5.0
beta_s = 5.0
zeta_lambda = 1e-4 * np.ones([gc.num_electrode])
zeta_s = 5e-3 * np.ones([gc.num_electrode])
zeta_rho = 1e-4 * np.ones([gc.num_electrode])
ki = 0.4
scale_1 = 0.15
scale_2 = 0.2
std_bool = True
beta_ising = 0.1
gamma_neighbor = 2
plot_threshold = 0.5
a = 1  # weight for target
b = 5  # weight for non-target
sim_note = 'std_bool={}, kappa={}'.format(std_bool, gc.kappa)
print(sim_note)

trn_repetitions = [10, 15]

for _, trn_repetition in enumerate(trn_repetitions):

    LDAGibbsObj = XDAGibbs(
        # hyper-parameters:
        sigma_sq_delta=100,
        mu_1_delta=np.zeros([gc.num_electrode, u, 1]),  # Later on, we need to change the mu_1_delta by OLS (done!)
        mu_0_delta=np.zeros([gc.num_electrode, u, 1]),
        u=u, a=a, b=b, kappa=gc.kappa, letter_dim=gc.letter_dim, trn_repetition=trn_repetition,
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
        scale_1=scale_1, u=u, ki=ki, scale_2=scale_2, display_plot=False
    )
    phi_val = np.tile(phi_val[:, np.newaxis], [1, gc.num_electrode])
    phi_fn = np.tile(phi_fn[np.newaxis, ...], [gc.num_electrode, 1, 1])
    # print('phi_val has shape {}'.format(phi_val.shape))
    # print('phi_fn has shape {}'.format(phi_fn.shape))

    phi_fn_inner_prod = np.transpose(phi_fn, [0, 2, 1]) @ phi_fn
    phi_fn_inner_prod_inv = LDAGibbsObj.compute_hermittan_matrix_inv(phi_fn_inner_prod)
    phi_fn_ols_operator = phi_fn_inner_prod_inv @ np.transpose(phi_fn, [0, 2, 1])
    # print('phi_fn_ols_operator has shape {}'.format(phi_fn_ols_operator.shape))

    # Import the training set without subsetting yet
    [signals, eeg_code, eeg_type] = LDAGibbsObj.import_eeg_processed_dat(gc.file_subscript, reshape_to_1d=False)

    eeg_code = np.reshape(eeg_code, [gc.letter_dim, gc.num_repetition * gc.num_rep])
    eeg_type = np.reshape(eeg_type, [gc.letter_dim, gc.num_repetition * gc.num_rep])

    print('trn_repetition = {}'.format(trn_repetition))

    [signals_all,
     train_x_mat_tar, train_x_mat_ntar,
     train_tar_mean, train_ntar_mean,
     train_x_tar_sum, train_x_ntar_sum,
     train_x_tar_indices, train_x_ntar_indices,
     eeg_code_3d] = LDAGibbsObj.obtain_pre_processed_signals(
            signals, eeg_code, eeg_type, std_bool=std_bool
    )

    # Use signals_train_t_mean and signals_train_nt_mean to compute mu_1/mu_0
    # train_t_mean = phi_fn @ mu_1_delta, train_nt_mean = phi_fn @ mu_0_delta
    # LDAGibbsObj.mu_1_delta = phi_fn_ols_operator @ train_tar_mean
    # LDAGibbsObj.mu_0_delta = phi_fn_ols_operator @ train_ntar_mean
    # Or without any prior information
    LDAGibbsObj.mu_1_delta = phi_fn_ols_operator @ np.zeros_like(train_tar_mean)
    LDAGibbsObj.mu_0_delta = phi_fn_ols_operator @ np.zeros_like(train_ntar_mean)
    # Compute preliminary s_sq_est
    s_sq_est = np.var(np.concatenate([train_x_mat_tar, train_x_mat_ntar], axis=0),
                      axis=(0, 2))

    # Initialize the parameters:
    [delta_tar_mcmc, delta_ntar_mcmc,
     lambda_mcmc, gamma_mcmc,
     s_sq_mcmc, rho_mcmc
     ] = LDAGibbsObj.create_initial_values_lda(s_sq_est=s_sq_est.T)

    mcmc_log_lhd = []

    for k in range(LDAGibbsObj.kappa):
        delta_ntar_old = np.copy(delta_ntar_mcmc[k, ...])
        lambda_old = np.copy(lambda_mcmc[k, :])
        gamma_mat_iter = np.copy(gamma_mcmc[k, ...])
        # gamma_mat_iter = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # gamma_mat_iter = np.ones([1, LDAGibbsObj.n_length])
        s_sq_old = np.copy(s_sq_mcmc[k, :])
        rho_old = np.copy(rho_mcmc[k, :])
        pres_mat_old = LDAGibbsObj.generate_proposal_ar1_pres_mat(s_sq_old, rho_old)

        # delta_tar_post = phi_fn_ols_operator @ true_beta_tar
        # delta_ntar_post = phi_fn_ols_operator @ true_beta_ntar
        delta_tar_post = LDAGibbsObj.update_delta_tar_post_lda_2(
            delta_ntar_old, lambda_old, gamma_mat_iter,
            pres_mat_old, train_x_tar_sum, train_x_ntar_sum, phi_fn
        )
        delta_ntar_post = LDAGibbsObj.update_delta_ntar_post_lda_2(
            delta_tar_post, lambda_old,
            gamma_mat_iter, pres_mat_old,
            train_x_tar_sum, train_x_ntar_sum, phi_fn
        )

        # kernel variance lambda_e
        [lambda_post, lambda_accept] = LDAGibbsObj.update_lambda_iter_post_mh(
            delta_tar_post, delta_ntar_post, lambda_old,
            gamma_mat_iter, s_sq_old, rho_old, train_x_mat_tar, train_x_mat_ntar,
            phi_fn, alpha_s, beta_s, zeta_lambda
        )
        # lambda_post = np.array([1])

        for tau in range(LDAGibbsObj.n_length):
            gamma_mat_iter = LDAGibbsObj.update_gamma_post_2(
                gamma_mat_iter, delta_tar_post, delta_ntar_post,
                lambda_post, pres_mat_old,
                tau, train_x_mat_tar, train_x_mat_ntar, phi_fn, beta_ising, gamma_neighbor
            )
        [s_sq_post, s_sq_accept] = LDAGibbsObj.update_s_sq_post_mh(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            s_sq_old, rho_old,
            train_x_mat_tar, train_x_mat_ntar, phi_fn,
            alpha_s, beta_s, zeta_s
        )
        # s_sq_post = np.copy(s_sq_old)

        [rho_post, rho_accept] = LDAGibbsObj.update_rho_post_mh(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            s_sq_post, rho_old, train_x_mat_tar, train_x_mat_ntar, phi_fn, zeta_rho
        )
        # rho_post = np.copy(rho_old)

        pres_mat_post = LDAGibbsObj.generate_proposal_ar1_pres_mat(s_sq_post, rho_post)
        mcmc_log_lhd_post = LDAGibbsObj.compute_sampling_log_lhd(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            pres_mat_post, train_x_mat_tar, train_x_mat_ntar, phi_fn
        )
        mcmc_log_lhd.append(mcmc_log_lhd_post)

        if k % gc.NUM_INTERVAL == 0 and k > 10:
            print('gibbs index = {}'.format(k+1))
            # print('delta_norm is \n {}'.format(delta_norm))
            print(np.mean(gamma_mcmc[(k-gc.NUM_INTERVAL):k, ...], axis=0))
            # print('gamma_mat_iter selected per channel = {}'
            #       .format(np.sum(gamma_mat_iter, axis=1)))
            print('mcmc_log_lhd_post = {}'.format(mcmc_log_lhd_post))

        # Append the result of each iteration
        delta_tar_mcmc = np.concatenate([delta_tar_mcmc, delta_tar_post[np.newaxis, ...]], axis=0)
        delta_ntar_mcmc = np.concatenate([delta_ntar_mcmc, delta_ntar_post[np.newaxis, ...]], axis=0)

        lambda_mcmc = np.concatenate([lambda_mcmc, lambda_post[np.newaxis, :]], axis=0)
        gamma_mcmc = np.concatenate([gamma_mcmc, gamma_mat_iter[np.newaxis, ...]], axis=0)

        s_sq_mcmc = np.concatenate([s_sq_mcmc, s_sq_post[np.newaxis, :]], axis=0)
        rho_mcmc = np.concatenate([rho_mcmc, rho_post[np.newaxis, :]], axis=0)

    burn_in = int(LDAGibbsObj.kappa * 0.75)
    gibbs_params_list = [
        delta_tar_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),
        delta_ntar_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),

        lambda_mcmc[burn_in:, :].astype(gc.DAT_TYPE),
        gamma_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),

        s_sq_mcmc[burn_in:, :].astype(gc.DAT_TYPE),
        rho_mcmc[burn_in:, :].astype(gc.DAT_TYPE)
    ]
    mcmc_log_lhd = np.stack(mcmc_log_lhd, axis=0)

    [delta_tar_mcmc_mean, delta_ntar_mcmc_mean,
     lambda_mcmc_mean, gamma_mcmc_mean,
     s_sq_mcmc_mean, rho_mcmc_mean] = [np.mean(gibbs_params_list[i], axis=0)
                                                        for i in range(len(gibbs_params_list))]
    message_eeg = gc.sub_file_name
    # Truncated mean with standardization
    LDAGibbsObj.save_lda_selection_indicator(
        train_tar_mean, train_ntar_mean,
        lambda_mcmc_mean, gamma_mcmc_mean,
        message_eeg, gc.sub_file_name[:4], phi_fn, threshold=plot_threshold, mcmc=False
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
        message_eeg, gc.sub_file_name[:4], phi_fn, threshold=plot_threshold, mcmc=True,
        beta_tar_lower=beta_tar_lower,
        beta_tar_upper=beta_tar_upper,
        beta_ntar_lower=beta_ntar_lower,
        beta_ntar_upper=beta_ntar_upper
    )

    # Visualize log-likelihood over MCMC iterations (we don't know the truth for real data, so I use 0 instead.)
    true_log_lhd = np.zeros([gc.num_electrode])
    LDAGibbsObj.save_mcmc_trace_plot(
        rho_mcmc, s_sq_mcmc, lambda_mcmc, mcmc_log_lhd, gamma_mcmc_mean,
        burn_in, true_log_lhd, gc.sub_file_name[:4]
    )
    # classification
    eeg_code_3d = np.reshape(eeg_code, [gc.letter_dim, gc.num_repetition, gc.num_rep])

    lda_bayes_result = LDAGibbsObj.produce_lda_bayes_result_dict(
        signals_all, eeg_code_3d,
        delta_tar_mcmc, delta_ntar_mcmc,
        lambda_mcmc,
        gamma_mcmc, s_sq_mcmc, rho_mcmc,
        phi_fn, trn_repetition, burn_in, gc.target_letters
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


