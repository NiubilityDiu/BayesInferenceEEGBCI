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

trn_repetitions = [15]


for _, trn_repetition in enumerate(trn_repetitions):

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
    s_sq_est = np.var(np.concatenate([train_x_mat_tar, train_x_mat_ntar], axis=0), axis=(0, 2))

    # Initialize the parameters:
    [delta_tar_mcmc, delta_ntar_mcmc,
     lambda_mcmc, gamma_mcmc,
     s_sq_mcmc, rho_mcmc
     ] = LDAGibbsObj.create_initial_values_lda(s_sq_est=s_sq_est.T)

    log_lhd_mcmc = []

    for k in range(LDAGibbsObj.kappa):
        delta_ntar_old = np.copy(delta_ntar_mcmc[k, ...])
        lambda_old = np.copy(lambda_mcmc[k, :])
        gamma_mat_iter = np.copy(gamma_mcmc[k, ...])
        # gamma_mat_iter = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # gamma_mat_iter = np.ones([1, LDAGibbsObj.n_length])
        s_sq_old = np.copy(s_sq_mcmc[k, :])
        rho_old = np.copy(rho_mcmc[k, :])
        # rho_old = 0.75 * np.ones(gc.num_electrode)
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
            phi_fn, gc.alpha_s, gc.beta_s, gc.zeta_lambda
        )
        # lambda_post = np.array([1])

        for tau in range(LDAGibbsObj.n_length):
            gamma_mat_iter = LDAGibbsObj.update_gamma_post_2(
                gamma_mat_iter, delta_tar_post, delta_ntar_post,
                lambda_post, pres_mat_old,
                tau, train_x_mat_tar, train_x_mat_ntar, phi_fn, gc.beta_ising, gc.gamma_neighbor
            )
        [s_sq_post, s_sq_accept] = LDAGibbsObj.update_s_sq_post_mh(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            s_sq_old, rho_old,
            train_x_mat_tar, train_x_mat_ntar, phi_fn,
            gc.alpha_s, gc.beta_s, gc.zeta_s
        )
        # s_sq_post = np.copy(s_sq_old)

        [rho_post, rho_accept] = LDAGibbsObj.update_rho_post_mh(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            s_sq_post, rho_old, train_x_mat_tar, train_x_mat_ntar, phi_fn, gc.zeta_rho
        )
        # rho_post = np.copy(rho_old)

        pres_mat_post = LDAGibbsObj.generate_proposal_ar1_pres_mat(s_sq_post, rho_post)
        mcmc_log_lhd_post = LDAGibbsObj.compute_sampling_log_lhd(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            pres_mat_post, train_x_mat_tar, train_x_mat_ntar, phi_fn
        )
        log_lhd_mcmc.append(mcmc_log_lhd_post)

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
    delta_tar_mcmc = delta_tar_mcmc[burn_in:, ...]
    delta_ntar_mcmc = delta_ntar_mcmc[burn_in:, ...]
    lambda_mcmc = lambda_mcmc[burn_in:, :]
    gamma_mcmc = gamma_mcmc[burn_in:, :]
    s_sq_mcmc = s_sq_mcmc[burn_in:, :]
    rho_mcmc = rho_mcmc[burn_in:, :]
    log_lhd_mcmc = np.stack(log_lhd_mcmc, axis=0)
    log_lhd_mcmc = log_lhd_mcmc[burn_in:, :]

    LDAGibbsObj.save_bayes_lda_mcmc(
        delta_tar_mcmc, delta_ntar_mcmc, lambda_mcmc, gamma_mcmc,
        s_sq_mcmc, rho_mcmc, log_lhd_mcmc)


