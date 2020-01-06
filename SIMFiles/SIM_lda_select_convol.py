import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.xDAFun import *
import self_py_fun.SIMConvolGlobal as sg
# import self_py_fun.EEGConvolGlobal as gc
# print(os.getcwd())
# tf.compat.v1.random.set_random_seed(612)
# np.random.seed(612)

sim_name = 'sim_' + str(sg.sim_name_id) + '_convol'
u = 8
alpha_s = 2.0
beta_s = 5.0
zeta_lambda = 1e-3 * np.ones([sg.num_electrode])
zeta_s = 1e-2 * np.ones([sg.num_electrode])
zeta_rho = 1e-4 * np.ones([sg.num_electrode])
ki = 0.4
scale_1 = 0.2
scale_2 = 0.2
std_bool = False
beta_ising = 0.1
gamma_neighbor = 2
plot_threshold = 0.5
a = 1  # weight for target
b = 5  # weight for non-target
sim_note = 'std_bool={}, kappa={}'.format(std_bool, sg.kappa)
print(sim_note)

trn_repetitions = [11]

for _, trn_repetition in enumerate(trn_repetitions):
    LDAGibbsObj = XDAGibbs(
        # hyper-parameters:
        sigma_sq_delta=100,
        u=u, a=a, b=b,
        mu_1_delta=np.zeros([sg.num_electrode, u, 1]),  # Later on, we need to change the mu_1_delta by OLS (done!)
        mu_0_delta=np.zeros([sg.num_electrode, u, 1]),
        # Later on, we need to change scale_mat by empirical bayes
        kappa=sg.kappa, letter_dim=sg.letter_dim, trn_repetition=trn_repetition,
        # EEGPreFun
        data_type=sg.data_type, sub_folder_name=sim_name,
        # EEGGeneralFun
        num_repetition=sg.num_repetition, num_electrode=sg.num_electrode,
        flash_and_pause_length=sg.flash_and_pause_length,
        num_letter=sg.letter_dim, n_multiple=sg.n_multiple,
        local_bool=sg.local_use
    )

    phi_val, phi_fn = LDAGibbsObj.create_gaussian_kernel_fn(
        scale_1=scale_1, u=u, ki=ki, scale_2=scale_2, display_plot=False
    )
    phi_val = np.tile(phi_val[:, np.newaxis], [1, sg.num_electrode])
    phi_fn = np.tile(phi_fn[np.newaxis, ...], [sg.num_electrode, 1, 1])
    '''
    phi_fn = np.eye(sg.n_length)[np.newaxis, ...]
    '''
    phi_fn_inner_prod = np.transpose(phi_fn, [0, 2, 1]) @ phi_fn
    phi_fn_inner_prod_inv = LDAGibbsObj.compute_hermittan_matrix_inv(phi_fn_inner_prod)
    phi_fn_ols_operator = phi_fn_inner_prod_inv @ np.transpose(phi_fn, [0, 2, 1])
    # print('phi_fn_ols_operator has shape {}'.format(phi_fn_ols_operator.shape))

    # MCMC with selection indicator
    [true_beta_tar, true_beta_ntar,
     cov_mat, _,
     signals, eeg_code, eeg_type,
     message] = LDAGibbsObj.import_simulation_results(sim_name, reshape_to_3d=False, as_pres=False)

    print('trn_repetition = {}'.format(LDAGibbsObj.trn_repetition))
    print('batch_dim_tar = {}'.format(LDAGibbsObj.batch_dim_tar))
    print('batch_dim_ntar = {}'.format(LDAGibbsObj.batch_dim_ntar))
    print('total_batch_train = {}'.format(LDAGibbsObj.total_batch_train))

    [signals_all,
     train_x_mat_tar, train_x_mat_ntar,
     train_tar_mean, train_ntar_mean,
     train_x_tar_sum, train_x_ntar_sum,
     train_x_tar_indices, train_x_ntar_indices,
     eeg_code_3d] = LDAGibbsObj.obtain_pre_processed_signals(
        signals, eeg_code, eeg_type, std_bool
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
     ] = LDAGibbsObj.create_initial_values_lda(s_sq_est.T)
    # print('delta_tar_mcmc has shape {}'.format(delta_tar_mcmc.shape))
    # print('gamma_mcmc has shape {}'.format(gamma_mcmc.shape))
    # print('s_sq_mcmc has shape {}'.format(s_sq_mcmc.shape))
    # print('rho_mcmc has shape {}'.format(rho_mcmc.shape))
    # s_sq_mcmc_accept = []
    # rho_mcmc_accept = []

    log_lhd_mcmc = []

    for k in range(LDAGibbsObj.kappa):

        delta_ntar_old = np.copy(delta_ntar_mcmc[k, ...])
        lambda_old = np.copy(lambda_mcmc[k, :])
        gamma_mat_iter = np.copy(gamma_mcmc[k, ...])
        # gamma_mat_iter = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # gamma_mat_iter = np.ones([1, LDAGibbsObj.n_length])
        s_sq_old = np.copy(s_sq_mcmc[k, :])
        # s_sq_old = np.copy(np.array([sim_name_id]))
        rho_old = np.copy(rho_mcmc[k, :])
        # rho_old = np.array([0.8])
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
        log_lhd_mcmc_post = LDAGibbsObj.compute_sampling_log_lhd(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            pres_mat_post, train_x_mat_tar, train_x_mat_ntar, phi_fn
        )
        log_lhd_mcmc.append(log_lhd_mcmc_post)

        if k % sg.NUM_INTERVAL == 0 and k > 10:
            print('gibbs index = {}'.format(k+1))
            print(np.mean(gamma_mcmc[(k-sg.NUM_INTERVAL):k, ...], axis=0))
            # print('gamma_mat_iter selected per channel = {}'
            #       .format(np.sum(gamma_mat_iter, axis=1)))
            print('log_lhd_mcmc_post = {}'.format(log_lhd_mcmc_post))

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

    delta_tar_mcmc_mean = np.mean(delta_tar_mcmc, axis=0)
    delta_ntar_mcmc_mean = np.mean(delta_ntar_mcmc, axis=0)
    lambda_mcmc_mean = np.mean(lambda_mcmc, axis=0)
    gamma_mcmc_mean = np.mean(gamma_mcmc, axis=0)
    s_sq_mcmc_mean = np.mean(s_sq_mcmc, axis=0)
    rho_mcmc_mean = np.mean(rho_mcmc, axis=0)

    # Truncated mean with standardization
    LDAGibbsObj.save_lda_selection_indicator(
        train_tar_mean, train_ntar_mean,
        lambda_mcmc_mean, gamma_mcmc_mean,
        message, sim_name, phi_fn, threshold=plot_threshold, mcmc=False
    )

    beta_tar_mcmc = phi_fn[np.newaxis, ...] @ (lambda_mcmc[..., np.newaxis, np.newaxis] * delta_tar_mcmc)
    beta_ntar_mcmc = phi_fn[np.newaxis, ...] @ (lambda_mcmc[..., np.newaxis, np.newaxis] * delta_ntar_mcmc)

    beta_tar_lower = np.quantile(beta_tar_mcmc, q=0.025, axis=0)
    beta_tar_upper = np.quantile(beta_tar_mcmc, q=0.975, axis=0)
    beta_ntar_lower = np.quantile(beta_ntar_mcmc, q=0.025, axis=0)
    beta_ntar_upper = np.quantile(beta_ntar_mcmc, q=0.975, axis=0)

    LDAGibbsObj.save_lda_selection_indicator(
        delta_tar_mcmc_mean, delta_ntar_mcmc_mean,
        lambda_mcmc_mean, gamma_mcmc_mean,
        message, sim_name, phi_fn, threshold=plot_threshold, mcmc=True,
        beta_tar_lower=beta_tar_lower,
        beta_tar_upper=beta_tar_upper,
        beta_ntar_lower=beta_ntar_lower,
        beta_ntar_upper=beta_ntar_upper
    )

    # Compare true log-likelihood and log-likelihood over MCMC iterations
    pres_mat_true = np.linalg.inv(cov_mat * sg.n_multiple)[np.newaxis, ...]
    gamma_mat_true = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # gamma_mat_true = np.ones([1, sg.n_length])
    pres_mat_true = np.tile(pres_mat_true, [sg.num_electrode, 1, 1])
    gamma_mat_true = np.tile(gamma_mat_true, [sg.num_electrode, 1])
    phi_fn_id = np.eye(sg.n_length)[np.newaxis, ...]
    lambda_true = np.ones([sg.num_electrode])

    # gamma_mat_true = np.ones([1, LDAGibbsObj.n_length])
    true_log_lhd = LDAGibbsObj.compute_sampling_log_lhd(
        train_tar_mean, train_ntar_mean,
        lambda_true, gamma_mat_true,
        pres_mat_true, train_x_mat_tar, train_x_mat_ntar,
        phi_fn_id
    )
    print('approx_true_log_lhd = {}'.format(true_log_lhd))

    LDAGibbsObj.save_mcmc_trace_plot(
        rho_mcmc, s_sq_mcmc, lambda_mcmc, log_lhd_mcmc, gamma_mcmc_mean,
        true_log_lhd, sim_name
    )

    # classification
    eeg_code_3d = np.reshape(eeg_code, [sg.letter_dim, sg.num_repetition, sg.num_rep])
    lda_bayes_result = LDAGibbsObj.produce_lda_bayes_result_dict(
        signals_all, eeg_code_3d,
        delta_tar_mcmc, delta_ntar_mcmc,
        lambda_mcmc, gamma_mcmc, s_sq_mcmc, rho_mcmc,
        phi_fn, trn_repetition, sg.letters
    )

    print('Proportion of correct prediction:')
    mean_prop_pred = lda_bayes_result['mean']
    max_prop_pred = lda_bayes_result['max']
    letter_max_prop_pred = lda_bayes_result['letter_max']

    for i, letter_i in enumerate(sg.letters):
        print('{}, Train:'.format(letter_i))
        print('Correctly pred: {}'.format(mean_prop_pred[i, :trn_repetition]))
        print('Max prob: {}'.format(max_prop_pred[i, :trn_repetition]))
        print('Max prob letter: {}'.format(letter_max_prop_pred[i, :trn_repetition]))
        print('{}, Test:'.format(letter_i))
        print('Correctly pred: {}'.format(mean_prop_pred[i, trn_repetition:]))
        print('Max prob: {}'.format(max_prop_pred[i, trn_repetition:]))
        print('Max prob letter: {}'.format(letter_max_prop_pred[i, trn_repetition:]))

    LDAGibbsObj.save_lda_bayes_results(lda_bayes_result, sim_name, sg.letters)
