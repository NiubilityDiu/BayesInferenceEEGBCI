import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.PreFun import *
from scipy import stats, linalg
plt.style.use('ggplot')
import seaborn as sns
sns.set_context('notebook')


class XDAGibbs(EEGPreFun):

    r"""
    Implement basic frequent method of LDA
    Implement Bayes LDA functions to BCI-speller with feature selection.
    Use cumulative log-likelihood over sequences as objective function.
    Produce credible intervals to determine automatic early stopping rule.
    """

    def __init__(self, sigma_sq_delta,
                 mu_1_delta, mu_0_delta,
                 a, b,  # weights
                 kappa,
                 letter_dim, trn_repetition,
                 *args,  **kwargs):
        super(XDAGibbs, self).__init__(*args, **kwargs)
        self.sigma_sq_delta = sigma_sq_delta
        self.mu_1_delta = mu_1_delta
        self.mu_0_delta = mu_0_delta
        self.a1 = a / (a + b),
        self.a0 = b / (a + b),
        self.kappa = kappa
        self.letter_dim = letter_dim  # Usually we use the entire letter dimension
        self.trn_repetition = trn_repetition
        self.batch_dim_tar = int(self.letter_dim * self.trn_repetition * self.flash_sum)
        self.batch_dim_ntar = int(self.letter_dim * self.trn_repetition * self.non_flash_sum)
        self.total_batch_train = self.batch_dim_tar + self.batch_dim_ntar
        self.total_batch_dim = int(self.num_letter * self.num_repetition * self.num_rep)
        self.feature_length = self.num_electrode * self.n_length
        self.identity_n = np.eye(self.n_length)

    def obtain_pre_processed_signals(
            self, raw_signals, eeg_code, eeg_type, std_bool=True
    ):
        r"""
        args:
        -----
            raw_signals: array_like, with shape (letter_dim, channel_dim, seq_length, 1)
            eeg_code: array_like, with shape (letter_dim, num_repetition * num_rep)
            eeg_type: array_like, with shape (letter_dim, num_repetition * num_rep)

        return:
        ----
            A list of arrays, including
                signals_all, signals_train, eeg_code_3d, \
                train_x_mat_tar, train_x_mat_ntar, \
                signals_train_tar_mean, signals_train_ntar_mean, \
                train_x_tar_sum, train_x_ntar_sum, \
                train_x_tar_indices, train_x_ntar_indices

        note:
        -----
            Convert convoluted signals to truncated segment signals as well as relevant summary statistics
        """
        # Extract eeg_code_train/eeg_type_train!
        eeg_code_3d = np.reshape(eeg_code, [self.num_letter, self.num_repetition, self.num_rep])
        eeg_type_train = eeg_type[:, :self.trn_repetition * self.num_rep]
        eeg_type_train_1d = np.reshape(eeg_type_train, [self.letter_dim * self.trn_repetition * self.num_rep])
        train_x_tar_indices = np.where(eeg_type_train_1d == 1)[0]
        train_x_ntar_indices = np.where(eeg_type_train_1d == 0)[0]

        signals_reshape = np.squeeze(raw_signals, axis=-1)
        # Convert to truncated segments
        signals_all, _ = self.create_truncate_segment_batch(
            signals_reshape, None,
            letter_dim=self.letter_dim, trn_repetition=self.num_repetition
        )
        signals_train, _ = self.create_truncate_segment_batch(
            signals_reshape, None,
            letter_dim=self.num_letter, trn_repetition=self.trn_repetition
        )
        [signals_train_tar_mean, signals_train_ntar_mean, _, _] = self.produce_trun_mean_cov_subset(
            signals_train, eeg_type_train_1d
        )
        # data, need 4d-array
        signals_all = signals_all[..., np.newaxis]
        signals_train = signals_train[..., np.newaxis]
        train_x_mat_tar = signals_train[train_x_tar_indices, ...]
        train_x_mat_ntar = signals_train[train_x_ntar_indices, ...]

        # mean, 3d-array
        signals_train_all_mean = np.mean(signals_train, axis=0)
        signals_train_tar_mean = signals_train_tar_mean[..., np.newaxis]
        signals_train_ntar_mean = signals_train_ntar_mean[..., np.newaxis]
        # sum, 3d-array
        train_x_tar_sum = np.sum(train_x_mat_tar, axis=0)
        train_x_ntar_sum = np.sum(train_x_mat_ntar, axis=0)

        if std_bool:
            print('We standard raw truncated segments by signals_train_all_mean.')
            # data, original 4d-dimension
            signals_all -= signals_train_all_mean[np.newaxis, ...]
            signals_train -= signals_train_all_mean[np.newaxis, ...]
            train_x_mat_tar -= signals_train_all_mean[np.newaxis, ...]
            train_x_mat_ntar -= signals_train_all_mean[np.newaxis, ...]
            # statistics, reduced dimension (no need add new dimension)
            train_x_tar_sum -= signals_train_all_mean
            train_x_ntar_sum -= signals_train_all_mean
            signals_train_tar_mean -= signals_train_all_mean
            signals_train_ntar_mean -= signals_train_all_mean

        return [
            signals_all, signals_train,
            train_x_mat_tar, train_x_mat_ntar,
            signals_train_tar_mean, signals_train_ntar_mean,
            train_x_tar_sum, train_x_ntar_sum,
            train_x_tar_indices, train_x_ntar_indices,
            eeg_code_3d
        ]

    def create_initial_values_bayes_lda(self, s_sq_est, rho_est, channel_ids):

        r"""
        args:
        -----
            s_sq_est: array_like, (num_electrode)
            rho_est: array_like, (num_electrode, q)
        return:
        -----
            A list of containing five arrays,
            delta_tar_mcmc, array_like, (1, num_electrode, u)
            delta_ntar_mcmc, array_like, (1, num_electrode, u)
            lambda_mcmc, array_like, (1, num_electrode)
            gamma_mcmc, array_like, (1, num_electrode, n_length)
            s_sq_mcmc, array_like, (1, num_electrode) (only when s_sq_est is not None)
            rho_mcmc, array_like, (1, num_electrode, q) (only when s_sq_est is not None)

        note:
        -----
            need to modify this function later on when there are many rhos in the precision matrix!
        """
        delta_tar_mcmc = self.mu_1_delta[np.newaxis, ...]
        delta_ntar_mcmc = self.mu_0_delta[np.newaxis, ...]
        ee = len(channel_ids)

        # lambda_tar_mcmc = np.ones([1, self.num_electrode])
        lambda_mcmc = np.ones([1, ee])
        gamma_mcmc = np.random.binomial(1, 0.5, size=(1, ee, self.n_length))

        s_sq_mcmc = s_sq_est[np.newaxis, :]
        rho_mcmc = rho_est[np.newaxis, ...]
        param_mcmc_list = [delta_tar_mcmc, delta_ntar_mcmc, lambda_mcmc, gamma_mcmc, s_sq_mcmc, rho_mcmc]

        return param_mcmc_list

    def update_delta_tar_post_lda_2(
            self, delta_ntar, lambda_iter,
            gamma_mat, pres_mat,
            trun_x_tar_sum, trun_x_ntar_sum, phi_fn, channel_ids
    ):
        r"""
        args:
        -----
            delta_ntar: array_like
                should have dimension (num_electrode, u, 1)
            lambda_iter: array_like
                should have dimension (num_electrode,)
            gamma_mat: array_like
                should have dimension (num_electrode, n_length)
            pres_mat: array_like
                precision matrix from previous iteration, shared by all truncated segments
                should have dimension (num_electrode, n_length, n_length)
            trun_x_tar_sum: array_like
                summation of individual truncated X_{v,i,j,e} segments with Z_{v,i,j} = 1
                should have the dimension (num_electrode, n_length, 1)
            trunx_ntar_sum: array_like
                summation of individual truncated X_{v,i,j,e} segments with Z_{v,i,j} = 0
               should have the dimension (num_electrode, n_length, 1)
            phi_fn: array_like
                basis function with gaussian kernel, should have dimension (num_electrode, n_length, u)
            channel_ids: integer_array

        return:
        -----
            array_like
                should have dimension (num_electrode, u, 1)
        """

        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]
        gamma_mat = self.block_diagonal_mat(gamma_mat, channel_ids)
        idns = self.identity_n[np.newaxis, ...]
        u = phi_fn.shape[-1]
        ids = np.eye(u)
        phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat
        a1_I_gamma = self.a1 * (idns - gamma_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat
        a0_I_gamma = self.a0 * (idns - gamma_mat)

        lambda_mat_tar = ids / self.sigma_sq_delta + lambda_iter**2 * phi_fn_t @ \
                         (self.batch_dim_tar * a1_I_a0_gamma @ pres_mat @ a1_I_a0_gamma +
                          self.batch_dim_ntar * a1_I_gamma @ pres_mat @ a1_I_gamma) @ phi_fn

        x_tar_sum_minus_ntar = trun_x_tar_sum - self.batch_dim_tar * lambda_iter * a0_I_gamma @ phi_fn @ delta_ntar
        x_ntar_sum_plus_ntar = trun_x_ntar_sum - self.batch_dim_ntar * lambda_iter * a0_I_a1_gamma @ phi_fn @ delta_ntar

        eta_tar = self.mu_1_delta / self.sigma_sq_delta + lambda_iter * phi_fn_t @ \
                  (a1_I_a0_gamma @ pres_mat @ x_tar_sum_minus_ntar +
                   a1_I_gamma @ pres_mat @ x_ntar_sum_plus_ntar)

        lambda_mat_tar_chol = np.linalg.cholesky(lambda_mat_tar)
        sigma_mat_tar_half = np.transpose(np.linalg.inv(lambda_mat_tar_chol), axes=(0, 2, 1))
        post_mu_delta_tar = np.stack([linalg.cho_solve((lambda_mat_tar_chol[i, ...], True), eta_tar[i, ...])
                                      for i in range(self.num_electrode)], axis=0)

        std_mvn_rv = stats.multivariate_normal(
            mean=np.zeros([u]),
            cov=1.0)
        delta_tar_post = std_mvn_rv.rvs(size=self.num_electrode)[..., np.newaxis]
        delta_tar_post = sigma_mat_tar_half @ delta_tar_post + post_mu_delta_tar

        return delta_tar_post

    def update_delta_ntar_post_lda_2(
            self, delta_tar, lambda_iter, gamma_mat, pres_mat,
            trun_x_tar_sum, trun_x_ntar_sum, phi_fn, channel_ids
    ):
        r"""
        args:
        -----
            delta_tar: array_like
                should have dimension (num_electrode, u, 1)
            lambda_iter: array_like,
                should have dimension (num_electrode,)
            gamma_mat: array_like
                should have dimension (num_electrode, n_length)
            pres_mat: array_like
                precision matrix from previous iteration, shared by all truncated segments
                should have dimension (num_electrode, n_length, n_length)
            trun_x_tar_sum: array_like
                summation of individual truncated X_{v,i,j,e} segments with Z_{v,i,j} = 1
                should have the same dimension as delta_post_1
            trunx_ntar_sum: array_like
                summation of individual truncated X_{v,i,j,e} segments with Z_{v,i,j} = 0
                should have the same dimension as delta_post_0
            phi_fn: array_like
                (num_electrode, n_length, u) array, each column represents the eigenfunction associated with the
                first u largest eigenvalues.

        return:
        -----
            array_like
                should have dimension (num_electrode, u, 1)
        """

        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]
        gamma_mat = self.block_diagonal_mat(gamma_mat, channel_ids)
        idns = self.identity_n[np.newaxis, ...]
        u = phi_fn.shape[-1]
        ids = np.eye(u)
        phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat
        a1_I_gamma = self.a1 * (idns - gamma_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat
        a0_I_gamma = self.a0 * (idns - gamma_mat)

        lambda_mat_ntar = ids / self.sigma_sq_delta + lambda_iter**2 * phi_fn_t @ \
                          (self.batch_dim_ntar * a0_I_a1_gamma @ pres_mat @ a0_I_a1_gamma +
                           self.batch_dim_tar * a0_I_gamma @ pres_mat @ a0_I_gamma) @ phi_fn

        x_tar_sum_plus_tar = trun_x_tar_sum - self.batch_dim_tar * lambda_iter * a1_I_a0_gamma @ phi_fn @ delta_tar
        x_ntar_sum_minus_tar = trun_x_ntar_sum - self.batch_dim_ntar * lambda_iter * a1_I_gamma @ phi_fn @ delta_tar

        eta_ntar = self.mu_0_delta / self.sigma_sq_delta + lambda_iter * phi_fn_t @ \
                   (a0_I_gamma @ pres_mat @ x_tar_sum_plus_tar +
                    a0_I_a1_gamma @ pres_mat @ x_ntar_sum_minus_tar)

        lambda_mat_ntar_chol = np.linalg.cholesky(lambda_mat_ntar)
        sigma_mat_ntar_half = np.transpose(np.linalg.inv(lambda_mat_ntar_chol), axes=(0, 2, 1))
        post_mu_delta_ntar = np.stack([linalg.cho_solve((lambda_mat_ntar_chol[i, ...], True), eta_ntar[i, ...])
                                       for i in range(self.num_electrode)], axis=0)

        std_mvn_rv = stats.multivariate_normal(
            mean=np.zeros([u]),
            cov=1.0)
        delta_ntar_post = std_mvn_rv.rvs(size=self.num_electrode)[..., np.newaxis]
        delta_ntar_post = sigma_mat_ntar_half @ delta_ntar_post + post_mu_delta_ntar

        return delta_ntar_post

    def update_gamma_post_2(
            self, gamma_mat_pre, delta_tar, delta_ntar,
            lambda_iter, pres_mat_pre,
            tau, trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
            channel_ids, beta_ising, gamma_neighbor
    ):
        r"""
        args:
        -----
            gamma_mat_pre: array_like
                probability of selection indicators of all electrodes from previous iteration,
                should have dimension of (num_electrode, self.n_length)
            delta_tar: array_like,
                should have dimension (num_electrode, u, 1)
            delta_ntar: array_like,
                should have dimension (num_electrode, u, 1)
            lambda_iter: array_like, should have dimension (num_electrode,)
            pres_mat_pre: square matrix,
                should have dimension (num_electrode, n_length, n_length)
            tau: integer, index of latency, from 0 to n-1
            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
            phi_fn: array_like
                (num_electrode, n_length, u) smoothing kernel matrix, each column represents the eigenfunction
                associated with first largest u eigenvalues.
            beta_ising: beta hyper-parameter for the ising prior
            gamma_neigbor: integer, the range for connectivity ~.

        return:
        -----
            array_like
            binary array the same dimension as gamma_mat_pre

        note:
        -----
            we assume independence across channels, and we update gamma_mat across channels as well.
        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        # Construct proposed state (binary, so we can enumerate them)
        gamma_mat_post_tau_0 = np.copy(gamma_mat_pre)
        gamma_mat_post_tau_0[:, tau] = 0
        gamma_mat_post_tau_1 = np.copy(gamma_mat_pre)
        gamma_mat_post_tau_1[:, tau] = 1

        # Compute the ising prior on the log scale
        log_prior_select = self.compute_ising_log_prior(
            gamma_mat_post_tau_1, tau, beta_ising, gamma_neighbor
        )
        log_prior_nselect = self.compute_ising_log_prior(
            gamma_mat_post_tau_0, tau, beta_ising, gamma_neighbor
        )
        # target, select tau-th feature across channels
        quad_select = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat_post_tau_1, pres_mat_pre,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids)
        # print('quad_select has shape {}'.format(quad_select.shape))
        # target, not select tau-th feature across channels
        quad_nselect = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat_post_tau_0, pres_mat_pre,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids)
        # print('quad_nselect has shape {}'.format(quad_nselect.shape))

        quad_log_odds = quad_select + log_prior_select - quad_nselect - log_prior_nselect
        quad_prop = np.zeros_like(quad_log_odds)
        for e in range(channel_dim):
            # Avoid np.exp() overflow
            if quad_log_odds[e] >= 100:
                quad_prop[e] = 1
            else:
                quad_odds_e = np.exp(quad_log_odds[e])
                quad_prop[e] = quad_odds_e / (1 + quad_odds_e)

        gamma_mat_post = np.copy(gamma_mat_pre)
        quad_ind = [np.random.binomial(1, quad_prop[i], 1)[0]
                    for i in range(channel_dim)]
        gamma_mat_post[:, tau] = np.array(quad_ind)

        return gamma_mat_post

    @staticmethod
    def compute_log_prior_ratio_lambda(lambda_old, lambda_new, alpha_s, beta_s, channel_ids):
        r"""
        args:
        -----
            lambda_old: array_like, (num_electrode,)
            lambda_new: array_like, (num_electrode,)
            alpha_s: shape, scalar value, >2 to have valid variance
            beta_s: scale, scalar value, >1 to have valid mean (rate = 1/scale)

        return:
        -----
            array_like value, (num_electrode,)
        note:
        -----
            we assume lambda ~ Gamma(alpha_s, beta_s)
            Not to be confused, I always use alpha, beta parametrization
            beta_s is the inverse of scale!
        """
        assert lambda_new.shape == lambda_old.shape == (len(channel_ids),)

        lambda_rv = stats.gamma(a=alpha_s)
        lambda_old_log_pdf = lambda_rv.logpdf(lambda_old * beta_s)
        lambda_new_log_pdf = lambda_rv.logpdf(lambda_new * beta_s)

        return lambda_new_log_pdf - lambda_old_log_pdf

    def compute_sampling_log_lhd(
            self, delta_tar, delta_ntar,
            lambda_iter, gamma_mat, pres_mat,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
    ):
        r"""
        args:
        ------
            delta_tar: array_like, (num_electrode, u, 1)
            delta_ntar: array_like, (num_electrode, u, 1)
            lambda_iter: array_like, (num_electrode,)
            gamma_mat: array_like, (num_electrode, n_length)
            pres_mat: array_like, (num_electrode, n_length, n_length)
            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
            phi_fn: array_like, (num_electrode, n_length, u)

        return:
        ------
            array_like, total_log_prob (excluding 2pi constant), (num_electrode,)

        note:
        ------
            which should be the universal function to compute the log-prob, easy to check later!
        """
        # print('lambda_tar has shape {}'.format(lambda_tar.shape))
        # print('lambda_ntar has shape {}'.format(lambda_ntar.shape))

        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]
        gamma_mat = self.block_diagonal_mat(gamma_mat, channel_ids)
        idns = self.identity_n[np.newaxis, ...]
        # phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat
        a1_I_gamma = self.a1 * (idns - gamma_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat
        a0_I_gamma = self.a0 * (idns - gamma_mat)

        mean_tar = lambda_iter * (a1_I_a0_gamma @ phi_fn @ delta_tar + a0_I_gamma @ phi_fn @ delta_ntar)
        mean_ntar = lambda_iter * (a0_I_a1_gamma @ phi_fn @ delta_ntar + a1_I_gamma @ phi_fn @ delta_tar)

        trun_x_mat_tar_diff = trun_x_mat_tar - mean_tar[np.newaxis, ...]
        trun_x_mat_ntar_diff = trun_x_mat_ntar - mean_ntar[np.newaxis, ...]
        trun_x_mat_tar_diff_t = np.transpose(trun_x_mat_tar_diff, [0, 1, 3, 2])
        trun_x_mat_ntar_diff_t = np.transpose(trun_x_mat_ntar_diff, [0, 1, 3, 2])

        # quadtraic part:
        log_quad_sum = -1/2 * (np.sum(trun_x_mat_tar_diff_t @ pres_mat[np.newaxis, ...] @ trun_x_mat_tar_diff, axis=0) +
                               np.sum(trun_x_mat_ntar_diff_t @ pres_mat[np.newaxis, ...] @ trun_x_mat_ntar_diff, axis=0))
        log_quad_sum = np.squeeze(log_quad_sum, axis=(-2, -1))
        [sgn, logdet_abs] = np.linalg.slogdet(pres_mat)
        log_pres_det = 1/2 * sgn * logdet_abs * self.total_batch_train

        return log_quad_sum + log_pres_det

    def update_lambda_iter_post_mh(
            self, delta_tar, delta_ntar,
            lambda_old,
            gamma_mat, pres_mat,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids,
            alpha_s, beta_s, zeta_lambda,
    ):
        r"""
           args:
           -----
               delta_tar: array_like, (num_electrode, u, 1)
               delta_ntar: array_like, (num_electrode, u, 1)
               lambda_old: array_like, (num_electrode,)
               gamma_mat: array_like, (num_electrode, n_length)
               pres_mat: array_like, (num_electrode, n_length, n_length)
               trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
               trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
               phi_fn: array_like, (num_electrode, n_length, u)
               alpha_s: hyper-parameter
               beta_s: hyper-parameter
               zeta_lambda: step size

           return:
           -----
               A list of two arrays including lambda_tar_post (num_electrode,), and
               acceptance indicator with shape (num_electrode,)
        """
        # Generate new state
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        lambda_new, lambda_accept_init = self.generate_proposal_sigma_sq_state(lambda_old, zeta_lambda, channel_ids)
        log_prior_ratio = self.compute_log_prior_ratio_s_sq(lambda_old, lambda_new, alpha_s, beta_s, channel_ids)

        log_sampling_old = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_old,
            gamma_mat, pres_mat, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
        )

        log_sampling_new = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_new,
            gamma_mat, pres_mat, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
        )
        log_sampling_ratio = log_sampling_new - log_sampling_old

        log_alpha_mh = log_prior_ratio + log_sampling_ratio  # log_proposal_ratio = 0
        log_uniform = np.log(np.random.uniform(low=0, high=1, size=channel_dim))

        lambda_post = np.copy(lambda_old)
        # Compute acceptance rate and help adjust step size
        lambda_accept = np.zeros_like(lambda_old)

        for e in range(channel_dim):
            if log_alpha_mh[e] > 0:
                log_alpha_mh[e] = 0

            if log_alpha_mh[e] >= log_uniform[e]:
                lambda_post[e] = np.copy(lambda_new[e])
                lambda_accept[e] = 1

        return [lambda_post, lambda_accept]

    def update_s_sq_post_mh(
            self, delta_tar, delta_ntar,
            lambda_iter, gamma_mat,
            s_sq_old, rho_old,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids,
            alpha_s, beta_s, zeta_s, q
    ):
        r"""
        args:
        -----
            delta_tar: array_like, (num_electrode, u, 1)
            delta_ntar: array_like, (num_electrode, u, 1)
            lambda_iter: array_like, (num_electrode,)
            gamma_mat: array_like, (num_electrode, n_length)
            s_sq_old: array_like, previous state of sigma_sq, (num_electrode, q)
            rho_old: array_like, previous state of rho, (num_electrode,)
            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
            phi_fn: array_like, (num_electrode, n_length, u)
            alpha_s: hyper-parameter
            beta_s: hyper-parameter
            zeta_s: step size
            q: autoregressive order

        return:
        -----
            A list of two arrays including s_sq_post (num_electrode,),
            and acceptance indicator with shape (num_electrode,)

        note:
        -----
            To simplify the computation, we use random walk with gaussian distribution,
            so the proposal ratio = 1
            log alpha (pres_mat_new | pres_mat_old) = min (0, log_sampling_ratio + log_prior_ratio)
            log_sampling_ratio = log_sampling_log_new - log_sampling_log_old
            log_prior_ratio = log_prior_prob_new - log_prior_prob_old

            then compare to log Uniform (0,1)
            Accept if log alpha >= log Uniform (0, 1), reject otherwise

        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        # Generate proposed state
        s_sq_new, s_sq_accept_init = self.generate_proposal_sigma_sq_state(s_sq_old, zeta_s, channel_ids)
        log_prior_ratio = self.compute_log_prior_ratio_s_sq(s_sq_old, s_sq_new, alpha_s, beta_s, channel_ids)

        # Generate pres_mat_old and pres_mat_new (only change tau-index of s_sq_old)
        # Currently, we use all channels to perform mcmc
        pres_mat_old = self.generate_ar1_pres_mat(s_sq_old, rho_old, channel_ids, q)
        pres_mat_new = self.generate_ar1_pres_mat(s_sq_new, rho_old, channel_ids, q)

        log_sampling_old = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_old, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
        )

        log_sampling_new = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_new, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
        )
        log_sampling_ratio = log_sampling_new - log_sampling_old

        log_alpha_mh = log_prior_ratio + log_sampling_ratio  # + log_proposal_ratio = 0
        log_uniform = np.log(np.random.uniform(low=0, high=1, size=channel_dim))

        s_sq_post = np.copy(s_sq_old)
        # Compute acceptance rate and help adjust step size
        s_sq_accept = np.zeros_like(s_sq_old)

        for e in range(channel_dim):
            if log_alpha_mh[e] > 0:
                log_alpha_mh[e] = 0

            if log_alpha_mh[e] >= log_uniform[e]:
                s_sq_post[e] = np.copy(s_sq_new[e])
                s_sq_accept[e] = 1

        return [s_sq_post, s_sq_accept]

    def update_rho_post_mh(
            self, delta_tar, delta_ntar,
            lambda_iter, gamma_mat,
            s_sq, rho_old,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids,
            zeta_rho, q
    ):
        r"""
        args:
        -----
            delta_tar: array_like, (num_electrode, u, 1)
            delta_ntar: array_like, (num_electrode, u, 1)
            lambda_iter: array_like, (num_electrode,)
            gamma_mat: array_like, (num_electrode, n_length)
            s_sq: array_like, iter state of sigma_sq, (num_electrode,)
            rho_old: array_like, previous state of rho, (num_electrode, q)
            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
            phi_fn: array_like, (num_electrode, n_length, u)
            zeta_rho: step size, (num_electrode,)
            q: autoregressive order,

        return:
        -----
            A list of two arrays including rho_post (num_electrode,),
            and acceptance indicator with shape (num_electrode,)

        note:
            To simplify the computation, we use random walk with gaussian distribution,
            so the proposal ratio = 1
            log alpha (pres_mat_new | pres_mat_old) = min (0, log_sampling_ratio + log_prior_ratio)
            log_sampling_ratio = log_sampling_log_new - log_sampling_log_old
            log_prior_ratio = log_prior_prob_new - log_prior_prob_old

            then compare to log Uniform (0,1)
            Accept if log alpha >= log Uniform (0, 1), reject otherwise

        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        # Generate proposed state
        rho_new = self.generate_proposal_rho_state(rho_old, zeta_rho, channel_ids, q)
        log_prior_ratio = self.compute_log_prior_ratio_rho(rho_old, rho_new, channel_ids, a=-1, b=1, q=q)

        # Generate pres_mat_old and pres_mat_new (only change rho)
        # channel_ids = np.arange(channel_dim)
        pres_mat_old = self.generate_ar1_pres_mat(s_sq, rho_old, self.n_length, channel_ids, q)
        pres_mat_new = self.generate_ar1_pres_mat(s_sq, rho_new, self.n_length, channel_ids, q)
        pdf_bool = []
        # determine whether new pres_mat is positive definite
        for i in range(channel_dim):
            pdf_bool_i = self.is_pos_def(pres_mat_new[i, ...])
            if not pdf_bool_i:
                pres_mat_new[i, ...] = np.copy(pres_mat_old[i, ...])
            pdf_bool.append(pdf_bool_i)
        # print(pdf_bool)

        log_sampling_old = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_old, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
        )
        log_sampling_new = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_new, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, channel_ids
        )
        log_sampling_ratio = log_sampling_new - log_sampling_old
        # print('log_sampling_new = {}'.format(log_sampling_new))
        # print('log_sampling_old = {}'.format(log_sampling_old))
        # print('log_sampling_ratio = {}'.format(log_sampling_ratio))

        log_alpha_mh = log_prior_ratio + log_sampling_ratio  # log_proposal_ratio = 0
        log_uniform = np.log(np.random.uniform(low=0, high=1, size=channel_dim))

        rho_post = np.copy(rho_old)
        # Compute acceptance rate and help adjust step size
        rho_accept = np.zeros([channel_dim])

        # log_alpha_mh[np.greater(log_alpha_mh, 0)] = 0

        for e in range(channel_dim):
            if log_alpha_mh[e] > 0:
                log_alpha_mh[e] = 0

            if log_alpha_mh[e] >= log_uniform[e] and pdf_bool[e]:
                rho_post[e, :] = np.copy(rho_new[e, :])
                # Here, we update q rhos each time,
                # shall change it when we sequentially update it.
                if rho_post[e, 0] != rho_old[e, 0]:
                    rho_accept[e] = 1

        return [rho_post, rho_accept]

    def adjust_s_sq_rho_step_size(self, zeta_s, zeta_rho, s_sq_accept_100, rho_accept_100):

        r"""
        args:
        -----
            zeta_s: array_like, previous step size of zeta_s, (num_electrode,)
            zeta_rho: array_like, previous step size of zeta_rho, (num_electrode,)
            s_sq_accept_100: array_like, the acceptance result array for 100 iterations, (100, num_electrode)
            rho_accept_100: array_like, the acceptance result array for 100 iterations, (100, num_electrode)

        return:
        -----
            A list of new step size w.r.t zeta_s and zeta_rho
        """

        accept_rate_low = 0.4
        accept_rate_high = 0.7
        s_sq_accept_rate = rho_accept_rate = np.array([0.5])

        if rho_accept_100 is None:
            zeta_rho = np.copy(zeta_rho)
        else:
            rho_accept_rate = np.mean(rho_accept_100, axis=0)

        if s_sq_accept_100 is None:
            zeta_s = np.copy(zeta_s)
        else:
            s_sq_accept_rate = np.mean(s_sq_accept_100, axis=0)

        for e in range(self.num_electrode):

            if s_sq_accept_rate[e] < accept_rate_low:
                zeta_s[e] = 0.9 * np.copy(zeta_s[e])
            elif s_sq_accept_rate[e] > accept_rate_high:
                zeta_s[e] = 1.1 * np.copy(zeta_s[e])

            if rho_accept_rate[e] < accept_rate_low:
                zeta_rho[e] = 0.9 * np.copy(zeta_rho[e])
            elif rho_accept_rate[e] > accept_rate_high:
                zeta_rho[e] = 1.1 * np.copy(zeta_rho[e])

        return [zeta_s, zeta_rho]

    def save_lda_selection_indicator(
            self, delta_tar, delta_ntar,
            lambda_iter,
            gamma_mcmc_mean, message, sim_folder_name, phi_fn,
            channel_ids, method_name, threshold=0.5, mcmc=True,
            beta_tar_lower=None, beta_tar_upper=None,
            beta_ntar_lower=None, beta_ntar_upper=None
    ):
        r"""
        args:
        -----
            delta_tar: array_like,
                if mcmc == True, (num_electrode, u, 1),
                else, (num_electrode, n_length, 1)
            delta_ntar: array_like,
                if mcmc == True, (num_electrode, u, 1),
                else, (num_electrode, n_length, 1)
            lambda_tar: array_like, (num_electrode),
            lambda_ntar: array_like, (num_electrode),
            gamma_mcmc_mean: array_like, (num_electrode, n_length)
            ...
            phi_fn: array_like, (num_electrode, n_length, u)
        return:
        -----
            plots of beta_tar vs beta_ntar
        """
        ee = len(channel_ids)
        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]

        if mcmc:
            beta_tar = phi_fn @ (lambda_iter * delta_tar)
            beta_ntar = phi_fn @ (lambda_iter * delta_ntar)
        else:
            beta_tar = np.copy(delta_tar)
            beta_ntar = np.copy(delta_ntar)

        gamma_mcmc_binary = np.zeros_like(gamma_mcmc_mean)
        gamma_mcmc_binary[gamma_mcmc_mean > threshold] = 1
        x = list(self.time_range)
        gamma_mcmc_mean = np.around(gamma_mcmc_mean, decimals=3)

        if 'convol' in sim_folder_name:
            if mcmc:
                plot_name = method_name + '_convol_mcmc_mean_trn_' + str(self.trn_repetition)
                message = message + '_mcmc_trn_' + str(self.trn_repetition)
            else:
                plot_name = method_name + '_convol_std_mean_trn_' + str(self.trn_repetition)
                message = message + '_std_trn_' + str(self.trn_repetition)
        else:
            if mcmc:
                plot_name = method_name + '_mcmc_mean_trn_' + str(self.trn_repetition)
                message = message + '_mcmc_trn_' + str(self.trn_repetition)
            else:
                plot_name = method_name + '_std_mean_trn_' + str(self.trn_repetition)
                message = message + '_std_trn_' + str(self.trn_repetition)

        plot_pdf = bpdf.PdfPages('{}/{}/{}/{}/{}_{}.pdf'
                                 .format(self.parent_path,
                                         self.data_type,
                                         sim_folder_name,
                                         method_name,
                                         sim_folder_name,
                                         plot_name))
        if mcmc:
            for i in range(ee):

                fig_1 = plt.figure(figsize=(12, 10))
                ax1 = fig_1.add_subplot(2, 1, 1)
                ax1.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="tar_mcmc")
                ax1.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="ntar_mcmc")
                ax1.fill_between(self.time_range, beta_tar_lower[i, :, 0], beta_tar_upper[i, :, 0],
                                 color='red', alpha=0.2)
                ax1.fill_between(self.time_range, beta_ntar_lower[i, :, 0], beta_ntar_upper[i, :, 0],
                                 color='blue', alpha=0.2)
                ax1.legend(loc='upper right')
                # ax1.title.set_text(message + '_95%_credible_band_chan_' + str(i+1))
                ax1.title.set_text('Mean Curve with 95% Credible Band')
                # plt.show()
                ax2 = fig_1.add_subplot(2, 1, 2)
                ax2.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="tar_mcmc")
                ax2.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="ntar_mcmc")
                for j in range(self.n_length):
                    if gamma_mcmc_binary[i, j] == 0:
                        half_value = 1/6 * beta_tar[i, j, 0] + 5/6 * beta_ntar[i, j, 0]
                        beta_tar[i, j, 0] = np.copy(half_value)
                        beta_ntar[i, j, 0] = np.copy(half_value)
                ax2.plot(self.time_range, beta_tar[i, :, 0], 'r-', label="tar_select")
                ax2.plot(self.time_range, beta_ntar[i, :, 0], 'b-', label="ntar_select")
                for x_i, y_i, prop_i in zip(x, list(beta_tar[i, :, 0]), list(gamma_mcmc_mean[i, :])):
                    plt.text(x_i, y_i, str(prop_i))
                ax2.hlines(y=0, xmin=self.time_range[0], xmax=self.time_range[-1])
                ax2.legend(loc="upper right")
                ax2.title.set_text(message + '_threshold_' + str(threshold) + '_chan_' + str(i+1))
                plt.show()
                plt.close()
                plot_pdf.savefig(fig_1)

                '''
                plt.figure()
                plt.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="target")
                plt.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="non-target")
                for j in range(self.n_length):
                    if gamma_mcmc_binary[i, j] == 0:
                        half_value = 1 / 6 * beta_tar[i, j, 0] + 5 / 6 * beta_ntar[i, j, 0]
                        beta_tar[i, j, 0] = np.copy(half_value)
                        beta_ntar[i, j, 0] = np.copy(half_value)
                plt.plot(self.time_range, beta_tar[i, :, 0], 'r-', label="target-select")
                plt.plot(self.time_range, beta_ntar[i, :, 0], 'b-', label="non-target-select")
                for x_i, y_i, prop_i in zip(x, list(beta_tar[i, :, 0]), list(gamma_mcmc_mean[i, :])):
                    plt.text(x_i, y_i, str(prop_i))
                plt.hlines(y=0, xmin=self.time_range[0], xmax=self.time_range[-1])
                plt.legend(loc="upper right")
                plt.title('Mean Curve with Selection Indicator')
                plt.show()
                '''
        else:
            for i in range(ee):
                fig = plt.figure(figsize=(12, 10))
                plt.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="tar_std")
                plt.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="ntar_std")
                for j in range(self.n_length):
                    if gamma_mcmc_binary[i, j] == 0:
                        half_value = 1/6 * beta_tar[i, j, 0] + 5/6 * beta_ntar[i, j, 0]
                        beta_tar[i, j, 0] = np.copy(half_value)
                        beta_ntar[i, j, 0] = np.copy(half_value)
                plt.plot(self.time_range, beta_tar[i, :, 0], 'r-', label="tar_select")
                plt.plot(self.time_range, beta_ntar[i, :, 0], 'b-', label="ntar_select")
                for x_i, y_i, prop_i in zip(x, list(beta_tar[i, :, 0]), list(gamma_mcmc_mean[i, :])):
                    plt.text(x_i, y_i, str(prop_i))
                plt.hlines(y=0, xmin=self.time_range[0], xmax=self.time_range[-1])
                plt.legend(loc="upper right")
                plt.title(message + '_threshold_' + str(threshold) + '_chan_' + str(i+1))
                # plt.show()
                plt.close()
                plot_pdf.savefig(fig)
        plot_pdf.close()

    def lda_two_step_estimation_mcmc_i(
            self, eeg_signals_trun, eeg_code,
            delta_tar_i, delta_ntar_i,
            lambda_iter_i,
            pres_mat_i, gamma_i,
            phi_fn, trn_repetition, channel_ids, soft_bool=True
    ):
        r"""
        args:
        -----
            eeg_signals_trun: array_like
                truncated eeg signals X_{v,i,j,e}, should have the dimension
                (letter_dim*num_repetition*self.num_rep, num_electrode, self.n_length, 1)
            eeg_code: array_like
                3d integer array representing the stimulus code, should have the dimension
                (letter_dim, num_repetition, self.num_rep)
            delta_tar_i: array_like
                mean vector of target stimuli during iteration i, should have dimension
                (num_electrode, self.n_length, 1)
            delta_ntar_i: array_like
                mean_vector of non-target stimuli during iteration i, should have dimension
                (num_electrode, self.n_length, 1)
            lambda_iter_i: array_like, (num_electrode,)
            pres_mat_i: array_like, (num_electrode, n_length, n_length)
            gamma_i: array_like
                when soft_bool is true, selection indicator, should have the same dimension (num_electrode, n_length)
                when soft_bool is false, it is fixed over i
            phi_fn: array_like
                should have input shape (num_electrode, n_length, u)
            trn_repetition: integer
                the number of sequence repetitions in the training set
            channel_ids: integer array_like
            soft_bool: boolean variable, whether we need to remove the feature completely

        return:
        -----
            array containing the predicted letter for the v-th target letter, i-th sequence.
            should have the dimension (letter_dim, rep_dim).

        note:
        -----
            the num_electrode may change to len(channel_dim),
            depending on whether we use the entire num_electrode dataset to predict.
        """

        lambda_iter_i = lambda_iter_i[:, np.newaxis, np.newaxis]
        gamma_i_mat = self.block_diagonal_mat(gamma_i, channel_ids)
        # pres_mat_i = self.generate_proposal_ar1_pres_mat(s_sq_i, rho_i, channel_ids=channel_ids)

        if soft_bool:
            idns = self.identity_n[np.newaxis, ...]
            a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_i_mat
            a1_I_gamma = self.a1 * (idns - gamma_i_mat)
            a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_i_mat
            a0_I_gamma = self.a0 * (idns - gamma_i_mat)
            mean_tar = lambda_iter_i * (a1_I_a0_gamma @ phi_fn @ delta_tar_i + a0_I_gamma @ phi_fn @ delta_ntar_i)
            mean_ntar = lambda_iter_i * (a0_I_a1_gamma @ phi_fn @ delta_ntar_i + a1_I_gamma @ phi_fn @ delta_tar_i)
        else:
            # Directly use the beta mean and ignore the mixture model
            # pres_mat_i is reduced to the smaller square matrix where those
            # selected features contribute to the log-likelihood only
            pres_mat_i = gamma_i_mat @ pres_mat_i @ gamma_i_mat
            mean_tar = lambda_iter_i * phi_fn @ delta_tar_i
            mean_ntar = lambda_iter_i * phi_fn @ delta_ntar_i

        trun_diff_i_1 = eeg_signals_trun - mean_tar[np.newaxis, ...]
        trun_diff_i_0 = eeg_signals_trun - mean_ntar[np.newaxis, ...]
        trun_diff_i_1_t = np.transpose(trun_diff_i_1, [0, 1, 3, 2])
        trun_diff_i_0_t = np.transpose(trun_diff_i_0, [0, 1, 3, 2])

        log_quad_i_1 = trun_diff_i_1_t @ pres_mat_i @ trun_diff_i_1
        log_quad_i_0 = trun_diff_i_0_t @ pres_mat_i @ trun_diff_i_0
        log_quad_1 = np.sum(log_quad_i_1, axis=(1, 2, 3))
        log_quad_0 = np.sum(log_quad_i_0, axis=(1, 2, 3))
        log_prob_mvn_1 = -1/2*log_quad_1
        log_prob_mvn_0 = -1/2*log_quad_0

        l_mvn_1_ordered = []
        l_mvn_0_ordered = []
        eeg_code_flat = np.reshape(eeg_code, [self.num_letter * self.num_repetition * self.num_rep])

        for i in range(self.num_rep):
            l_mvn_1_ordered.append(log_prob_mvn_1[eeg_code_flat == i + 1])
            l_mvn_0_ordered.append(log_prob_mvn_0[eeg_code_flat == i + 1])

        l_mvn_1_ordered = np.stack(l_mvn_1_ordered, axis=1)
        l_mvn_0_ordered = np.stack(l_mvn_0_ordered, axis=1)

        l_mvn_1_ordered = np.reshape(l_mvn_1_ordered,
                                     [self.num_letter, self.num_repetition, self.num_rep])
        l_mvn_0_ordered = np.reshape(l_mvn_0_ordered,
                                     [self.num_letter, self.num_repetition, self.num_rep])

        log_lhd_row = np.zeros([self.num_letter, self.num_repetition, int(self.num_rep / 2)])
        log_lhd_col = np.zeros([self.num_letter, self.num_repetition, int(self.num_rep / 2)])
        row_indices = np.arange(1, self.row_column_length+1)
        col_indices = np.arange(self.row_column_length+1, self.num_rep+1)

        for i in range(1, self.row_column_length+1):
            row_not_i = np.setdiff1d(row_indices, i)
            log_lhd_row[..., i - 1] = l_mvn_1_ordered[..., i - 1] + \
                                       np.sum(l_mvn_0_ordered[..., row_not_i - 1], axis=2)

        for j in range(self.row_column_length+1, self.num_rep+1):
            col_not_j = np.setdiff1d(col_indices, j)
            log_lhd_col[..., j - self.row_column_length - 1] = l_mvn_1_ordered[..., j - 1] + \
                                       np.sum(l_mvn_0_ordered[..., col_not_j - 1], axis=2)
        log_lhd_row_trn, log_lhd_row_test = np.split(log_lhd_row, [trn_repetition], axis=1)
        log_lhd_col_trn, log_lhd_col_test = np.split(log_lhd_col, [trn_repetition], axis=1)
        log_lhd_row_trn = np.cumsum(log_lhd_row_trn, axis=1)
        log_lhd_col_trn = np.cumsum(log_lhd_col_trn, axis=1)

        log_lhd_row_test = np.cumsum(log_lhd_row_test, axis=1)
        log_lhd_col_test = np.cumsum(log_lhd_col_test, axis=1)

        log_lhd_row_comb = np.concatenate([log_lhd_row_trn, log_lhd_row_test], axis=1)
        log_lhd_col_comb = np.concatenate([log_lhd_col_trn, log_lhd_col_test], axis=1)

        argmax_row_id = np.argmax(log_lhd_row_comb, axis=2)
        argmax_col_id = np.argmax(log_lhd_col_comb, axis=2)
        argmax_row_id += 1
        argmax_col_id += self.row_column_length+1
        argmax_row_col_id = np.stack([argmax_row_id, argmax_col_id], axis=2)
        letter_pred_matrix = np.zeros([self.num_letter, self.num_repetition]).astype('str')

        for letter_id in range(self.num_letter):
            for rep_id in range(self.num_repetition):
                letter_pred_matrix[letter_id, rep_id] = \
                    self.determine_letter(*argmax_row_col_id[letter_id, rep_id, :])

        return letter_pred_matrix

    # Need to modify this function by adding selected version
    def produce_lda_bayes_result_dict(
            self, eeg_signals_trun_all, eeg_code,
            delta_tar_mcmc, delta_ntar_mcmc,
            lambda_mcmc,
            gamma_mcmc, s_sq_mcmc, rho_mcmc, emp_pres_mat,
            phi_fn, trn_repetition, target_letters,
            channel_ids, soft_bool=True, q=1
    ):
        r"""
        args:
        -----
            eeg_signals_trun_all: array_like
                truncated eeg signals X_{v,i,j,e}, should have the dimension
                (letter_dim*num_repetition*self.num_rep, channel_dim, self.n_length, 1)
            eeg_code: 3d-array
                should have the input shape (letter_dim, num_repetition, num_rep)
            delta_tar_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim, n_length, 1)
            delta_ntar_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim, n_length, 1)
            lambda_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim)
            gamma_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim, n_length)
            s_sq_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim)
            rho_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim)
            emp_pres_mat: array_like,
                should have input shape (channel_dim, n_length, n_length)
            phi_fn: array_like
                should have the input shape (channel_dim, n_length, u)
            trn_repetition: integer
            burn_in: integer
            target_letters: list of characters, len(target_letters) = letter_dim
            channel_ids: a list of selected channel ids, indexing from 0 to num_electrode-1
            soft_bool: boolean variable, whether we remove the noisy feature completely when we make predictions
            q: autoregressive order

        return:
        -----
            A dict of prediction result including sampling number and
            probability of being predicted correctly
        """

        lda_accuracy = []
        mcmc_iter = gamma_mcmc.shape[0]
        print('The selected channels are {}'.format(channel_ids + 1))
        if emp_pres_mat is None:
            # fully bayesian framework
            eeg_signals_trun_all = eeg_signals_trun_all[:, channel_ids, ...]
            delta_tar_mcmc = delta_tar_mcmc[:, channel_ids, ...]
            delta_ntar_mcmc = delta_ntar_mcmc[:, channel_ids, ...]
            lambda_mcmc = lambda_mcmc[:, channel_ids]
            gamma_mcmc = gamma_mcmc[:, channel_ids, :]
            s_sq_mcmc = s_sq_mcmc[:, channel_ids]
            rho_mcmc = rho_mcmc[:, channel_ids, :]
            phi_fn = phi_fn[channel_ids, ...]

            for i in range(mcmc_iter):
                # print('i={}, kappa={}'.format(i, self.kappa))
                pres_mat_i = self.generate_ar1_pres_mat(
                    s_sq_mcmc[i, :], rho_mcmc[i, ...], channel_ids, q
                )
                pred_matrix_i = self.lda_two_step_estimation_mcmc_i(
                    eeg_signals_trun_all, eeg_code,
                    delta_tar_mcmc[i, ...], delta_ntar_mcmc[i, ...],
                    lambda_mcmc[i, :], pres_mat_i, gamma_mcmc[i, ...],
                    phi_fn, trn_repetition, channel_ids, soft_bool=soft_bool
                )
                lda_accuracy.append(pred_matrix_i)
        else:
            # fix precision matrix with sample covariance matrix
            eeg_signals_trun_all = eeg_signals_trun_all[:, channel_ids, ...]
            delta_tar_mcmc = delta_tar_mcmc[:, channel_ids, ...]
            delta_ntar_mcmc = delta_ntar_mcmc[:, channel_ids, ...]
            lambda_mcmc = lambda_mcmc[:, channel_ids]
            gamma_mcmc = gamma_mcmc[:, channel_ids, :]
            phi_fn = phi_fn[channel_ids, ...]

            for i in range(mcmc_iter):
                # print('i={}, kappa={}'.format(i, self.kappa))
                pred_matrix_i = self.lda_two_step_estimation_mcmc_i(
                    eeg_signals_trun_all, eeg_code,
                    delta_tar_mcmc[i, ...], delta_ntar_mcmc[i, ...],
                    lambda_mcmc[i, :], emp_pres_mat, gamma_mcmc[i, ...],
                    phi_fn, trn_repetition, channel_ids, soft_bool=soft_bool
                )
                lda_accuracy.append(pred_matrix_i)
        lda_accuracy = np.stack(lda_accuracy, axis=0)
        print('lda_accuracy has shape {}'.format(lda_accuracy.shape))
        lda_accuracy_dist = np.zeros([self.letter_dim, self.num_repetition, self.letter_table_sum])

        for i_dist, i_letter in enumerate(self.letter_table):
            lda_accuracy_dist[..., i_dist] = np.around(np.mean((lda_accuracy == i_letter) * 1, axis=0), decimals=4)

        lda_accuracy_mean = np.zeros([self.letter_dim, self.num_repetition])
        for i, letter in enumerate(target_letters):
            [row_i, col_i] = self.determine_row_column_indices(letter)
            lda_accuracy_mean[i, :] = lda_accuracy_dist[i, :, (row_i-1)*self.row_column_length+
                                                               col_i-self.row_column_length-1]
        lda_accuracy_max = np.around(np.max(lda_accuracy_dist, axis=-1), decimals=4)
        lda_accuracy_argmax = np.argmax(lda_accuracy_dist, axis=-1)
        lda_accuracy_letter_max = []
        for i in range(self.letter_dim):
            for j in range(self.num_repetition):
                lda_accuracy_letter_max.append(self.letter_table[lda_accuracy_argmax[i, j]])
        lda_accuracy_letter_max = np.reshape(np.stack(lda_accuracy_letter_max, axis=0),
                                             [self.letter_dim, self.num_repetition])
        lda_bayes_result_dict = {
            "sample_num": mcmc_iter,
            "mean": lda_accuracy_mean,
            "max": lda_accuracy_max,
            "letter_max": lda_accuracy_letter_max
            # "dist_mean": lda_accuracy_dist
        }
        return lda_bayes_result_dict

    def save_lda_bayes_results_single_test(
            self, new_lda_bayes_result, trn_repetition, sub_folder_name, target_letters
    ):
        file_dir = "{}/EEGBayesLDA/{}_lda_bayes_pred_select_trn_{}.csv" \
            .format(self.parent_path, sub_folder_name, trn_repetition)

        assert 1 <= self.trn_repetition <= self.num_repetition, print('wrong training repetition dim!')
        task = 'a'
        with open(file_dir, task) as f:
            f_writer = csv.writer(f)
            if task == "a":
                f_writer.writerow([' '])

            for i, letter_i in enumerate(target_letters):
                l_pred_correct = [letter_i + ', Correctly pred: ']
                l_pred_correct.extend(list(new_lda_bayes_result['mean'][i, :]))
                f_writer.writerow(l_pred_correct)

                l_pred_max = ['Max prob: ']
                l_pred_max.extend(list(new_lda_bayes_result['max'][i, :]))
                f_writer.writerow(l_pred_max)

                l_pred_arg_max = ['Max prob letter: ']
                l_pred_arg_max.extend(list(new_lda_bayes_result['letter_max'][i, :]))
                f_writer.writerow(l_pred_arg_max)

                f_writer.writerow([' '])

    def determine_selected_feature_matrix(self, gamma_mean, thres_level, channel_ids):

        r"""
        :param gamma_mean: array_like, with input dimension (num_electrode, n_length)
        :param thres_level: floating number between 0 and 1
        :return: selected feature matrix with the same size as gamma_mean.
        """
        ee = len(channel_ids)
        feature_mat = np.zeros([ee, self.n_length])
        feature_mat[gamma_mean >= thres_level] = 1

        return feature_mat

