import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.EEGPreFun import *
from scipy import stats, linalg
import csv
plt.style.use('ggplot')
import seaborn as sns
sns.set_context('notebook')


class XDAGibbs(EEGPreFun):

    r"""
    Implement basic frequenist method of LDA
    Implement Bayes LDA functions to BCI-speller with feature selection.
    Use cumulative log-likelihood over sequences as objective function.
    Produce credible intervals to determine automatic early stopping rule.
    """

    def __init__(self, sigma_sq_delta, u,
                 mu_1_delta, mu_0_delta,
                 a, b,  # weights
                 kappa,
                 letter_dim, trn_repetition,
                 *args,  **kwargs):
        super(XDAGibbs, self).__init__(*args, **kwargs)
        self.sigma_sq_delta = sigma_sq_delta
        self.u = u
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
        self.identity = np.eye(self.u)
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
        # print('train_x_mat_tar has shape {}'.format(train_x_mat_tar.shape))
        # print('train_x_mat_ntar has shape {}'.format(train_x_mat_ntar.shape))
        # print('signals_train_all_mean has shape {}'.format(signals_train_all_mean.shape))
        # print('signals_train_tar_mean has shape {}'.format(signals_train_tar_mean.shape))
        # print('signals_train_ntar_mean has shape {}'.format(signals_train_ntar_mean.shape))

        if std_bool:
            print('We standard raw truncated segments by signals_train_all_mean.')
            # data, original 4d-dimension
            signals_all -= signals_train_all_mean[np.newaxis, ...]
            # signals_train -= signals_train_all_mean[np.newaxis, ...]
            train_x_mat_tar -= signals_train_all_mean[np.newaxis, ...]
            train_x_mat_ntar -= signals_train_all_mean[np.newaxis, ...]
            # statistics, reduced dimension (no need add new dimension)
            train_x_tar_sum -= signals_train_all_mean
            train_x_ntar_sum -= signals_train_all_mean
            signals_train_tar_mean -= signals_train_all_mean
            signals_train_ntar_mean -= signals_train_all_mean

        return [
            signals_all,
            train_x_mat_tar, train_x_mat_ntar,
            signals_train_tar_mean, signals_train_ntar_mean,
            train_x_tar_sum, train_x_ntar_sum,
            train_x_tar_indices, train_x_ntar_indices,
            eeg_code_3d
        ]

    def create_initial_values_lda(self, s_sq_est):

        r"""
        args:
        -----
            s_sq_est: array_like, (1, num_electrode), preliminary estimates from raw signals
        return:
        -----
            A list of containing five arrays,
            delta_tar_mcmc, array_like, (1, num_electrode, u)
            delta_ntar_mcmc, array_like, (1, num_electrode, u)
            lambda_mcmc, array_like, (1, num_electrode)
            gamma_mcmc, array_like, (1, num_electrode, n_length)
            s_sq_mcmc, array_like, (1, num_electrode)
            rho_mcmc, array_like, (1, num_electrode)
        """
        delta_tar_mcmc = np.stack([stats.multivariate_normal(
            mean=self.mu_1_delta[i, :, 0],
            cov=self.sigma_sq_delta).rvs(1)[np.newaxis, :, np.newaxis]
                                 for i in range(self.num_electrode)], axis=1)

        delta_ntar_mcmc = np.stack([stats.multivariate_normal(
            mean=self.mu_0_delta[i, :, 0],
            cov=self.sigma_sq_delta).rvs(1)[np.newaxis, :, np.newaxis]
                                 for i in range(self.num_electrode)], axis=1)

        # lambda_tar_mcmc = np.ones([1, self.num_electrode])
        lambda_mcmc = np.ones([1, self.num_electrode])

        gamma_mcmc = np.random.binomial(1, 0.5, size=(1, self.num_electrode, self.n_length))

        rho_mcmc = np.random.uniform(low=0, high=1, size=(1, self.num_electrode))
        s_sq_mcmc = s_sq_est * np.ones([1, self.num_electrode])

        return delta_tar_mcmc, delta_ntar_mcmc, lambda_mcmc, gamma_mcmc, s_sq_mcmc, rho_mcmc

    def block_diagonal_mat(self, gamma_mat, channel_ids=None):
        r"""
        :param channel_ids: integer array_like
        :param gamma_mat: array_like, (num_electrode, n_length):

        :return:
            diagonal gamma matrix, array_like, (num_electrode, n_length, n_length):
        """
        if channel_ids is None:
            assert gamma_mat.shape == (self.num_electrode, self.n_length), \
                print('wrong channel dimension {}, should be {}'.format(gamma_mat.shape,
                                                                        (self.num_electrode, self.n_length)))
            gamma_mat = np.stack([np.diag(gamma_mat[i, :])
                                  for i in range(self.num_electrode)], axis=0)
        else:
            assert gamma_mat.shape == (len(channel_ids), self.n_length), \
                print('wrong channel dimension {}, should be {}'.format(gamma_mat.shape,
                                                                        (len(channel_ids), self.n_length)))
            gamma_mat = np.stack([np.diag(gamma_mat[i, :])
                                  for i in range(len(channel_ids))], axis=0)
        return gamma_mat

    def update_delta_tar_post_lda_2(
            self, delta_ntar, lambda_iter,
            gamma_mat, pres_mat,
            trun_x_tar_sum, trun_x_ntar_sum, phi_fn
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

        return:
        -----
            array_like
                should have dimension (num_electrode, u, 1)
        """

        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]
        gamma_mat = self.block_diagonal_mat(gamma_mat)
        idns = self.identity_n[np.newaxis, ...]
        phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat
        a1_I_gamma = self.a1 * (idns - gamma_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat
        a0_I_gamma = self.a0 * (idns - gamma_mat)

        lambda_mat_tar = self.identity / self.sigma_sq_delta + lambda_iter**2 * phi_fn_t @ \
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
            mean=np.zeros([self.u]),
            cov=1.0)
        delta_tar_post = std_mvn_rv.rvs(size=self.num_electrode)[..., np.newaxis]
        delta_tar_post = sigma_mat_tar_half @ delta_tar_post + post_mu_delta_tar

        return delta_tar_post

    def update_delta_ntar_post_lda_2(
            self, delta_tar, lambda_iter, gamma_mat, pres_mat,
            trun_x_tar_sum, trun_x_ntar_sum, phi_fn
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
        gamma_mat = self.block_diagonal_mat(gamma_mat)
        idns = self.identity_n[np.newaxis, ...]
        phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat
        a1_I_gamma = self.a1 * (idns - gamma_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat
        a0_I_gamma = self.a0 * (idns - gamma_mat)

        lambda_mat_ntar = self.identity / self.sigma_sq_delta + lambda_iter**2 * phi_fn_t @ \
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
            mean=np.zeros([self.u]),
            cov=1.0)
        delta_ntar_post = std_mvn_rv.rvs(size=self.num_electrode)[..., np.newaxis]
        delta_ntar_post = sigma_mat_ntar_half @ delta_ntar_post + post_mu_delta_ntar

        return delta_ntar_post

    def compute_trun_x_sum(self, trun_x_mat, trun_x_1_indices, trun_x_0_indices):

        r"""
        args:
        -----
            trun_x_mat: array_like
                truncated matrix X_{v,i,j,e}, should have the dimension of
                (total_beta_num, self.num_electrode, self.n_length, 1)
            trun_x_1_indices: array_like
                1d binary array containing indices of truncated segments associated with
                target stimuli
            trun_x_0_indices: array_like
                1d binary array containing indices of truncated segments associated with
                non-target stimuli

        Returns:
        -----
            a list with two elements. Each element is an array with dimension
            (self.num_electrode, self.n_length, 1)
        """
        total_batch_num = int(self.batch_dim_tar + self.batch_dim_ntar)
        assert trun_x_mat.shape == (total_batch_num,
                                    self.num_electrode,
                                    self.n_length, 1), \
            print('Wrong trun_x_mat shape {}, should be {}'
                  .format(trun_x_mat.shape, (total_batch_num,
                                             self.num_electrode,
                                             self.n_length, 1)))

        trun_x_mat_1 = trun_x_mat[trun_x_1_indices, ...]
        trun_x_mat_0 = trun_x_mat[trun_x_0_indices, ...]

        trun_x_mat_1 = np.sum(trun_x_mat_1, axis=0)
        trun_x_mat_0 = np.sum(trun_x_mat_0, axis=0)

        return trun_x_mat_1, trun_x_mat_0

    @staticmethod
    def compute_x_p(trun_x_sum, pres_mat):

        r"""
        trun_x_sum: array_like
        pres_mat: square matrix
        return:
            the array (sum X)^T @ precision_matrix,
            should have the dimension (16, 1, 25)
        """
        return np.matmul(np.transpose(trun_x_sum, [0, 2, 1]), pres_mat)

    @staticmethod
    def compute_ising_log_prior(
            gamma_mat_pre, tau, beta_ising, gamma_neighbor
    ):
        r"""
        args:
        -----
            gamma_mat_pre: array_like, should have shape (num_electrode, n_length)
            tau: integer, [0, n_length-1]
            beta_ising: beta hyper-parameter
            gamma_neighbor: integer, the neighborhood region range

        return:
        -----
            array_like, the ising log prior value, should have shape (num_electrode,)

        note:
        -----
            beta tau sum_{tau': tau'~tau} tau'
            Notice that here we need to shift from 0-1 coding to -1,1 coding scheme.
        """

        gamma_neighbor_mat = gamma_mat_pre[:, max(tau - gamma_neighbor, 0):(tau + gamma_neighbor + 1)]
        # print(gamma_neighbor_mat)
        z_neighbor = -np.ones_like(gamma_neighbor_mat)
        z_neighbor[gamma_neighbor_mat == gamma_mat_pre[:, tau, np.newaxis]] = 1
        z_neighbor = np.sum(z_neighbor, axis=-1) - 1  # exclude w.r.t itself (which is always 1)

        return beta_ising * z_neighbor

    def update_gamma_post_2(
            self, gamma_mat_pre, delta_tar, delta_ntar,
            lambda_iter, pres_mat_pre,
            tau, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, beta_ising, gamma_neighbor
    ):
        r"""
        args:
        -----
            gamma_mat_pre: array_like
                probability of selection indicators of all electrodes from previous iteration,
                should have dimension of (self.num_electrode, self.n_length)
            delta_tar: array_like,
                should have dimension (num_electrode, u, 1)
            delta_ntar: array_like,
                should have dimension (num_electrode, u, 1)
            lambda_iter: array_like, should have dimension (num_electrode,)
            pres_mat_pre: square matrix,
                should have dimension (num_electrode, n_length, n_length)
            tau: integer
                index of latency, from 0 to n-1
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

        # Construct proposed state (binary, so we can enumerate them)
        gamma_mat_post_channel_tau_0 = np.copy(gamma_mat_pre)
        gamma_mat_post_channel_tau_0[:, tau] = 0
        gamma_mat_post_channel_tau_1 = np.copy(gamma_mat_pre)
        gamma_mat_post_channel_tau_1[:, tau] = 1

        # Compute the ising prior on the log scale
        ising_select_log = self.compute_ising_log_prior(
            gamma_mat_post_channel_tau_1, tau, beta_ising, gamma_neighbor
        )
        ising_nselect_log = self.compute_ising_log_prior(
            gamma_mat_post_channel_tau_0, tau, beta_ising, gamma_neighbor
        )
        # target, select tau-th feature across channels
        quad_select = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat_post_channel_tau_1, pres_mat_pre,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn)
        # print('quad_select has shape {}'.format(quad_select.shape))
        # target, not select tau-th feature across channels
        quad_nselect = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat_post_channel_tau_0, pres_mat_pre,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn)
        # print('quad_nselect has shape {}'.format(quad_nselect.shape))

        quad_log_odds = quad_select + ising_select_log - quad_nselect - ising_nselect_log
        quad_prop = np.zeros_like(quad_log_odds)
        for e in range(self.num_electrode):
            # Avoid np.exp() overflow
            if quad_log_odds[e] >= 100:
                quad_prop[e] = 1
            else:
                quad_prop[e] = np.exp(quad_log_odds[e]) / (1 + np.exp(quad_log_odds[e]))
        # quad_prop[np.isnan(quad_prop)] = 1.0

        gamma_mat_post = np.copy(gamma_mat_pre)
        quad_ind = [np.random.binomial(1, quad_prop[i], 1)[0]
                    for i in range(self.num_electrode)]
        gamma_mat_post[:, tau] = np.array(quad_ind)

        return gamma_mat_post

    # def compute_quad_sum_2(
    #         self, delta_tar, delta_ntar,
    #         lambda_iter,
    #         pres_mat, gamma_mat_pre,
    #         xp_tar, xp_ntar, phi_fn
    # ):
    #     r"""
    #     args:
    #     -----
    #         delta_tar: array_like
    #         delta_ntar: array_like
    #         lambda_iter: array_like, (num_electrode,)
    #         pres_mat: array_like, (num_electrode, n_length, n_length)
    #         gamma_mat_pre: array_like, (num_electrode, n_length)
    #         xp_tar: array_like
    #             xp_tar = (sum X_tar)^T @ pres_mat with shape (channel_dim, 1, n_length)
    #         xp_ntar: array_like
    #             xp_ntar = (sum X_ntar)^T @ pres_mat with shape (channel_dim, 1, n_length)
    #         phi_fn: array_like
    #             (num_electrode, n_length, u) smoothing kernel matrix
    #
    #     return:
    #     -----
    #         array_like, expect to have shape (num_electrode, 1, 1)
    #
    #         (sum X_tar)^T @ pres_mat @ mean_tar - 1/2 * batch_dim_tar * mean_tar^T @ pres_mat @ mean_tar +
    #         (sum X_ntar)^T @ pres_mat @ mean_ntar - 1/2 * batch_dim_ntar * mean_ntar^T @ pres_mat @ mean_ntar
    #
    #     notes:
    #     -----
    #         for target signals:
    #         mean_tar = 1/2 * (I + Gamma_e) @ phi_fn @ (lambda_tar * delta_tar)
    #         + 1/2 * (I - Gamma_e) @ phi_fn @ (lambda_ntar * delta_ntar)
    #         for non-target signals:
    #         mean_ntar = 1/2 * (I + Gamma_e) @ phi_fn @ (lambda_ntar * delta_ntar)
    #         + 1/2 * (I - Gamma_e) @ phi_fn @ (lambda_tar * delta_tar)
    #
    #         In general, for target:
    #         mean_tar = (a1 * I + a0 * Gamma_e) @ phi_fn @ (lambda_iter * delta_tar)
    #         + a0 (I - Gamma_e) @ phi_fn @ (lambda_iter * delta_ntar)
    #         for non-target:
    #         mean_ntar = (a0 * I + a1 * Gamma_e) @ phi_fn @ (lambda_iter @ delta_ntar)
    #         + a1 (I - Gamma_e) @ phi_fn @ (lambda_iter * delta_tar)
    #     """
    #
    #     lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]
    #     gamma_mat_pre = self.block_diagonal_mat(gamma_mat_pre)
    #     idns = self.identity_n[np.newaxis, ...]
    #     # phi_fn_t = np.transpose(phi_fn, [0, 2, 1])
    #
    #     a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat_pre
    #     a1_I_gamma = self.a1 * (idns - gamma_mat_pre)
    #     a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat_pre
    #     a0_I_gamma = self.a0 * (idns - gamma_mat_pre)
    #
    #     mean_tar = lambda_iter * (a1_I_a0_gamma @ phi_fn @ delta_tar + a0_I_gamma @ phi_fn @ delta_ntar)
    #     mean_ntar = lambda_iter * (a0_I_a1_gamma @ phi_fn @ delta_ntar + a1_I_gamma @ phi_fn @ delta_tar)
    #     mean_tar_t = np.transpose(mean_tar, [0, 2, 1])
    #     mean_ntar_t = np.transpose(mean_ntar, [0, 2, 1])
    #
    #     q_tar_1 = xp_tar @ mean_tar
    #     q_tar_2 = self.batch_dim_tar * mean_tar_t @ pres_mat @ mean_tar
    #
    #     q_ntar_1 = xp_ntar @ mean_ntar
    #     q_ntar_2 = self.batch_dim_ntar * mean_ntar_t @ pres_mat @ mean_ntar
    #
    #     return q_tar_1 - 1/2 * q_tar_2 + q_ntar_1 - 1/2 * q_ntar_2

    # def compute_outer_prod_sum_2(
    #         self, delta_tar, delta_ntar, lambda_tar, lambda_ntar,
    #         gamma_mat, trun_x_mat, trun_x_tar_indices, trun_x_ntar_indices, phi_fn
    # ):
    #     r"""
    #     args:
    #     -----
    #         delta_tar: array_like
    #             The mean vector for target stimuli across 16 electrodes
    #             should have the dimension of (self.feature_length, 1)
    #         delta_ntar: array_like
    #             The mean vector for non-target stimuli across 16 electrodes
    #             should have the same dimension as delta_1
    #         lambda_tar: array_like
    #             should have (num_electrode,)
    #         lambda_ntar: array_like
    #             should have (num_electrode,)
    #         gamma_mat: array_like
    #             selection indicator matrix,
    #             should have the dimension (self.num_electrode, self.n_length)
    #         trun_x_mat: array_like
    #             The entire truncated segments X_{v,i,j,e}, should have the dimension of
    #             (letter_dim * repet_dim * self.num_rep, self.num_electrode, self.n_length, 1)
    #         trun_x_tar_indices: array_like
    #             1d binary array corresponding to all target stimuli
    #         trun_x_ntar_indices: array_like
    #             1d binary array corresponding to all non-target stimuli
    #         phi_fn: array_like
    #             (num_electrode, n_length, u) smoothing kernel matrix
    #
    #     Returns:
    #     -----
    #     A list with two arrays,
    #         1st: summation of outer product of
    #             X_tar - gamma_plus_half_phi @ delta_tar - gamma_minus_half_phi @ delta_ntar,
    #         2nd: summation of outer product of
    #             X_ntar - gamma_plus_half_phi @ delta_ntar - gamma_minus_half_phi @ delta_tar
    #     Each array should have the dimension (channel_dim, n_length, n_length).
    #     """
    #     lambda_tar = lambda_tar[:, np.newaxis, np.newaxis]
    #     lambda_ntar = lambda_ntar[:, np.newaxis, np.newaxis]
    #
    #     gamma_mat = self.block_diagonal_mat(gamma_mat)
    #     trun_x_mat_tar = trun_x_mat[trun_x_tar_indices, ...]
    #     trun_x_mat_ntar = trun_x_mat[trun_x_ntar_indices, ...]
    #     '''
    #     gamma_plus_half_phi = 1/2 * (np.eye(self.n_length) + gamma_mat) @ phi_fn
    #     gamma_minus_half_phi = 1/2 * (np.eye(self.n_length) - gamma_mat) @ phi_fn
    #     '''
    #     gamma_plus_half_phi = (self.a1 * np.eye(self.n_length)[np.newaxis, ...] + self.a0 * gamma_mat) @ phi_fn
    #     gamma_minus_half_phi = self.a0 * (np.eye(self.n_length)[np.newaxis, ...] - gamma_mat) @ phi_fn
    #
    #     mean_tar = gamma_plus_half_phi @ (lambda_tar * delta_tar) + gamma_minus_half_phi @ (lambda_tar * delta_ntar)
    #     mean_ntar = gamma_plus_half_phi @ (lambda_ntar * delta_ntar) + gamma_minus_half_phi @ (lambda_tar * delta_tar)
    #
    #     trun_x_mat_tar_diff = trun_x_mat_tar - mean_tar[np.newaxis, ...]
    #     trun_x_mat_ntar_diff = trun_x_mat_ntar - mean_ntar[np.newaxis, ...]
    #
    #     out_prod_tar = np.sum(trun_x_mat_tar_diff @ np.transpose(trun_x_mat_tar_diff, [0, 1, 3, 2]), axis=0)
    #     out_prod_ntar = np.sum(trun_x_mat_ntar_diff @ np.transpose(trun_x_mat_ntar_diff, [0, 1, 3, 2]), axis=0)
    #
    #     return [out_prod_tar, out_prod_ntar]

    def generate_proposal_ar1_pres_mat(self, sigma_sq, rho, channel_ids=None):
        r"""
        args:
        -----
            sigma_sq: array_like, should have dimension (channel_dim,)
            rho: array_like, the auto-correlation, assumed between 0 and 1, should have dimension (channel_dim,)
            channel_ids: integer array_like
        return:
        -----
            array_like
            channel_dim batch precision matrix

        note:
        -----
            For AR(1) with sigma_sq and rho,
            Kac-Murdock-Szego matrix provides an analytical solution to the inverse:
            Given Corr = AR(1), rho = q, sigma_sq
            Corr_inv = 1/sigma_sq/(1-rho**2) * tri-diagonal matrix, where
            main diagonal = (1, 1+rho**2, ..., 1+rho**2, 1)
            +1/-1 off-diagonal = -rho
            The final precision matrix, P = U**(-1/2) @ Corr_inv @ U**(-1/2) if we have heterogeneous sigma_sq.
            https://mathoverflow.net/questions/65795/inverse-of-an-ar1-or-laplacian-or-kac-murdock-szegö-matrix/65819
        """
        if channel_ids is None:
            ee = self.num_electrode
        else:
            ee = len(channel_ids)
        tri_diag_mat = np.zeros(shape=[ee, self.n_length, self.n_length])
        for e in range(ee):
            first_col_e = np.zeros([self.n_length])
            first_col_e[1] = -rho[e]
            tri_diag_mat[e, ...] = self.create_toeplitz_cov_mat(sigma_sq=1.0, first_column=first_col_e)
            vary_diag_vec = np.zeros([self.n_length]) + 1 + rho[e] ** 2
            vary_diag_vec[0] = 1
            vary_diag_vec[-1] = 1
            tri_diag_mat[e, ...] += np.diag(vary_diag_vec)
            tri_diag_mat[e, ...] *= 1 / (sigma_sq[e] * (1 - rho[e] ** 2))

        '''
        # heterogeneous diagonal sigma_sq, so each part should be sqrt(sigma_sq) = sigma
        # sigma = 1/np.sqrt(sigma_sq)
        # sigma_mat = self.block_diagonal_mat(sigma)
        # tri_diag_mat = sigma_mat @ tri_diag_mat @ sigma_mat
        '''

        return tri_diag_mat

    def generate_proposal_lambda_state(self, lambda_old, zeta_lambda):
        r"""
        args:
        -----
            lambda_old: array_like, (num_electrode,)
            zeta_lambda: array_like, (num_electrode,)

        return:
        -----
            proposed state of lambda, should have the same dimension as lambda_old

        note:
        -----
            proposal distribution to generate kernel variance parameter
        """
        lambda_new = stats.multivariate_normal(
            mean=lambda_old, cov=zeta_lambda*np.eye(self.num_electrode)
        ).rvs(1)
        if self.num_electrode == 1:
            lambda_new = np.array([lambda_new])
        for e in range(self.num_electrode):
            if lambda_new[e] <= 0:
                lambda_new[e] = np.copy(lambda_old[e])

        return lambda_new

    def generate_proposal_s_sq_state(self, s_sq_old, zeta_s):
        r"""
        args:
        -----
            s_sq_old: array_like, previous state of s_sq, (num_electrode,)
            zeta_s: array_like, step size of random walk, (num_electrode,)

        return:
        -----
            s_sq_new, only update tau-index

        note:
        -----
            I change the s_sq to sigma_sq for MCMC estimation purpose.
        """
        s_sq_new = stats.multivariate_normal(
            mean=s_sq_old, cov=zeta_s*np.eye(self.num_electrode)
        ).rvs(1)
        if self.num_electrode == 1:
            s_sq_new = np.array([s_sq_new])
        for e in range(self.num_electrode):
            if s_sq_new[e] <= 0:
                s_sq_new[e] = np.copy(s_sq_old[e])

        return s_sq_new

    def generate_proposal_rho_state(self, rho_old, zeta_rho):
        r"""
        args:
        -----
            rho_old: array_like, previous state of rho, (num_electrode,)
            zeta_rho: arrya_like, step size of random walk, (num_electrode,)

        return:
        -----
            A new array of the same size as rho_old,

        note:
        -----
            Need to check the constraint before final return
        """
        rho_new = stats.multivariate_normal(
            mean=rho_old, cov=zeta_rho*np.eye(self.num_electrode)
        ).rvs(1)
        if self.num_electrode == 1:
            rho_new = np.array([rho_new])

        for e in range(self.num_electrode):
            if rho_new[e] <= 0 or rho_new[e] >= 1:
                rho_new[e] = rho_old[e]

        return rho_new

    def compute_log_prior_ratio_s_sq(self, s_sq_old, s_sq_new, alpha_s, beta_s):
        r"""
        args:
        -----
            s_sq_old: array_like, previous state of s_sq, (num_electrode,)
            s_sq_new: array_like, proposed state of s_sq, (num_electrode,)
            tau: integer, 0 to n_length
            alpha_s: scalar value > 0
            beta_s: scalar value > 0 (rate = 1/scale)

        return:
        -----
            array_like value, (num_electrode,)
        note:
        -----
            we assume s_sq ~ InvGamma(alpha_s, beta_s), rho ~ Uniform(0, 1)
            Not to be confused, I always use alpha, beta parametrization
            where beta_s is the inverse of scale.
        """
        assert s_sq_new.shape == s_sq_old.shape == (self.num_electrode,)

        s_sq_rv = stats.invgamma(a=alpha_s)
        s_sq_old_log_pdf = s_sq_rv.logpdf(s_sq_old * beta_s)
        s_sq_new_log_pdf = s_sq_rv.logpdf(s_sq_new * beta_s)

        return s_sq_new_log_pdf - s_sq_old_log_pdf

    def compute_log_prior_ratio_lambda(self, lambda_old, lambda_new, alpha_s, beta_s):
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
        assert lambda_new.shape == lambda_old.shape == (self.num_electrode,)

        lambda_rv = stats.gamma(a=alpha_s)
        lambda_old_log_pdf = lambda_rv.logpdf(lambda_old * beta_s)
        lambda_new_log_pdf = lambda_rv.logpdf(lambda_new * beta_s)

        return lambda_new_log_pdf - lambda_old_log_pdf

    def compute_log_prior_ratio_rho(self, rho_old, rho_new, a=1, b=1):
        r"""
        args:
        -----
            rho_old: array_like, proposed state of rho, (num_electrode,)
            rho_new: array_like, proposed state of rho, (num_electrode,)

        return:
        -----
             log pi(rho_new) - log pi(rho_old) with dimension (num_electrode,)

        note:
        -----
            I assume rho ~ Uniform(0, 1) (or Beta(1, 1)
        """
        assert rho_new.shape == rho_old.shape == (self.num_electrode,)

        rho_rv = stats.beta(a=a, b=b)
        rho_old_log_pdf = rho_rv.logpdf(rho_old)
        rho_new_log_pdf = rho_rv.logpdf(rho_new)

        return rho_new_log_pdf - rho_old_log_pdf

    def compute_sampling_log_lhd(
            self, delta_tar, delta_ntar,
            lambda_iter, gamma_mat, pres_mat,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn
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

        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]
        gamma_mat = self.block_diagonal_mat(gamma_mat)
        idns = self.identity_n[np.newaxis, ...]
        # phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_mat
        a1_I_gamma = self.a1 * (idns - gamma_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_mat
        a0_I_gamma = self.a0 * (idns - gamma_mat)

        mean_tar = lambda_iter * (a1_I_a0_gamma @ phi_fn @ delta_tar + a0_I_gamma @ phi_fn @ delta_ntar)
        mean_ntar = lambda_iter * (a0_I_a1_gamma @ phi_fn @ delta_ntar + a1_I_gamma @ phi_fn @ delta_tar)
        # print('mean_tar = {}'.format(mean_tar))
        # print('mean_ntar = {}'.format(mean_ntar))

        trun_x_mat_tar_diff = trun_x_mat_tar - mean_tar[np.newaxis, ...]
        trun_x_mat_ntar_diff = trun_x_mat_ntar - mean_ntar[np.newaxis, ...]
        trun_x_mat_tar_diff_t = np.transpose(trun_x_mat_tar_diff, [0, 1, 3, 2])
        trun_x_mat_ntar_diff_t = np.transpose(trun_x_mat_ntar_diff, [0, 1, 3, 2])
        # print(trun_x_mat_tar_diff.shape)
        # print(trun_x_mat_tar_diff_t.shape)

        # quadtraic part:
        log_quad_sum = -1/2 * (np.sum(trun_x_mat_tar_diff_t @ pres_mat[np.newaxis, ...] @ trun_x_mat_tar_diff, axis=0) +
                               np.sum(trun_x_mat_ntar_diff_t @ pres_mat[np.newaxis, ...] @ trun_x_mat_ntar_diff, axis=0))
        log_quad_sum = np.squeeze(log_quad_sum, axis=(-2, -1))
        [sgn, logdet_abs] = np.linalg.slogdet(pres_mat)
        log_pres_det = 1/2 * sgn * logdet_abs * self.total_batch_train

        # y_tar_sum = np.sum([stats.multivariate_normal(
        #     mean=mean_tar[0, :, 0],
        #     cov=np.linalg.inv(pres_mat[0, ...])).logpdf(x=trun_x_mat_tar[i, 0, :, 0])
        #                     for i in range(self.batch_dim_tar)], axis=0)
        # y_ntar_sum = np.sum([stats.multivariate_normal(
        #     mean=mean_ntar[0, :, 0],
        #     cov=np.linalg.inv(pres_mat[0, ...])).logpdf(x=trun_x_mat_ntar[i, 0, :, 0])
        #                      for i in range(self.batch_dim_ntar)], axis=0)
        # # print('y_tar_sum + y_ntar_sum = {}'.format(y_tar_sum + y_ntar_sum))

        return log_quad_sum + log_pres_det

    def update_lambda_iter_post_mh(
            self, delta_tar, delta_ntar,
            lambda_old,
            gamma_mat, s_sq, rho,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
            alpha_s, beta_s, zeta_lambda_tar
    ):
        r"""
           args:
           -----
               delta_tar: array_like, (num_electrode, u, 1)
               delta_ntar: array_like, (num_electrode, u, 1)
               lambda_old: array_like, (num_electrode,)
               gamma_mat: array_like, (num_electrode, n_length)
               s_sq: array_like, previous state of sigma_sq (covariance), (num_electrode,)
               rho_old: array_like, previous state of rho, (num_electrode,)
               trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
               trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
               phi_fn: array_like, (num_electrode, n_length, u)
               alpha_s: hyper-parameter
               beta_s: hyper-parameter
               zeta_lambda_tar: step size

           return:
           -----
               A list of two arrays including lambda_tar_post (num_electrode,), and
               acceptance indicator with shape (num_electrode,)
        """
        # Generate new state
        lambda_new = self.generate_proposal_lambda_state(lambda_old, zeta_lambda_tar)
        log_prior_ratio = self.compute_log_prior_ratio_s_sq(lambda_old, lambda_new, alpha_s, beta_s)

        # Generate pres_mat_old and pres_mat_new (only change tau-index of lambda_tar_old)
        pres_mat = self.generate_proposal_ar1_pres_mat(s_sq, rho)

        log_sampling_old = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_old,
            gamma_mat, pres_mat, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
        )

        log_sampling_new = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_new,
            gamma_mat, pres_mat, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
        )
        log_sampling_ratio = log_sampling_new - log_sampling_old

        log_alpha_mh = log_prior_ratio + log_sampling_ratio  # log_proposal_ratio = 0
        log_uniform = np.log(np.random.uniform(low=0, high=1, size=self.num_electrode))

        lambda_post = np.copy(lambda_old)
        # Compute acceptance rate and help adjust step size
        lambda_accept = np.zeros_like(lambda_old)

        for e in range(self.num_electrode):
            if log_alpha_mh[e] > 0:
                log_alpha_mh[e] = 0

            if log_alpha_mh[e] >= log_uniform[e]:
                lambda_post[e] = np.copy(lambda_new[e])
                lambda_accept[e] = 1

        return [lambda_post, lambda_accept]

    # def update_lambda_ntar_post_mh(
    #         self, delta_tar, delta_ntar,
    #         lambda_tar, lambda_ntar_old,
    #         gamma_mat, s_sq, rho,
    #         trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
    #         alpha_s, beta_s, zeta_lambda_ntar
    # ):
    #     r"""
    #        args:
    #        -----
    #            delta_tar: array_like, (num_electrode, u, 1)
    #            delta_ntar: array_like, (num_electrode, u, 1)
    #            lambda_tar: array_like, (num_electrode,)
    #            lambda_ntar_old: array_like, (num_electrode,)
    #            gamma_mat: array_like, (num_electrode, n_length)
    #            s_sq: array_like, previous state of sigma_sq, (num_electrode,)
    #            rho_old: array_like, previous state of rho, (num_electrode,)
    #            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
    #            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
    #            phi_fn: array_like, (num_electrode, n_length, u)
    #            alpha_s: hyper-parameter
    #            beta_s: hyper-parameter
    #            zeta_lambda_ntar: step size
    #
    #        return:
    #        -----
    #            A list of two arrays including lambda_tar_post (num_electrode,), and
    #            acceptance indicator with shape (num_electrode,)
    #     """
    #     # Generate new state
    #     lambda_ntar_new = self.generate_proposal_lambda_state(lambda_ntar_old, zeta_lambda_ntar)
    #     log_prior_ratio = self.compute_log_prior_ratio_s_sq(lambda_ntar_old, lambda_ntar_new, alpha_s, beta_s)
    #
    #     # Generate pres_mat_old and pres_mat_new (only change tau-index of lambda_tar_old)
    #     pres_mat = self.generate_proposal_ar1_pres_mat(s_sq, rho)
    #
    #     log_sampling_old = self.compute_sampling_log_lhd(
    #         delta_tar, delta_ntar, lambda_tar, lambda_ntar_old,
    #         gamma_mat, pres_mat, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
    #     )
    #
    #     log_sampling_new = self.compute_sampling_log_lhd(
    #         delta_tar, delta_ntar, lambda_tar, lambda_ntar_new,
    #         gamma_mat, pres_mat, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
    #     )
    #     log_sampling_ratio = log_sampling_new - log_sampling_old
    #
    #     log_alpha_mh = log_prior_ratio + log_sampling_ratio  # log_proposal_ratio = 0
    #     log_uniform = np.log(np.random.uniform(low=0, high=1, size=self.num_electrode))
    #
    #     lambda_ntar_post = np.copy(lambda_ntar_old)
    #     # Compute acceptance rate and help adjust step size
    #     lambda_ntar_accept = np.zeros_like(lambda_ntar_old)
    #
    #     for e in range(self.num_electrode):
    #         if log_alpha_mh[e] > 0:
    #             log_alpha_mh[e] = 0
    #
    #         if log_alpha_mh[e] >= log_uniform[e]:
    #             lambda_ntar_post[e] = np.copy(lambda_ntar_new[e])
    #             lambda_ntar_accept[e] = 1
    #
    #     return [lambda_ntar_post, lambda_ntar_accept]

    def update_s_sq_post_mh(
            self, delta_tar, delta_ntar,
            lambda_iter,
            gamma_mat,
            s_sq_old, rho_old,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
            alpha_s, beta_s, zeta_s
    ):
        r"""
        args:
        -----
            delta_tar: array_like, (num_electrode, u, 1)
            delta_ntar: array_like, (num_electrode, u, 1)
            lambda_iter: array_like, (num_electrode,)
            gamma_mat: array_like, (num_electrode, n_length)
            s_sq_old: array_like, previous state of sigma_sq, (num_electrode,)
            rho_old: array_like, previous state of rho, (num_electrode,)
            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
            phi_fn: array_like, (num_electrode, n_length, u)
            alpha_s: hyper-parameter
            beta_s: hyper-parameter
            zeta_s: step size

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

        # Generate proposed state
        s_sq_new = self.generate_proposal_s_sq_state(s_sq_old, zeta_s)
        log_prior_ratio = self.compute_log_prior_ratio_s_sq(s_sq_old, s_sq_new, alpha_s, beta_s)

        # Generate pres_mat_old and pres_mat_new (only change tau-index of s_sq_old)
        pres_mat_old = self.generate_proposal_ar1_pres_mat(s_sq_old, rho_old)
        pres_mat_new = self.generate_proposal_ar1_pres_mat(s_sq_new, rho_old)

        log_sampling_old = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_old, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
        )

        log_sampling_new = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_new, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
        )
        log_sampling_ratio = log_sampling_new - log_sampling_old

        log_alpha_mh = log_prior_ratio + log_sampling_ratio  # + log_proposal_ratio = 0
        log_uniform = np.log(np.random.uniform(low=0, high=1, size=self.num_electrode))

        s_sq_post = np.copy(s_sq_old)
        # Compute acceptance rate and help adjust step size
        s_sq_accept = np.zeros_like(s_sq_old)

        for e in range(self.num_electrode):
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
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
            zeta_rho
    ):
        r"""
        args:
        -----
            delta_tar: array_like, (num_electrode, u, 1)
            delta_ntar: array_like, (num_electrode, u, 1)
            lambda_iter: array_like, (num_electrode,)
            gamma_mat: array_like, (num_electrode, n_length)
            s_sq: array_like, iter state of sigma_sq, (num_electrode,)
            rho_old: array_like, previous state of rho, (num_electrode,)
            trun_x_mat_tar: array_like, (batch_dim_tar, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim_ntar, num_electrode, n_length, 1)
            phi_fn: array_like, (num_electrode, n_length, u)
            zeta_rho: step size, (num_electrode,)

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
        # Generate proposed state
        rho_new = self.generate_proposal_rho_state(rho_old, zeta_rho)
        log_prior_ratio = self.compute_log_prior_ratio_rho(rho_old, rho_new)

        # Generate pres_mat_old and pres_mat_new (only change rho)
        pres_mat_old = self.generate_proposal_ar1_pres_mat(s_sq, rho_old)
        pres_mat_new = self.generate_proposal_ar1_pres_mat(s_sq, rho_old)

        log_sampling_old = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_old, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
        )
        log_sampling_new = self.compute_sampling_log_lhd(
            delta_tar, delta_ntar, lambda_iter,
            gamma_mat, pres_mat_new, trun_x_mat_tar, trun_x_mat_ntar, phi_fn
        )
        log_sampling_ratio = log_sampling_new - log_sampling_old

        log_alpha_mh = log_prior_ratio + log_sampling_ratio  # log_proposal_ratio = 0
        log_uniform = np.log(np.random.uniform(low=0, high=1, size=self.num_electrode))

        rho_post = np.copy(rho_old)
        # Compute acceptance rate and help adjust step size
        rho_accept = np.zeros_like(rho_old)

        for e in range(self.num_electrode):
            if log_alpha_mh[e] > 0:
                log_alpha_mh[e] = 0

            if log_alpha_mh[e] >= log_uniform[e]:
                rho_post[e] = np.copy(rho_new[e])
                if rho_new[e] != rho_old[e]:
                    rho_accept[e] = 1

        return [rho_post, rho_accept]

    def gibbs_single_iteration(
            self, delta_ntar_old, lambda_old,
            gamma_mat_iter, s_sq_old, rho_old,
            trun_x_tar_sum_train, trun_x_ntar_sum_train,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
            alpha_s, beta_s, zeta_lambda,
            zeta_s, zeta_rho, beta_ising, gamma_neighbor
    ):
        r"""
        args:
        -----
            delta_ntar_old: array_like, (num_electrode, u, 1)
            lambda_old: array_like, (num_electrode,)
            gamma_mat_iter: array_like, (num_electrode, n_length)
            s_sq_old: array_like, (num_electrode,)
            rho_old: array_like, (num_electrode,)
            trun_x_tar_sum_train: array_like, (num_electrode, n_length, 1)
            trun_x_ntar_sum_train: array_like, (num_electrode, n_length, 1)
            trun_x_mat_tar: array_like, (batch_dim, num_electrode, n_length, 1)
            trun_x_mat_ntar: array_like, (batch_dim, num_electrode, n_length, 1)
            phi_fn: array_like, (num_electrode, n, u)
            alpha_s, beta_s: floating numbers
            zeta_lambda: array_like, (num_electrode,)
            zeta_s: array_like, (num_electrode,)
            zeta_rho: array_like, (num_electrode,)
            beta_ising: scale number, hyper-parameter for ising prior
            gamma_neighbor: integer, threshold for ~ relationship.

        return:
        -----
            A list of all parameters updated within one iteration of Gibbs sampler, including
            delta_tar_post,
            delta_ntar_post,
            lambda_post,
            gamma_mat_iter,
            s_sq_post,
            rho_post,
            s_sq_accept,
            rho_accept.
        """
        pres_mat_old = self.generate_proposal_ar1_pres_mat(s_sq_old, rho_old)
        # First update delta_tar, delta_ntar
        delta_tar_post = self.update_delta_tar_post_lda_2(
            delta_ntar_old, lambda_old,
            gamma_mat_iter, pres_mat_old,
            trun_x_tar_sum_train, trun_x_ntar_sum_train, phi_fn
        )
        delta_ntar_post = self.update_delta_ntar_post_lda_2(
            delta_tar_post, lambda_old,
            gamma_mat_iter, pres_mat_old,
            trun_x_tar_sum_train, trun_x_ntar_sum_train, phi_fn
        )
        # kernel variance lambda_e
        [lambda_post, lambda_accept] = self.update_lambda_iter_post_mh(
            delta_tar_post, delta_ntar_post, lambda_old,
            gamma_mat_iter, s_sq_old, rho_old, trun_x_mat_tar, trun_x_mat_ntar,
            phi_fn, alpha_s, beta_s, zeta_lambda
        )
        # [lambda_ntar_post, lambda_ntar_accept] = self.update_lambda_ntar_post_mh(
        #     delta_tar_post, delta_ntar_post, lambda_tar_post, lambda_ntar_old,
        #     gamma_mat_iter, s_sq_old, rho_old, trun_x_mat_tar, trun_x_mat_ntar,
        #     phi_fn, alpha_s, beta_s, zeta_lambda_ntar
        # )

        # xp_tar_train = self.compute_x_p(trun_x_tar_sum_train, pres_mat_old)
        # xp_ntar_train = self.compute_x_p(trun_x_ntar_sum_train, pres_mat_old)

        # Update selection indicator gamma_matrix
        for tau in range(self.n_length):
            gamma_mat_iter = self.update_gamma_post_2(
                gamma_mat_iter, delta_tar_post, delta_ntar_post,
                lambda_post, pres_mat_old,
                tau, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, beta_ising, gamma_neighbor
            )
        # Update s_sq
        [s_sq_post, s_sq_accept] = self.update_s_sq_post_mh(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            s_sq_old, rho_old,
            trun_x_mat_tar, trun_x_mat_ntar, phi_fn,
            alpha_s, beta_s, zeta_s
        )
        # Update rho
        [rho_post, rho_accept] = self.update_rho_post_mh(
            delta_tar_post, delta_ntar_post,
            lambda_post, gamma_mat_iter,
            s_sq_post, rho_old, trun_x_mat_tar, trun_x_mat_ntar, phi_fn, zeta_rho
        )

        return [delta_tar_post, delta_ntar_post,
                lambda_post, gamma_mat_iter,
                s_sq_post, rho_post,
                lambda_accept, s_sq_accept, rho_accept]

    def save_bayes_lda_mcmc(
            self, delta_tar_mcmc, delta_ntar_mcmc, lambda_mcmc, gamma_mcmc, s_sq_mcmc, rho_mcmc, log_lhd_mcmc
    ):
        sio.savemat('{}/{}/{}/{}_lda_mcmc_trn_{}.mat'.
                    format(self.parent_path,
                           self.data_type,
                           self.sub_folder_name[:4],
                           self.sub_folder_name,
                           self.trn_repetition),
                    {
                        'delta_tar': delta_tar_mcmc,
                        'delta_ntar': delta_ntar_mcmc,
                        'lambda': lambda_mcmc,
                        'gamma': gamma_mcmc,
                        's_sq': s_sq_mcmc,
                        'rho': rho_mcmc,
                        'log_lhd': log_lhd_mcmc
                    })

    def import_bayes_lda_mcmc(self):
        lda_bayes_mcmc = sio.loadmat('{}/{}/{}/{}_lda_mcmc_trn_{}.mat'
                                     .format(self.parent_path,
                                             self.data_type,
                                             self.sub_folder_name[:4],
                                             self.sub_folder_name,
                                             self.trn_repetition))

        lda_bayes_mcmc_keys, _ = zip(*lda_bayes_mcmc.items())
        delta_tar = lda_bayes_mcmc['delta_tar']
        delta_ntar = lda_bayes_mcmc['delta_ntar']
        lambda_val = lda_bayes_mcmc['lambda']
        gamma_val = lda_bayes_mcmc['gamma']
        s_sq = lda_bayes_mcmc['s_sq']
        rho = lda_bayes_mcmc['rho']
        log_lhd = lda_bayes_mcmc['log_lhd']

        return [delta_tar, delta_ntar, lambda_val, gamma_val, s_sq, rho, log_lhd]

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
        # print(s_sq_accept_rate)
        # print(rho_accept_rate)

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
            threshold=0.5, mcmc=True,
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

        lambda_iter = lambda_iter[:, np.newaxis, np.newaxis]

        if mcmc:
            beta_tar = phi_fn @ (lambda_iter * delta_tar)
            beta_ntar = phi_fn @ (lambda_iter * delta_ntar)
        else:
            beta_tar = np.copy(delta_tar)
            beta_ntar = np.copy(delta_ntar)

        gamma_mcmc_binary = np.zeros_like(gamma_mcmc_mean)
        gamma_mcmc_binary[gamma_mcmc_mean > threshold] = 1
        # print(gamma_mcmc_binary.shape)

        x = list(self.time_range)
        gamma_mcmc_mean = np.around(gamma_mcmc_mean, decimals=3)
        common_name = 'lda_select_'

        if 'convol' in sim_folder_name:
            if mcmc:
                plot_name = common_name + 'convol_mcmc_mean_trn_' + str(self.trn_repetition)
                message = message + '_mcmc_trn_' + str(self.trn_repetition)
            else:
                plot_name = common_name + 'convol_std_mean_trn_' + str(self.trn_repetition)
                message = message + '_std_trn_' + str(self.trn_repetition)
        else:
            if mcmc:
                plot_name = common_name + 'mcmc_mean_trn_' + str(self.trn_repetition)
                message = message + '_mcmc_trn_' + str(self.trn_repetition)
            else:
                plot_name = common_name + 'std_mean_trn_' + str(self.trn_repetition)
                message = message + '_std_trn_' + str(self.trn_repetition)

        plot_pdf = bpdf.PdfPages('{}/{}/{}/{}.pdf'
                                 .format(self.parent_path,
                                         self.data_type,
                                         sim_folder_name,
                                         plot_name))
        if mcmc:
            for i in range(self.num_electrode):

                fig_1 = plt.figure(figsize=(12, 10))
                ax1 = fig_1.add_subplot(2, 1, 1)
                ax1.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="tar_mcmc")
                ax1.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="ntar_mcmc")
                ax1.fill_between(self.time_range, beta_tar_lower[i, :, 0], beta_tar_upper[i, :, 0],
                                 color='red', alpha=0.2)
                ax1.fill_between(self.time_range, beta_ntar_lower[i, :, 0], beta_ntar_upper[i, :, 0],
                                 color='blue', alpha=0.2)
                ax1.legend(loc='upper right')
                ax1.title.set_text(message + '_95%_credible_band_chan_' + str(i+1))

                ax2 = fig_1.add_subplot(2, 1, 2)
                ax2.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="tar_mcmc")
                ax2.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="ntar_mcmc")
                for j in range(self.n_length):
                    if gamma_mcmc_binary[i, j] == 0:
                        half_value = 1/2 * (beta_tar[i, j, 0] + beta_ntar[i, j, 0])
                        beta_tar[i, j, 0] = half_value
                        beta_ntar[i, j, 0] = half_value
                ax2.plot(self.time_range, beta_tar[i, :, 0], 'r-', label="tar_select")
                ax2.plot(self.time_range, beta_ntar[i, :, 0], 'b-', label="ntar_select")
                for x_i, y_i, prop_i in zip(x, list(beta_tar[i, :, 0]), list(gamma_mcmc_mean[i, :])):
                    plt.text(x_i, y_i, str(prop_i))
                ax2.hlines(y=0, xmin=self.time_range[0], xmax=self.time_range[-1])
                ax2.legend(loc="upper right")
                ax2.title.set_text(message + '_threshold_' + str(threshold) + '_chan_' + str(i+1))
                # plt.show()
                plt.close()
                plot_pdf.savefig(fig_1)
        else:
            for i in range(self.num_electrode):
                fig = plt.figure(figsize=(12, 10))
                plt.plot(self.time_range, beta_tar[i, :, 0], 'r-.', label="tar_std")
                plt.plot(self.time_range, beta_ntar[i, :, 0], 'b-.', label="ntar_std")
                for j in range(self.n_length):
                    if gamma_mcmc_binary[i, j] == 0:
                        half_value = 1/2 * (beta_tar[i, j, 0] + beta_ntar[i, j, 0])
                        beta_tar[i, j, 0] = half_value
                        beta_ntar[i, j, 0] = half_value
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
            s_sq_i, rho_i, gamma_i,
            phi_fn, trn_repetition, channel_ids=None
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
            s_sq_i: array_like, (num_electrode,)
            rho_i: array_like, (num_electrode,)
            gamma_i: array_like
                selection indicator, should have the same dimension (num_electrode, self.n_length)
            phi_fn: array_like
                should have input shape (num_electrode, n_length, u)
            trn_repetition: integer
                the number of sequence repetitions in the training set
            channel_ids: integer array_like

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
        gamma_i_mat = self.block_diagonal_mat(gamma_i, channel_ids=channel_ids)
        pres_mat_i = self.generate_proposal_ar1_pres_mat(s_sq_i, rho_i, channel_ids=channel_ids)
        idns = self.identity_n[np.newaxis, ...]
        # phi_fn_t = np.transpose(phi_fn, [0, 2, 1])

        a1_I_a0_gamma = self.a1 * idns + self.a0 * gamma_i_mat
        a1_I_gamma = self.a1 * (idns - gamma_i_mat)
        a0_I_a1_gamma = self.a0 * idns + self.a1 * gamma_i_mat
        a0_I_gamma = self.a0 * (idns - gamma_i_mat)

        # print('lambda_iter_i has shape {}'.format(lambda_iter_i.shape))
        # print('phi_fn has shape {}'.format(phi_fn.shape))

        mean_tar = lambda_iter_i * (a1_I_a0_gamma @ phi_fn @ delta_tar_i + a0_I_gamma @ phi_fn @ delta_ntar_i)
        mean_ntar = lambda_iter_i * (a0_I_a1_gamma @ phi_fn @ delta_ntar_i + a1_I_gamma @ phi_fn @ delta_tar_i)

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

        l_mvn_1_ordered = np.reshape(l_mvn_1_ordered, [self.num_letter,
                                                       self.num_repetition,
                                                       self.num_rep])
        l_mvn_0_ordered = np.reshape(l_mvn_0_ordered, [self.num_letter,
                                                       self.num_repetition,
                                                       self.num_rep])

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
            gamma_mcmc, s_sq_mcmc, rho_mcmc,
            phi_fn, trn_repetition, target_letters,
            channel_ids=None
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
            s_sq_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim)
            rho_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim)
            gamma_mcmc: array_like
                should have the input shape (mcmc_iterations, channel_dim, n_length)
            phi_fn: array_like
                should have the input shape (channel_dim, n_length, u)
            trn_repetition: integer
            burn_in: integer
            target_letters: list of characters, len(target_letters) = letter_dim
            channel_ids: a list of selected channel ids, indexing from 0 to num_electrode-1

        return:
        -----
            A dict of prediction result including sampling number and
            probability of being predicted correctly
        """
        lda_accuracy = []
        mcmc_iter = rho_mcmc.shape[0]
        if channel_ids is not None:
            print('The selected channels are {}'.format(channel_ids+1))
            eeg_signals_trun_all = eeg_signals_trun_all[:, channel_ids, ...]
            delta_tar_mcmc = delta_tar_mcmc[:, channel_ids, ...]
            delta_ntar_mcmc = delta_ntar_mcmc[:, channel_ids, ...]
            lambda_mcmc = lambda_mcmc[:, channel_ids]
            gamma_mcmc = gamma_mcmc[:, channel_ids, :]
            s_sq_mcmc = s_sq_mcmc[:, channel_ids]
            rho_mcmc = rho_mcmc[:, channel_ids]
            phi_fn = phi_fn[channel_ids, ...]

            print('eeg_signals_trun_all has shape {}'.format(eeg_signals_trun_all.shape))
            print('delta_tar_mcmc has shape {}'.format(delta_tar_mcmc.shape))
            print('delta_ntar_mcmc has shape {}'.format(delta_ntar_mcmc.shape))
            print('lambda_mcmc has shape {}'.format(lambda_mcmc.shape))
            print('gamma_mcmc has shape {}'.format(gamma_mcmc.shape))
            print('s_sq_mcmc has shape {}'.format(s_sq_mcmc.shape))
            print('rho_mcmc has shape {}'.format(rho_mcmc.shape))

        for i in range(mcmc_iter):
            # print('i={}, kappa={}'.format(i, self.kappa))
            pred_matrix_i = self.lda_two_step_estimation_mcmc_i(
                eeg_signals_trun_all, eeg_code,
                delta_tar_mcmc[i, ...], delta_ntar_mcmc[i, ...],
                lambda_mcmc[i, :], s_sq_mcmc[i, :], rho_mcmc[i, :], gamma_mcmc[i, ...],
                phi_fn, trn_repetition, channel_ids=channel_ids
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

    def save_mcmc_trace_plot(
            self, rho_mcmc, s_sq_mcmc, lambda_mcmc, mcmc_log_lhd, gamma_mcmc_mean,
            true_log_lhd, sim_folder_name
    ):
        r"""

        :param rho_mcmc:
        :param s_sq_mcmc:
        :param lambda_mcmc:
        :param mcmc_log_lhd:
        :param gamma_mcmc_mean: mean selection rate over MCMC iteratios, with shape (num_electrode, n_length)
        :param true_log_lhd: true log-likelihood, with shape (num_electrode,)
        :param sim_folder_name: string
        :return:
            A systematic plot including traceplot of rho, sigma_sq, and kernel variance lambda,
            log-likelihood change, and mean selection rate across channels.
        """
        plot_pdf = bpdf.PdfPages('{}/{}/{}/lda_select_mcmc_trace_plot_trn_{}.pdf'
                                 .format(self.parent_path,
                                         self.data_type,
                                         sim_folder_name,
                                         self.trn_repetition))
        # MCMC traceplot check
        for i in range(self.num_electrode):
            fig_1 = plt.figure(figsize=(12, 12))
            ax1 = fig_1.add_subplot(3, 1, 1)
            ax1.plot(rho_mcmc[:, i])
            ax1.set_ylim(0, 1)
            ax1.title.set_text('rho_channel_' + str(i + 1))
            ax2 = fig_1.add_subplot(3, 1, 2)
            ax2.plot(s_sq_mcmc[:, i])
            ax2.title.set_text('sim_sq_channel_' + str(i + 1))
            ax3 = fig_1.add_subplot(3, 1, 3)
            ax3.plot(lambda_mcmc[:, i])
            ax3.title.set_text('kernel-variance_channel_' + str(i + 1))
            plt.close()
            plot_pdf.savefig(fig_1)

        # log-likelihood traceplot and mean selection rate
        for i in range(self.num_electrode):
            fig_2 = plt.figure(figsize=(12, 12))
            ax1 = fig_2.add_subplot(2, 1, 1)
            ax1.plot(mcmc_log_lhd[:, i])
            ax1.title.set_text('log_lhd_with_truth_{}_channel_{}'.format(true_log_lhd[i], i+1))

            ax2 = fig_2.add_subplot(2, 1, 2)
            ax2.plot(gamma_mcmc_mean[i, :])
            ax2.title.set_text('mean_selection_rate_channel_' + str(i + 1))
            plt.close()
            plot_pdf.savefig(fig_2)

        plot_pdf.close()

    def save_lda_bayes_results(self, new_lda_bayes_result,
                               sub_folder_name, target_letters):

        file_dir = "{}/EEGBayesLDA/{}_lda_bayes_pred_select_trn_{}.csv"\
            .format(self.parent_path, sub_folder_name, self.trn_repetition)

        assert 1 <= self.trn_repetition <= self.num_repetition, print('wrong training repetition dim!')
        '''
        if self.trn_repetition == 1:
            task = 'w'
        else:
            task = 'a'
        '''
        # Let task be w for now
        task = 'w'

        with open(file_dir, task) as f:
            f_writer = csv.writer(f)
            if task == "a":
                f_writer.writerow([' '])
            l0 = ['trn_repetition', self.trn_repetition, 'mcmc_sample', new_lda_bayes_result['sample_num']]
            f_writer.writerow(l0)
            l_trn_test = ['train_test_divide']
            l_trn_test.extend(['train'] * self.trn_repetition)
            l_trn_test.extend(['test'] * (self.num_repetition - self.trn_repetition))
            f_writer.writerow(l_trn_test)
            l_seq_id = ['sequence_id']
            l_seq_id.extend([i for i in range(1, self.num_repetition+1)])
            f_writer.writerow(l_seq_id)
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





