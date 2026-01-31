import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.PreFun import *
from scipy import stats, linalg, special
from arspy import ars
from timeit import default_timer as timer


class BayesGenSeq(EEGPreFun):
    r"""
    Implement the Bayes generative method to sequence-based signals with feature selection.
    """

    def __init__(self, *args, **kwargs):
        super(BayesGenSeq, self).__init__(*args, **kwargs)

    def tile_and_combine_beta(self, beta_tar, beta_ntar):
        r"""
        :param beta_tar: array_like, (channel_dim, n_length, 1)
        :param beta_ntar: array_like, (channel_dim, n_length, 1)
        :return: beta_combined: array_like, (channel_dim, num_rep, n_length, 1)
        """
        beta_tar = np.tile(beta_tar[:, np.newaxis, ...], [1, self.flash_sum, 1, 1])
        beta_ntar = np.tile(beta_ntar[:, np.newaxis, ...], [1, self.non_flash_sum, 1, 1])
        beta_combined = np.concatenate([beta_tar, beta_ntar], axis=1)
        return beta_combined

    def permute_beta_by_type(self, beta_combined, eeg_type, channel_dim=1):
        r"""
        :param beta_combined:
        :param eeg_type: array_like, (num_rep,)
        :param channel_dim: integer
        :return: beta_permuted: array_like, (channel_dim, num_rep*n_length, 1)
        """
        permute_id = np.zeros_like(eeg_type, dtype='int8')
        permute_id[np.where(eeg_type == 1)[0]] = np.array([0, 1])
        permute_id[np.where(eeg_type != 1)[0]] = np.arange(2, 12)
        beta_combined = beta_combined[:, permute_id, ...]
        beta_combined = np.reshape(beta_combined, [channel_dim, self.num_rep * self.n_length, 1])

        return beta_combined

    def create_select_mat_gp(self, gamma_mat, channel_dim=1):

        r"""
        :param gamma_mat: array_like, (channel_dim, n_length)
        :param channel_dim: integer,
        :return: square matrix of (2*n_length, 2*n_length) from (alpha1,alpha0) to (beta1,beta0)
        See the notation from the manuscript

        beta1(t) = alpha1(t)I(t) + alpha0(t)(1-I(t))
        beta0(t) = alpha0(t)

        Notice that this function also applies to tgp setting
        where I(t) becomes a soft weight.
        """
        # if channel_ids is None:
        #     channel_ids = np.arange(self.num_electrode)
        # channel_dim = len(channel_ids)

        select_mat = np.tile(np.eye(self.n_length*2)[np.newaxis, ...], [channel_dim, 1, 1])
        gamma_mat_upp_left = self.block_diagonal_mat(gamma_mat, channel_dim)
        # gamma_mat_neg = -(gamma_mat - 1)
        gamma_mat_upp_right = np.eye(self.n_length) - self.block_diagonal_mat(gamma_mat, channel_dim)
        # gamma_mat_upp = np.concatenate([gamma_mat_upp_left, gamma_mat_upp_right], axis=-1)
        select_mat[:, :self.n_length, :self.n_length] = gamma_mat_upp_left
        select_mat[:, :self.n_length, self.n_length:] = gamma_mat_upp_right

        return select_mat

    def compute_beta_select_from_theta(
            self, theta_comb, s_theta_sq, zeta, phi_mat
    ):
        r"""
        :param theta_comb: (channel_dim, utar+untar, 1)
        :param s_theta_sq: (channel_dim,) or (channel_dim, 2)
        :param zeta: (channel_dim, n_length)
        :param phi_mat: (channel_dim, 2*n_length, utar+untar), similar-block-diag matrix
        :return:
                 alpha_post, beta_post,
                 each has dimension (channel_dim, 2*n_length, 1)
        """
        if len(s_theta_sq.shape) == 1:
            alpha_post = s_theta_sq[:, np.newaxis, np.newaxis] * np.matmul(phi_mat, theta_comb)
        else:
            s_theta_sq_mat = self.generate_s_theta_sq_mat(s_theta_sq[:, 0], s_theta_sq[:, 1])
            alpha_post = np.matmul(s_theta_sq_mat, np.matmul(phi_mat, theta_comb))
        beta_post = self.compute_beta_from_alpha(alpha_post, zeta)
        return alpha_post, beta_post

    def compute_beta_from_alpha(
            self, alpha_iter, zeta_mat_iter, channel_dim=1
    ):
        beta_post = np.matmul(self.create_select_mat_gp(zeta_mat_iter, channel_dim), alpha_iter)
        return beta_post

    def create_transform_mat(self, eeg_type, letter_dim, repet_num, reshape_option):
        r"""

        :param eeg_type: array_like, (letter_dim * repet_num * num_rep)
        :param letter_dim: integer
        :param repet_num: integer
        :param reshape_option:
        :return: long_matrix, convert from (2*n_length, 1) to
            (repet_num*letter_dim, 12*n_length, 2*n_length)
            by eeg_type
        """

        left_zero_mat = np.zeros([self.num_rep * self.n_length * repet_num * letter_dim,
                                  self.n_length])
        right_diag_mat = np.tile(np.eye(self.n_length),
                                 [self.num_rep * repet_num * letter_dim, 1])
        target_ids = np.where(eeg_type == 1)[0]
        for _, idi in enumerate(target_ids):
            row_low_i = idi * self.n_length
            row_upp_i = row_low_i + self.n_length
            left_zero_mat[row_low_i:row_upp_i, :] = np.eye(self.n_length)
            right_diag_mat[row_low_i:row_upp_i, :] = np.zeros([self.n_length, self.n_length])
            # right_diag_mat[row_low_i:row_upp_i, :] = 0
        transform_mat = np.concatenate([left_zero_mat, right_diag_mat], axis=-1)
        # print(transform_mat.shape)
        if reshape_option == 'seq':
            transform_mat = np.reshape(
                transform_mat,
                [repet_num * letter_dim, self.num_rep * self.n_length, 2 * self.n_length]
            )
        if reshape_option == 'super_seq':
            transform_mat = np.reshape(
                transform_mat,
                [letter_dim, repet_num * self.num_rep * self.n_length, 2 * self.n_length]
            )
        return transform_mat

    def reshape_x_mat_seq(self, x_mat_seq, letter_dim, repet_num, channel_ids):
        r"""
        :param x_mat_seq: array_like, (channel_dim, letter_dim, repet_num, seq_length, 1)
        :param letter_dim: integer
        :param repet_num: integer
        :param channel_ids: array_like,
        :return: x_mat_seq, array_like,
            (channel_dim, letter_dim * repet_num, seq_length, 1)
        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        assert x_mat_seq.shape == (channel_dim, letter_dim, repet_num, self.seq_length, 1)
        x_mat_seq = np.reshape(
            x_mat_seq, [channel_dim, letter_dim * repet_num, self.seq_length, 1])
        return x_mat_seq

    def create_alpha_transform_matrix(
            self, alpha_iter, channel_dim
    ):
        r"""
        :param alpha_iter: (channel_dim, 2*n_length, 1) before selection,
            kernel_lambda * phi_beta_mat @ theta_iter
        :param channel_dim:
        :return: a tuple with two arrays:
            a_left: (channel_dim, 2*n_length, n_length)
            a_right: (channel_dim, 2*n_length, 1)
        """
        alpha_tar, alpha_ntar = np.split(np.squeeze(alpha_iter, axis=-1), [self.n_length], axis=1)
        a_left = self.block_diagonal_mat(alpha_tar - alpha_ntar, channel_dim)
        a_left = np.concatenate([a_left, np.zeros_like(a_left)], axis=1)
        a_right = np.tile(alpha_ntar, [1, 2])
        return a_left, a_right[..., np.newaxis]

    def compute_log_likelihood_multi(
            self, alpha_iter, zeta_mat_iter, beta_iter, cov_s_iter, arg_rho_iter, cov_t_set,
            x_mat_seq, dt_mat
    ):
        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape

        if beta_iter is None:
            s_zeta_mat = self.create_select_mat_gp(zeta_mat_iter, channel_dim)
            beta_iter = np.matmul(s_zeta_mat, alpha_iter)[:, np.newaxis, ...]
        else:
            if len(beta_iter.shape) == 3:
                beta_iter = beta_iter[:, np.newaxis, ...]

        mean_mat = np.matmul(dt_mat, beta_iter)
        cov_t_iter = cov_t_set[arg_rho_iter]
        cov_t_chky_inv = np.linalg.inv(np.linalg.cholesky(cov_t_iter))
        pres_t_iter = np.matmul(np.transpose(cov_t_chky_inv), cov_t_chky_inv)[np.newaxis, ...]

        cov_s_chky_inv = np.linalg.inv(np.linalg.cholesky(cov_s_iter))
        pres_s_iter = np.matmul(np.transpose(cov_s_chky_inv), cov_s_chky_inv)[np.newaxis, ...]
        residual_mat = np.transpose(np.squeeze(x_mat_seq - mean_mat, axis=-1), axes=(1, 2, 0))
        # (seq_num, seq_length, channel_dim)
        quadratic_mat = np.matmul(
            np.matmul(np.transpose(residual_mat, axes=(0, 2, 1)), pres_t_iter), residual_mat
        )
        quadratic_part = -1/2 * np.trace(np.sum(np.matmul(pres_s_iter, quadratic_mat), axis=0))
        det_part = -1/2 * seq_num * seq_length * np.linalg.slogdet(cov_s_iter)[0] - \
                   1/2 * seq_num * channel_dim * np.linalg.slogdet(cov_t_iter)[0]

        return det_part + quadratic_part

    def update_s_x_sq(
            self, beta_iter, pres_chky_t_iter,
            dt_mat, x_mat_seq, letter_dim, repet_num,
            a_prior, b_prior,
            seq_bool, super_seq_length
    ):
        r"""
        :param beta_iter: array_like,
        :param pres_chky_t_iter: array_like, (channel_dim, seq_length, seq_length)
        :param dt_mat:
        :param x_mat_seq:
        :param letter_dim:
        :param repet_num:
        :param a_prior:
        :param b_prior: rate, either floating number or (channel_dim,)
        :param seq_bool: bool_like
        :param super_seq_length: integer, only valid when seq_bool is False
        :return:
        """
        if seq_bool:
            a_post = a_prior + 1 / 2 * letter_dim * repet_num * self.seq_length
        else:
            a_post = a_prior + 1 / 2 * letter_dim * super_seq_length
        beta_iter = beta_iter[:, np.newaxis, ...]
        pres_chky_t_iter = pres_chky_t_iter[:, np.newaxis, ...]
        pres_x_diff = np.matmul(pres_chky_t_iter, x_mat_seq - np.matmul(dt_mat, beta_iter))
        quadratic_sum = np.linalg.norm(pres_x_diff, ord=2, axis=-2, keepdims=True)**2
        b_post = b_prior + 1 / 2 * np.sum(quadratic_sum, axis=(-3, -2, -1))

        s_x_sq_post = []
        for e in range(self.num_electrode):
            gamma_e = np.random.gamma(shape=a_post, scale=1 / b_post[e], size=1)
            s_x_sq_post.append(1 / gamma_e)
        s_x_sq_post = np.stack(s_x_sq_post, axis=0)

        return np.squeeze(s_x_sq_post, axis=-1)

    @staticmethod
    def update_cov_s_multi(
            beta_iter, cov_t_iter, x_mat_seq, dt_mat, nu_s, psi_mat_s
    ):
        r"""
        :param beta_iter:
        :param cov_t_iter:
        :param x_mat_seq:
        :param dt_mat:
        :param nu_s:
        :param psi_mat_s:
        :return:
        """

        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape
        nu_post = nu_s + seq_num * seq_length
        mean_mat = np.matmul(dt_mat, beta_iter[:, np.newaxis, ...])  # (channel_dim, seq_num, seq_length, 1)
        cov_t_chky_inv = np.linalg.inv(np.linalg.cholesky(cov_t_iter))
        pres_t_iter = np.matmul(np.transpose(cov_t_chky_inv), cov_t_chky_inv)[np.newaxis, ...]

        residual_mat = np.transpose(np.squeeze(x_mat_seq - mean_mat, axis=-1), axes=(1, 2, 0))
        quadratic_mat = np.sum(
            np.matmul(np.matmul(np.transpose(residual_mat, axes=(0, 2, 1)), pres_t_iter), residual_mat), axis=0
        )  # (channel_dim, channel_dim)
        psi_post = psi_mat_s + quadratic_mat  # (channel_dim, seq_num, seq_length, 2*n_length)
        scale_cov_s_post = stats.invwishart(df=nu_post, scale=psi_post).rvs(1)
        if channel_dim == 1:
            scale_cov_s_post = np.ones([1, 1]) * scale_cov_s_post
        return scale_cov_s_post

    def update_corr_s_cs(
            self, beta_iter, sigma_s_sq_iter, arg_rho_iter, cov_t_set, x_mat_seq, dt_mat
    ):
        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape

        def log_pdf_rho_s(rho_s_input):
            rho_s = 1 / (1 + np.exp(-rho_s_input))
            # rho_s = np.exp(-rho_s_input)
            # print('inner logit_rho_s = {}, rho_s = {}'.format(rho_s_input, rho_s))
            cov_s = self.create_compound_symmetry_cov_mat(sigma_s_sq_iter, rho_s, channel_dim)
            log_lkd = self.compute_log_likelihood_multi(
                None, None, beta_iter, cov_s, arg_rho_iter, cov_t_set, x_mat_seq, dt_mat
            )
            return log_lkd

        # Use Adaptive Rejection Sampling
        n_samples = 1
        # the algorithm may be sensitive to the bound values
        # Can use the NoC data to estimate compound symmetry first.
        const_low = 0
        const_upp = 1.5
        rho_s_output_post = ars.adaptive_rejection_sampling(
            logpdf=log_pdf_rho_s, a=const_low, b=const_upp, domain=(const_low, const_upp), n_samples=n_samples
        )[n_samples-1]
        # rho_s_post = np.exp(-rho_s_output_post)
        rho_s_post = 1 / (1 + np.exp(-rho_s_output_post))
        # print('rho_s_post = {}'.format(rho_s_post))
        return self.create_compound_symmetry_cov_mat(1, rho_s_post, channel_dim)

    def update_corr_s_cs_general(
            self, beta_iter, sigma_s_sq_iter, arg_rho_iter, cov_t_set, x_mat_seq, dt_mat, spatial_corr_fix
    ):
        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape
        distance_mat = 1 - np.maximum(spatial_corr_fix, 0)
        # print('distance_mat = {}'.format(distance_mat))

        def log_pdf_rho_s(rho_s_input):
            # rho_s = np.exp(-rho_s_input)
            # print('inner rho_s_input = {}'.format(rho_s_input))
            cov_s = np.exp(-rho_s_input * distance_mat) * sigma_s_sq_iter
            log_lkd = self.compute_log_likelihood_multi(
                None, None, beta_iter, cov_s, arg_rho_iter, cov_t_set, x_mat_seq, dt_mat
            )
            return log_lkd

        # Use adaptive rejection sampling
        n_samples = 1
        rho_s_output_post = ars.adaptive_rejection_sampling(
            logpdf=log_pdf_rho_s, a=0.1, b=2, domain=(0.1, 2), n_samples=n_samples
        )[n_samples-1]

        rho_s_post = np.exp(-rho_s_output_post)
        # print('rho_s_post = {}'.format(rho_s_post))
        return np.exp(-rho_s_output_post * distance_mat)

    def update_sigma_s_sq(
            self, beta_iter, corr_s_iter, cov_t_iter, x_mat_seq, dt_mat, alpha_s, beta_s
    ):

        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape
        alpha_post = alpha_s + seq_num * seq_length * channel_dim / 2
        corr_s_inv_iter = self.compute_matrix_inv_cholesky(corr_s_iter)

        mean_mat = np.matmul(dt_mat, beta_iter[:, np.newaxis, ...])  # (channel_dim, seq_num, seq_length, 1)
        cov_t_chky_inv = np.linalg.inv(np.linalg.cholesky(cov_t_iter))
        pres_t_iter = np.matmul(np.transpose(cov_t_chky_inv), cov_t_chky_inv)[np.newaxis, ...]
        residual_mat = np.transpose(np.squeeze(x_mat_seq - mean_mat, axis=-1), axes=(1, 2, 0))
        quad_mat_iter = np.sum(
           np.matmul(np.matmul(np.transpose(residual_mat, axes=(0, 2, 1)), pres_t_iter), residual_mat), axis=0
        )  #

        beta_post = beta_s + 1 / 2 * np.trace(np.matmul(corr_s_inv_iter, quad_mat_iter))
        sigma_sq_s_post = 1 / np.random.gamma(shape=alpha_post, scale=1 / beta_post, size=1)
        return sigma_sq_s_post

    def update_lambda_x(self, s_x_sq_post, a_val):
        r"""
        :param s_x_sq_post:
        :param a_val: hyper-parameter for HalfCauchy(0, a_val)
        :return:
        """
        # https://arxiv.org/pdf/1508.03884.pdf
        lambda_x_post = []
        for e in range(self.num_electrode):
            beta_e = 1 / a_val ** 2 + 1 / s_x_sq_post[e]
            lambda_x_post.append(stats.invgamma(a=1, scale=1 / beta_e).rvs(1))
        lambda_x_post = np.stack(lambda_x_post, axis=0)
        return lambda_x_post

    @staticmethod
    def update_rho_t(
            beta_iter, s_x_sq_iter, pres_chky_t_set, logdet_pres_set,
            x_mat_seq, dt_mat, letter_dim, repet_num, seq_bool=True
    ):
        r"""
        :param beta_iter:
        :param s_x_sq_iter:
        :param pres_chky_t_set: array_like, (rho_level, channel_dim, 1, length, length)
        :param logdet_pres_set:
        :param x_mat_seq:
        :param dt_mat:
        :param letter_dim:
        :param repet_num:
        :param seq_bool: bool_like
        :return: Instead of using MH, we assume rho takes values from (0, 0.1, 0.2, ..., 0.9)
        """
        if seq_bool:
            noise_size = letter_dim * repet_num
        else:
            noise_size = letter_dim
        beta_iter = beta_iter[:, np.newaxis, ...]
        x_diff = (x_mat_seq - np.matmul(dt_mat, beta_iter))[np.newaxis, ...]
        log_sampling_q = - 1 / (2 * s_x_sq_iter) * np.sum(
            np.linalg.norm(np.matmul(pres_chky_t_set, x_diff), ord=2, axis=-2, keepdims=True)**2,
            axis=(2, 3, 4)
        )  # (rho_level, channel_dim)
        log_sampling = log_sampling_q + noise_size * logdet_pres_set
        rho_arg_max = np.argmax(log_sampling, axis=0)
        return rho_arg_max

    def update_rho_t_multi(
            self, alpha_iter, zeta_mat_iter, beta_iter, cov_s_iter,
            cov_t_set, x_mat_seq, dt_mat
    ):
        rho_set_level = len(cov_t_set)
        log_lkd_vec = np.zeros([rho_set_level])
        # print('rho_set_level has shape {}'.format(rho_set_level))
        # print('beta_iter has shape {}'.format(beta_iter.shape))
        for rho_id in range(rho_set_level):
            log_lkd_vec[rho_id] = self.compute_log_likelihood_multi(
                alpha_iter, zeta_mat_iter, beta_iter[:, np.newaxis, ...],
                cov_s_iter, rho_id, cov_t_set,
                x_mat_seq, dt_mat
            )
        # print(log_lkd_vec)
        arg_rho_iter = np.argmax(log_lkd_vec)
        # arg_rho_iter = 20
        return arg_rho_iter, log_lkd_vec[arg_rho_iter]

    def update_theta(
            self, zeta_iter, s_theta_sq_iter, s_x_sq_iter,
            pres_chky_t_iter, theta_prior_pres,
            x_mat_seq, dt_mat, phi_theta_mat, theta_prior_mean
    ):
        r"""
        :param zeta_iter: array_like, (channel_dim, n_length)
        :param s_theta_sq_iter: array_like, (channel_dim, 2*n_length, 2*n_length)
        :param s_x_sq_iter: array_like, (channel_dim,)
        :param pres_chky_t_iter: array_like, (channel_dim, super_seq_length, super_seq_length)
        :param r_mat_neghalf_iter: array_like, (channel_dim, 1, 2*n_length, 1)
        :param theta_prior_pres: array_like, (channel_dim, 2u, 2u)
        :param x_mat_seq: array_like, (channel_dim, seq_num, seq_length, 1)
        :param dt_mat: array_like,
        :param phi_theta_mat: array_like, (channel_dim, 50, 2u)
        :param theta_prior_mean: array_like,
        :return:
        """

        # channel_dim = 1
        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape
        theta_length = phi_theta_mat.shape[-1]

        s_x_sq_iter = s_x_sq_iter[:, np.newaxis, np.newaxis]
        pres_chky_t_iter = pres_chky_t_iter[:, np.newaxis, ...]
        s_zeta_mat = self.create_select_mat_gp(zeta_iter, channel_dim)
        quad_mat_part = np.matmul(
            dt_mat, np.matmul(s_zeta_mat, np.matmul(s_theta_sq_iter, phi_theta_mat))[:, np.newaxis, ...]
        )
        pres_chky_s_phi = np.matmul(pres_chky_t_iter, quad_mat_part)
        pres_chky_s_phi_t = np.transpose(pres_chky_s_phi, axes=(0, 1, 3, 2))
        pres_part = 1 / s_x_sq_iter * np.sum(np.matmul(pres_chky_s_phi_t, pres_chky_s_phi), axis=1) + theta_prior_pres
        pres_chky_x_diff = np.matmul(pres_chky_t_iter, x_mat_seq)  # no eta part

        eta_part = 1 / s_x_sq_iter * np.sum(np.matmul(pres_chky_s_phi_t, pres_chky_x_diff), axis=1)
        eta_part = eta_part + np.matmul(theta_prior_pres, theta_prior_mean)

        std_mvn_sample = np.random.normal(
            loc=0, scale=1, size=(self.num_electrode, theta_length, 1)
        )

        cov_part_half_t = np.linalg.inv(np.linalg.cholesky(pres_part))
        cov_part_half = np.transpose(cov_part_half_t, axes=(0, 2, 1))
        theta_post = np.matmul(cov_part_half, (std_mvn_sample + np.matmul(cov_part_half_t, eta_part)))

        return theta_post

    def update_theta_multi(
            self, zeta_mat_iter, pres_s_t_iter,
            x_mat_seq, dt_mat, theta_prior_pres, theta_length
    ):
        r"""
        :param zeta_mat_iter: array_like, (channel_dim, n_length)
        :param pres_s_t_iter: array_like, (channel_dim * seq_length, channel_dim * seq_length) pdf matrix
        :param x_mat_seq: array_like, (channel_dim, seq_num, seq_length, 1)
        :param dt_mat: array_like, (channel_dim, seq_num, seq_length, 2*n_length)
        :param theta_prior_pres: array_like, (2*n_length, 2*n_length)
        :param theta_length: integer
        :return: post theta estimates, (channel_dim, 2 * n_length, 1)
        """
        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape
        theta_prior_pres_kron = np.kron(np.eye(channel_dim), theta_prior_pres)
        s_zeta_mat = self.create_select_mat_gp(zeta_mat_iter, channel_dim)[:, np.newaxis, ...]
        # (channel_dim, 1, 2*n_length, 2*n_length)
        g_s_zeta_mat = np.matmul(dt_mat, s_zeta_mat)  # (channel_dim, seq_num, seq_length, 2*n_length)
        g_s_zeta_mat_diag = np.zeros([seq_num, seq_length*channel_dim, self.n_length*2*channel_dim])
        for e in range(channel_dim):
            row_e_low = e * seq_length
            row_e_upp = row_e_low + seq_length
            col_e_low = e * 2 * self.n_length
            col_e_upp = col_e_low + self.n_length * 2
            g_s_zeta_mat_diag[:, row_e_low:row_e_upp, col_e_low:col_e_upp] = np.copy(g_s_zeta_mat[e, ...])

        pres_s_t_iter = pres_s_t_iter[np.newaxis, ...]  # (1, channel_dim*seq_length, channel_dim*seq_length)
        common_left_iter = np.matmul(np.transpose(g_s_zeta_mat_diag, axes=(0, 2, 1)), pres_s_t_iter)
        pres_part = np.sum(np.matmul(common_left_iter, g_s_zeta_mat_diag), axis=0) + theta_prior_pres_kron

        x_mat_seq = np.reshape(np.transpose(x_mat_seq, axes=(1, 0, 2, 3)), [seq_num, channel_dim*seq_length, 1])
        eta_part = np.sum(np.matmul(common_left_iter, x_mat_seq), axis=0)

        std_mvn_sample = np.random.normal(
            loc=0, scale=1, size=(channel_dim * theta_length, 1)
        )
        cov_part_half_t = np.linalg.inv(np.linalg.cholesky(pres_part))
        cov_part_half = np.transpose(cov_part_half_t)
        theta_post = np.matmul(cov_part_half, (std_mvn_sample + np.matmul(cov_part_half_t, eta_part)))
        theta_post = np.reshape(theta_post, [channel_dim, 2 * self.n_length, 1])

        return theta_post

    def update_zeta(
            self, theta_iter, s_theta_sq_iter, s_x_sq_iter,
            pres_chky_t_iter,
            x_mat_seq, dt_mat, phi_theta_mat, lower, upper,
            burn_in_samples, s_zeta_sq, mu_zeta_prior
    ):
        r"""
        :param theta_iter: array_like, (channel_dim, 2u, 1)
        :param s_theta_sq_iter: array_like, (channel_dim, 2*n_length, 2*n_length)
        :param s_x_sq_iter: array_like, (channel_dim,)
        :param pres_chky_t_iter: (channel_dim, 2*n_length, 2*n_length),
        :param x_mat_seq: array_like,
        :param dt_mat: array_like, (1, seq_num, seq_length, 2*n_length)
        :param phi_theta_mat: array_like, (channel_dim, 2*n_length, 2u)
        :param lower: array_like, (channel_dim,)
        :param upper: array_like, (channel_dim,)
        :param burn_in_samples: integer
        :param s_zeta_sq: hyper-parameter, (channel_dim,)
        :param mu_zeta_prior: hyper-parameter, (channel_dim, n_length, 1)
        :return: truncated normal sample
        """

        # eta_iter = eta_iter[:, np.newaxis, ...]
        alpha_iter = np.matmul(s_theta_sq_iter, np.matmul(phi_theta_mat, theta_iter))  # use matrix representation
        a_diff, a2 = self.create_alpha_transform_matrix(alpha_iter, 1)
        a_diff = a_diff[:, np.newaxis, ...]
        a2 = a2[:, np.newaxis, ...]
        s_x_sq_iter = s_x_sq_iter[:, np.newaxis, np.newaxis]
        s_zeta_sq = s_zeta_sq[:, np.newaxis, np.newaxis]  # hyper-parameter
        pres_chky_t_iter = pres_chky_t_iter[:, np.newaxis, ...]

        pres_dt_a = np.matmul(pres_chky_t_iter, np.matmul(dt_mat, a_diff))
        pres_dt_a_t = np.transpose(pres_dt_a, axes=(0, 1, 3, 2))

        pres_part = 1 / s_x_sq_iter * np.sum(np.matmul(pres_dt_a_t, pres_dt_a), axis=1) + \
                    1 / s_zeta_sq * np.eye(self.n_length)[np.newaxis, ...]

        pres_x_diff = np.matmul(pres_chky_t_iter, (x_mat_seq - np.matmul(dt_mat, a2)))
        eta_part = 1 / s_x_sq_iter * np.sum(np.matmul(pres_dt_a_t, pres_x_diff), axis=1) + 1 / s_zeta_sq * mu_zeta_prior

        zeta_chky = np.linalg.cholesky(pres_part)
        # zeta_L = np.linalg.inv(zeta_chky)
        # joint_cov = np.transpose(zeta_L, (0, 2, 1)) @ zeta_L  # (channel_dim, n_length, n_length)
        joint_cov = self.compute_matrix_inv_cholesky(pres_part)
        joint_mu = np.stack([linalg.cho_solve((zeta_chky[i, ...], True), eta_part[i, ...])
                             for i in range(self.num_electrode)], axis=0)
        zeta_post = []
        for e in range(self.num_electrode):
            zeta_post_e = self.generate_trunc_norm(
                joint_mu[e, ...], joint_cov[e, ...],
                lower[e, :], upper[e, :],
                np.eye(self.n_length), np.zeros_like(joint_mu[0, ...]) + 0.5,
                n=1, burn_in_samples=burn_in_samples, thinning=1
            )
            zeta_post.append(zeta_post_e)
        zeta_post = np.stack(zeta_post, axis=0)
        zeta_post = np.squeeze(zeta_post, axis=-1)

        return zeta_post, joint_mu, joint_cov

    def update_zeta_multi(
            self, alpha_iter, pres_s_t_iter,
            x_mat_seq, dt_mat, zeta_prior_pres, zeta_prior_mean,
            lower, upper, burn_in_samples
    ):
        r"""
        :param alpha_iter: array_like, (channel_dim * 2 * n_length, 1)
        :param pres_s_t_iter: array_like, (channel_dim * seq_length, channel_dim * seq_length)
        :param x_mat_seq: (channel_dim, seq_num, seq_length, 1)
        :param dt_mat: (1, seq_num, seq_length, 2*n_length)
        :param zeta_prior_pres: (n_length, n_length)
        :param zeta_prior_mean: (n_length, 1)
        :param lower: array_like, (n_length,)
        :param upper: array_like, (n_length,)
        :parma burn_in_samples: integer
        :return:
        """
        channel_dim, seq_num, seq_length, _ = x_mat_seq.shape
        zeta_prior_pres_kron = np.kron(np.eye(channel_dim), zeta_prior_pres)
        # lower and upper are the same across channels
        lower = np.tile(lower, reps=channel_dim)
        upper = np.tile(upper, reps=channel_dim)

        alpha_iter = np.reshape(alpha_iter, [channel_dim, self.n_length * 2, 1])
        a_diff, a2 = self.create_alpha_transform_matrix(alpha_iter, channel_dim)
        a_diff = a_diff[:, np.newaxis, ...]  # (channel_dim, 1, 2*n_length, n_length)
        a2 = a2[:, np.newaxis, ...]  # (channel_dim, 1, 2*n_length, 1)
        g_a1_mat = np.matmul(dt_mat, a_diff) # (channel_dim, seq_num, seq_length, n_length)
        g_a1_mat_diag = np.zeros([seq_num, channel_dim * seq_length, channel_dim * self.n_length])
        g_a2_mat = np.reshape(np.transpose(np.matmul(dt_mat, a2), axes=(1, 0, 2, 3)),
                              [seq_num, channel_dim * seq_length, 1])

        for e in range(channel_dim):
            row_e_low = e * seq_length
            row_e_upp = row_e_low + seq_length
            col_e_low = e * self.n_length
            col_e_upp = col_e_low + self.n_length
            g_a1_mat_diag[:, row_e_low:row_e_upp, col_e_low:col_e_upp] = np.copy(g_a1_mat[e, ...])

        # cov_t_iter = cov_t_set[arg_rho_iter]  # (seq_length, seq_length)
        # cov_s_t_iter = np.kron(cov_s_iter, cov_t_iter)
        # cov_s_t_chky_inv = np.linalg.inv(np.linalg.cholesky(cov_s_t_iter))
        # pres_s_t_iter = np.matmul(np.transpose(cov_s_t_chky_inv), cov_s_t_chky_inv)[np.newaxis, ...]
        pres_s_t_iter = pres_s_t_iter[np.newaxis, ...]
        common_left_iter = np.matmul(np.transpose(g_a1_mat_diag, axes=(0, 2, 1)), pres_s_t_iter)
        pres_part = np.sum(np.matmul(common_left_iter, g_a1_mat_diag), axis=0) + zeta_prior_pres_kron
        x_mat_seq = np.reshape(np.transpose(x_mat_seq, axes=(1, 0, 2, 3)), [seq_num, channel_dim*seq_length, 1])
        eta_part = np.sum(np.matmul(common_left_iter, x_mat_seq - g_a2_mat), axis=0) + \
                   np.tile(zeta_prior_mean, reps=(channel_dim, 1))

        # Given posterior mean and covariance matrix, generate truncated normal samples
        zeta_chky = np.linalg.cholesky(pres_part)
        zeta_L = np.linalg.inv(zeta_chky)
        joint_cov = np.matmul(np.transpose(zeta_L), zeta_L)  # (channel_dim * n_length, channel_dim * n_length)
        joint_mu = linalg.cho_solve((zeta_chky, True), eta_part)
        x0 = np.random.uniform(
            low=0.2, high=0.8, size=(channel_dim * self.n_length, 1)
        )
        zeta_post = self.generate_trunc_norm(
            joint_mu, joint_cov,
            lower, upper,
            np.eye(channel_dim*self.n_length), x0,
            n=1, burn_in_samples=burn_in_samples, thinning=1
        )
        zeta_post = np.reshape(zeta_post, [channel_dim, self.n_length])

        return zeta_post, joint_mu, joint_cov

    def update_var_kernel_mat(
            self, alpha_mean_iter, kappa_inv_iter, hyper_a, hyper_b
    ):
        r"""
        :param alpha_mean_iter: mean curve, (channel_dim, n_length_fit, 1)
        :param kappa_inv_iter: precision matrix, (channel_dim, n_length_fit, n_length_fit)
        :param hyper_a: shape
        :param hyper_b: rate
        :return:
            If we want to estimate variance term, it is based on latency stage, where
            alpha_tar ~ GP(0, kappa_tar), alpha_ntar ~ GP(0, kappa_ntar)
            equivalent of
            alpha_tar ~ N(0, var_tar * kappa_tar), alpha_ntar ~ N(0, var_ntar * kappa_ntar)
            The posterior of var_tar/var_ntar are still inverse gamma format.
            var_tar ~ IG (hyper_a + 1/2, hyper_b + 1/2 * alpha_tar^T @ kappa_tar_inv @ alpha_tar)
            var_ntar ~ IG (hyper_a + 1/2, hyper_b + 1/2 * alpha_ntar^T @ kappa_ntar_inv @ alpha_ntar)
        """
        hyper_a_post = hyper_a + 1 / 2
        hyper_b_post = hyper_b + 1 / 2 * np.transpose(alpha_mean_iter, [0, 2, 1]) @ kappa_inv_iter @ alpha_mean_iter
        var_kernel_post = []
        for e in range(self.num_electrode):
            gamma_e = np.random.gamma(shape=hyper_a_post, scale=1 / hyper_b_post[e], size=1)
            var_kernel_post.append(1 / gamma_e)
        var_kernel_post = np.stack(var_kernel_post, axis=0)
        return np.squeeze(var_kernel_post, axis=-1)

    def update_scale_kernel_mat_mh(
            self, scale_kernel_pre, scale_pres_mat_pre, var_kernel_iter, alpha_mean_iter, **kwargs
    ):
        r"""
        :param scale_kernel_pre: array_like, (channel_dim,)
        :param scale_pres_mat_pre: array_like, (channel_dim, n_length_fit, n_length_fit)
        :param var_kernel_iter: array_like, (channel_dim,)
        :param alpha_mean_iter: array_like, (channel_dim, n_length_fit, 1)
        :param **kwargs: include option, step_size, rate
        :return:
            To estimate scale_kernel_mat_mh, we use Metropolis-Hastings Algorithm.
            We try random walk and exponential dist as proposal dist
            For exponential dist, we need to consider proposal ratio
        """
        # Propose new scale_kernel state
        scale_kernel_new = np.zeros([self.num_electrode])
        accept_init = np.ones([self.num_electrode])
        for e in range(self.num_electrode):
            scale_kernel_new_e, accept_init_e = self.generate_proposal_proportion_state(scale_kernel_pre, **kwargs)
            scale_kernel_new[e] = scale_kernel_new_e
            accept_init[e] = accept_init_e

        # log prior ratio
        a = kwargs['a']
        b = kwargs['b']
        log_prior_ratio = self.compute_log_prior_ratio_proportion(scale_kernel_pre, scale_kernel_new, a, b)
        option = kwargs['option']

        # log proposal ratio
        if option == 'rw':
            log_proposal_ratio = np.zeros([self.num_electrode])
        else:
            scale = kwargs['scale']
            q_new = stats.expon.logpdf(scale_kernel_new, scale=scale)
            q_old = stats.expon.logpdf(scale_kernel_pre, scale=scale)
            log_proposal_ratio = q_old - q_new

        # log likelihood ratio
        index_points = kwargs['index_points']
        log_lkd_new = np.zeros([self.num_electrode])
        log_lkd_old = np.zeros([self.num_electrode])

        def compute_log_lkd_latency(kernel_pres, kernel_var, alpha_mean):
            log_lkd_temp = np.linalg.slogdet(kernel_pres / kernel_var)[1] / 2 - \
                           1 / (2 * kernel_var) * alpha_mean.T @ kernel_pres @ alpha_mean
            log_lkd_temp = np.squeeze(log_lkd_temp)
            return log_lkd_temp

        for e in range(self.num_electrode):
            scale_pres_mat_new_e = self.create_kernel_mat_complex(
                [scale_kernel_new[e]], [self.n_length], index_points
            )
            log_lkd_new[e] = compute_log_lkd_latency(
                scale_pres_mat_new_e, var_kernel_iter[e], alpha_mean_iter[e, ...]
            )
            log_lkd_old[e] = compute_log_lkd_latency(
                scale_pres_mat_pre[e, ...], var_kernel_iter[e], alpha_mean_iter[e, ...]
            )
        log_lkd_ratio = log_lkd_new - log_lkd_old

        alpha_mh = log_prior_ratio + log_proposal_ratio + log_lkd_ratio
        alpha_unif = np.log(np.random.uniform(low=0, high=1, size=self.num_electrode))

        for e in range(self.num_electrode):
            if alpha_mh[e] <= alpha_unif[e]:
                # reject, keep it with pre values
                scale_kernel_new[e] = scale_kernel_pre[e]
                accept_init[e] = 0
        return scale_kernel_new, accept_init

    def generate_s_theta_sq_mat(
            self, s_theta_sq_tar, s_theta_sq_ntar
    ):
        r"""
        :param s_theta_sq_tar: array_like, (channel_dim,)
        :param s_theta_sq_ntar: array_like, (channel_dim,)
        :return:
        """
        s_theta_sq_tar = s_theta_sq_tar[:, np.newaxis, np.newaxis]
        s_theta_sq_ntar = s_theta_sq_ntar[:, np.newaxis, np.newaxis]
        s_theta_sq_mat = np.eye(2*self.n_length)[np.newaxis, ...]
        s_theta_sq_mat = np.tile(s_theta_sq_mat, [self.num_electrode, 1, 1])
        s_theta_sq_mat[:, :self.n_length, :self.n_length] = \
            s_theta_sq_tar * s_theta_sq_mat[:, :self.n_length, :self.n_length]
        s_theta_sq_mat[:, self.n_length:, self.n_length:] = \
            s_theta_sq_ntar * s_theta_sq_mat[:, self.n_length:, self.n_length:]

        return s_theta_sq_mat

    # Visualization and classification-related functions in png format.
    def plot_mean_curve_inference(
            self, beta_tar_list, beta_ntar_list, channel_ids
    ):
        r"""
        :param beta_tar_list:
        :param beta_ntar_list:
        :param channel_ids:
        :return:
                This function is intended for manuscript/presentation plot purpose!
        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)

        # target, post
        beta_tar_post_s_est, beta_tar_post_s_low, beta_tar_post_s_upp = beta_tar_list
        # non-target, post
        beta_ntar_post_s_est, beta_ntar_post_s_low, beta_ntar_post_s_upp = beta_ntar_list

        for i in range(channel_dim):
            # post selection, point estimate & 95% credible intervals
            plt.figure(figsize=(6, 7))
            plt.plot(self.time_range, beta_tar_post_s_est[i, :, 0], 'r-.', label="post-target")
            plt.fill_between(self.time_range, beta_tar_post_s_low[i, :, 0], beta_tar_post_s_upp[i, :, 0],
                             color='red', alpha=0.2)
            plt.plot(self.time_range, beta_ntar_post_s_est[i, :, 0], 'b-.', label="post-non-target")
            plt.fill_between(self.time_range, beta_ntar_post_s_low[i, :, 0], beta_ntar_post_s_upp[i, :, 0],
                             color='blue', alpha=0.2)
            plt.legend(loc='upper right')
            # plt.title('Mean Curves with 95% Credible Band after Selection')
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude (muV)')
            plt.show()

    def visualize_selection_indicator(
            self, beta_tar_true, beta_ntar_true,
            beta_tar_post, beta_ntar_post,
            zeta_binary, num_rep_fit, channel_ids,
            method_name, sim_type, scenario_name,
            job_id=None, sim_dat=True, **kwargs
    ):
        r"""
        :param beta_tar_true: None or array_like, (channel_dim, n_length_fit, 1)
        :param beta_ntar_true: (channel_dim, n_length_fit, 1)
        :param beta_tar_post: list of 3 arrays, (channel_dim, n_length_fit, 1)
        :param beta_ntar_post: list of 3 arrays, (channel_dim, n_length_fit, 1)
        :param zeta_binary: array_like, (channel_dim, n_length_fit)
        :param num_rep_fit:
        :param channel_ids:
        :param method_name:
        :param sim_type:
        :param scenario_name:
        :param job_id:
        :param sim_dat: bool
        :return:
        """

        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        x = list(self.time_range)
        if job_id is None:
            plot_name = method_name + '_convol_seq_trn_' + str(num_rep_fit)
        else:
            plot_name = method_name + '_convol_seq_trn_' + str(num_rep_fit) + '_' + str(job_id)
        if sim_dat:
            plot_pdf = bpdf.PdfPages('{}/{}/{}/{}_{}_{}.pdf'.format(
                self.parent_sim_output_path, method_name, scenario_name,
                self.sub_folder_name, sim_type, plot_name)
            )
        else:
            if 'scale' in kwargs.keys() and 'gamma' in kwargs.keys():
                hyper_scale = kwargs['scale']
                hyper_gamma = kwargs['gamma']
                file_dir0 = '{}/{}/{}/scale={}, gamma={}'.format(
                    self.parent_eeg_output_path, method_name, scenario_name,
                    hyper_scale, hyper_gamma
                )
                try:
                    os.mkdir(file_dir0)
                    print('Directory', file_dir0, ' is created.')
                except FileExistsError:
                    print('Directory ', file_dir0, ' already exists.')
                plot_pdf = bpdf.PdfPages('{}/{}_{}_{}.pdf'.format(
                    file_dir0, self.sub_folder_name, sim_type, plot_name)
                )
            else:
                plot_pdf = bpdf.PdfPages('{}/{}/{}/{}_{}_{}.pdf'.format(
                    self.parent_eeg_output_path, method_name, scenario_name,
                    self.sub_folder_name, sim_type, plot_name)
                )

        tar_post_s_est = np.mean(beta_tar_post, axis=0)
        ntar_post_s_est = np.mean(beta_ntar_post, axis=0)
        tar_post_s_low, tar_post_s_upp = np.quantile(beta_tar_post, q=[0.025, 0.975], axis=0)
        ntar_post_s_low, ntar_post_s_upp = np.quantile(beta_ntar_post, q=[0.025, 0.975], axis=0)

        if sim_dat:
            for i in range(channel_dim):
                fig_1 = plt.figure(figsize=(16, 8))
                # true curve and credible band (to see if they cover all relevant temporal locations
                ax1 = fig_1.add_subplot(1, 2, 1)
                ax1.plot(self.time_range, beta_tar_true[i, :, 0], 'r-', label="tar-true")
                ax1.fill_between(self.time_range, tar_post_s_low[i, :, 0], tar_post_s_upp[i, :, 0],
                                 color='red', alpha=0.2)
                ax1.plot(self.time_range, beta_ntar_true[i, :, 0], 'b-', label="ntar-true")
                ax1.fill_between(self.time_range, ntar_post_s_low[i, :, 0], ntar_post_s_upp[i, :, 0],
                                 color='blue', alpha=0.2)
                ax1.legend(loc='upper right')
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (muV)')
                ax1.title.set_text('True Curves with 95% Credible Band after Selection')

                # post selection, point estimate & true curve
                ax2 = fig_1.add_subplot(1, 2, 2)
                ax2.plot(self.time_range, tar_post_s_est[i, :, 0], 'r-', label="tar-post")
                ax2.plot(self.time_range, ntar_post_s_est[i, :, 0], 'b-', label="ntar-post")
                ax2.plot(self.time_range, beta_tar_true[i, :, 0], 'r-.', label="tar-true")
                ax2.plot(self.time_range, beta_ntar_true[i, :, 0], 'b-.', label="ntar-true")
                ax2.legend(loc='upper right')
                ax2.title.set_text('Mean Curves with 95% Credible Band after Selection')
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (muV)')

                # Add selection rates
                for x_i, y_i, select_i in zip(x, list(tar_post_s_est[i, :, 0]), list(zeta_binary[i, :])):
                    plt.text(x_i, y_i, str(select_i))
                ax2.hlines(y=0, xmin=0, xmax=self.time_n_length)
                ax2.legend(loc="upper right")
                fig_1.suptitle(sim_type + '_chan_' + str(i+1))
                # plt.show()
                plt.close()
                plot_pdf.savefig(fig_1)
        else:
            for i in range(channel_dim):
                fig_2 = plt.figure(figsize=(10, 10))
                plt.plot(self.time_range, tar_post_s_est[i, :, 0], 'r', label='tar-post')
                plt.plot(self.time_range, ntar_post_s_est[i, :, 0], 'b', label='ntar-post')
                plt.fill_between(self.time_range, tar_post_s_low[i, :, 0], tar_post_s_upp[i, :, 0],
                                 color='red', alpha=0.2)
                plt.fill_between(self.time_range, ntar_post_s_low[i, :, 0], ntar_post_s_upp[i, :, 0],
                                 color='blue', alpha=0.2)
                plt.legend(loc='best')
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (muV)')
                plt.title('Posterior Mean Curve with 95% Credible Band after Selection.')
                plt.close()
                plot_pdf.savefig(fig_2)
        plot_pdf.close()
        return 'Estimated Mean Functions with 95% Credible Band are done!'

    def bayes_generate_pred_latency_full(
            self, beta_comb, s_x_sq, pres_chky_t,
            eeg_code, x_mat_seq, d_x_mat,
            letter_dim, repet_num, reshape_option,
            normal_bool, t_df=None, log_prior_prob=np.log(1/36*np.ones([36]))
    ):
        code_1d = np.reshape(eeg_code, [letter_dim * repet_num * self.num_rep])
        channel_dim = beta_comb.shape[0]
        log_lkd_test = []
        beta_comb = beta_comb[:, np.newaxis, ...]
        pres_chky_t = pres_chky_t[:, np.newaxis, ...]
        super_seq_len = x_mat_seq.shape[2]

        if normal_bool:
            for _, row_idx in enumerate(self.row_set):
                for _, col_idx in enumerate(self.column_set):
                    type_1d_36 = np.zeros_like(code_1d)
                    type_1d_36[code_1d == row_idx] = 1
                    type_1d_36[code_1d == col_idx] = 1

                    t_mat_36 = self.create_transform_mat(
                        type_1d_36, letter_dim, repet_num, reshape_option
                    )
                    dt_mat_36 = np.matmul(d_x_mat, t_mat_36)[np.newaxis, ...]
                    pr_x_mat_diff = np.matmul(pres_chky_t, x_mat_seq - np.matmul(dt_mat_36, beta_comb))
                    log_quad_sum_part = np.sum(np.linalg.norm(pr_x_mat_diff, ord=2, axis=-2, keepdims=True) ** 2,
                                               axis=(-2, -1))  # (channel_dim, letter_dim)
                    log_det_pres_chkt_t_part = np.squeeze(np.linalg.slogdet(pres_chky_t)[1], axis=-1)
                    log_s_x_sq_part = -super_seq_len/ 2 * np.log(s_x_sq)
                    log_quad_sum_part = np.sum(-1 / (2 * s_x_sq[:, np.newaxis]) * log_quad_sum_part, axis=-1)
                    log_lkd_total = log_s_x_sq_part + log_det_pres_chkt_t_part + log_quad_sum_part
                    log_lkd_test.append(log_lkd_total)

        else:
            # student-t log-likelihood
            for _, row_idx in enumerate(self.row_set):
                for _, col_ids in enumerate(self.column_set):
                    type_1d_36 = np.zeros_like(code_1d)
                    type_1d_36[code_1d == row_idx] = 1
                    type_1d_36[code_1d == col_ids] = 1

                    t_mat_36 = self.create_transform_mat(
                        type_1d_36, letter_dim, repet_num, reshape_option
                    )
                    dt_mat_36 = np.matmul(d_x_mat, t_mat_36)[np.newaxis, ...]
                    pr_x_mat_diff = np.matmul(pres_chky_t, x_mat_seq - np.matmul(dt_mat_36, beta_comb))
                    log_quad_sum_part = np.sum(np.linalg.norm(pr_x_mat_diff, ord=2, axis=-2, keepdims=True) ** 2,
                                             axis=(-2, -1))  # (channel_dim, letter_dim)
                    log_quad_sum_part = -(t_df + super_seq_len) / 2 * np.sum(np.log(1 + 1 / t_df * log_quad_sum_part), axis=-1)
                    log_lkd_test.append(log_quad_sum_part)

        log_lkd_test = np.stack(log_lkd_test, axis=-1) + log_prior_prob[np.newaxis, :]
        if channel_dim == 1:
            return np.squeeze(log_lkd_test, axis=0)
        else:
            return log_lkd_test

    def bayes_generate_pred_latency_full_multi(
            self, beta_iter, cov_s_iter, arg_rho_iter, cov_t_set,
            eeg_code, x_mat_seq, d_x_mat, eeg_dat_type,
            log_prior_prob=np.log(1/36*np.ones([36]))
    ):
        single_letter_dim = 1; repet_num = 1
        channel_dim, _, seq_length, _ = x_mat_seq.shape
        log_lkd_test = []
        for _, row_idx in enumerate(self.row_set):
            for _, col_idx in enumerate(self.column_set):
                eeg_type_temp = np.zeros_like(eeg_code)
                eeg_type_temp[eeg_code == row_idx] = 1
                eeg_type_temp[eeg_code == col_idx] = 1

                t_mat_36_sub = self.create_transform_mat(
                    eeg_type_temp, single_letter_dim, repet_num,
                    reshape_option=eeg_dat_type
                )
                dt_mat_36_sub = np.matmul(d_x_mat, t_mat_36_sub)[np.newaxis, ...]
                log_lkd_test.append(
                    self.compute_log_likelihood_multi(
                        None, None, beta_iter, cov_s_iter,
                        arg_rho_iter, cov_t_set,
                        x_mat_seq, dt_mat_36_sub
                    )
                )
        log_lkd_test = np.stack(log_lkd_test, axis=-1) + log_prior_prob

        return log_lkd_test

    @staticmethod
    def bayes_factor_harmonic_mean(log_lkd_mcmc_ls, step=2):

        r"""
        :param log_lkd_mcmc_ls: list of 1d-array of dimension (B,)
        :param step: integer
        :return: harmonic-mean of log_lkd and the index
        The formula: pi_HM (y) = B /(sum_{t=1}^B pi(y | Theta_t)^{-1})
        """
        model_num = len(log_lkd_mcmc_ls)
        log_lkd_hm_ls = []
        B = log_lkd_mcmc_ls[0].shape[0]
        id = np.arange(0, B, step)
        B_id = len(id)
        for i in range(model_num):
            log_lkd_min_i = np.min(log_lkd_mcmc_ls[i][id, 0])
            log_lkd_i_diff = log_lkd_min_i - log_lkd_mcmc_ls[i][id, 0]
            log_lkd_hm_i = np.log(B_id) + log_lkd_min_i - np.log(np.sum(np.exp(log_lkd_i_diff)))
            log_lkd_hm_ls.append(log_lkd_hm_i)
        arg_max_log_lkd = np.argmax(log_lkd_hm_ls)

        return log_lkd_hm_ls, arg_max_log_lkd

    def split_convol_super_seq_by_seq_ids(
            self, signal_x, eeg_type, eeg_code, rep_odd_id, rep_even_id,
            odd_reshape=3, even_reshape=3
    ):
        r"""
        :param signal_x: array_like, (channel_dim, num_letter, super_seq_length, 1)
        :param eeg_type: array_like, (num_letter, num_repetition, num_rep)
        :param eeg_code: array_like, (num_letter, num_repetition, num_rep)
        :param rep_odd_id: array_like, subset of 1:num_repetition
        :param rep_even_id: array_like, subset of 1:num_repetition, complementary of rep_train_id
        :param odd_reshape: integer_like,
        :param even_reshape: integer_like,

        :return: split signal_x, eeg_type, eeg_code by rep_train_id, rep_test_id,
        split signal_x is single-seq-based, i.e.,
        (channel_dim, num_letter * len(rep_train_id), single_seq_len, 1)
        (channel_dim, num_letter * len(rep_test_id), single_seq_len, 1)
        eeg_type, eeg_code has shapes, i.e.,
        (num_letter * len(rep_train_id), 1, 12)
        (num_letter * len(rep_test_id), 1, 12)
        """
        rep_odd_len = len(rep_odd_id)
        rep_even_len = len(rep_even_id)
        single_seq_len = (self.num_rep - 1) * self.flash_and_pause_length + self.n_length
        signal_x_odd = np.zeros([
            self.num_electrode, self.num_letter, rep_odd_len, single_seq_len, 1]
        )
        signal_x_even = np.zeros([
            self.num_electrode, self.num_letter, rep_even_len, single_seq_len, 1]
        )
        eeg_type_odd = np.zeros([self.num_letter, rep_odd_len, 1, self.num_rep])
        eeg_type_even = np.zeros([self.num_letter, rep_even_len, 1, self.num_rep])
        eeg_code_odd = np.zeros([self.num_letter, rep_odd_len, 1, self.num_rep])
        eeg_code_even = np.zeros([self.num_letter, rep_even_len, 1, self.num_rep])

        for l in range(self.num_letter):
            for i_odd_seq, i_odd_id in enumerate(rep_odd_id):
                i_odd_id = i_odd_id - 1
                i_odd_low = i_odd_id * self.num_rep * self.flash_and_pause_length
                i_odd_upp = i_odd_low + single_seq_len
                signal_x_odd[:, l, i_odd_seq, ...] = signal_x[:, l, i_odd_low: i_odd_upp, :]
                eeg_type_odd[l, i_odd_seq, 0, :] = eeg_type[l, i_odd_id, :]
                eeg_code_odd[l, i_odd_seq, 0, :] = eeg_code[l, i_odd_id, :]

            for i_even_seq, i_even_id in enumerate(rep_even_id):
                i_even_id = i_even_id - 1
                i_even_low = i_even_id * self.num_rep * self.flash_and_pause_length
                i_even_upp = i_even_low + single_seq_len
                signal_x_even[:, l, i_even_seq, ...] = signal_x[:, l, i_even_low: i_even_upp, :]
                eeg_type_even[l, i_even_seq, 0, :] = eeg_type[l, i_even_id, :]
                eeg_code_even[l, i_even_seq, 0, :] = eeg_code[l, i_even_id, :]

        if odd_reshape == 1:
            # final reshape for training set
            signal_x_odd = np.reshape(
                signal_x_odd,
                [self.num_electrode, self.num_letter * rep_odd_len, single_seq_len, 1]
            )
            eeg_type_odd = np.reshape(eeg_type_odd, [self.num_letter * rep_odd_len * self.num_rep])
            eeg_code_odd = np.reshape(eeg_code_odd, [self.num_letter * rep_odd_len * self.num_rep])

        elif odd_reshape == 3:
            signal_x_odd = np.reshape(
                signal_x_odd,
                [self.num_electrode, self.num_letter, rep_odd_len, single_seq_len, 1]
            )
            eeg_type_odd = np.reshape(eeg_type_odd, [self.num_letter, rep_odd_len, self.num_rep])
            eeg_code_odd = np.reshape(eeg_code_odd, [self.num_letter, rep_odd_len, self.num_rep])

        else:
            print('Keep 2-dim format, do nothing but exit.')
            pass

        if even_reshape == 1:
            # final reshape for testing set
            signal_x_even = np.reshape(
                signal_x_even,
                [self.num_electrode, self.num_letter * rep_even_len, single_seq_len, 1]
            )
            eeg_type_even = np.reshape(eeg_type_even, [self.num_letter * rep_even_len * self.num_rep])
            eeg_code_even = np.reshape(eeg_code_even, [self.num_letter * rep_even_len * self.num_rep])
        elif even_reshape == 3:
            # final reshape for testing set
            signal_x_even = np.reshape(
                signal_x_even,
                [self.num_electrode, self.num_letter, rep_even_len, single_seq_len, 1]
            )
            eeg_type_even = np.reshape(eeg_type_even, [self.num_letter, rep_even_len, self.num_rep])
            eeg_code_even = np.reshape(eeg_code_even, [self.num_letter, rep_even_len, self.num_rep])
        else:
            pass

        print('signals_x_odd, signals_x_even with shape {}, {}'.format(signal_x_odd.shape, signal_x_even.shape))
        print('type_odd, type_even with shape {}, {}'.format(eeg_type_odd.shape, eeg_type_even.shape))
        print('code_odd, code_even with shape {}, {}'.format(eeg_code_odd.shape, eeg_code_even.shape))

        return [signal_x_odd, eeg_type_odd, eeg_code_odd,
                signal_x_even, eeg_type_even, eeg_code_even,
                single_seq_len]

    def standard_mcmc_prepare(
            self, s_tar, s_ntar, var_tar, var_ntar, kernel_option,
            gamma_val, scale_sine, periodicity,
            n_length_fit, seq_length_fit, num_electrode, q_mcmc, level_const
    ):

        # Hyper-parameter setting
        s_zeta_sq = np.ones([num_electrode]) * 1
        mu_zeta_prior = np.zeros([num_electrode, n_length_fit, 1]) + 0.5
        lower = np.zeros([num_electrode, n_length_fit])
        upper = np.ones_like(lower)

        [rho_set, rho_level, pres_chky_t_set_init,
         pres_chky_t_set, logdet_pres_chky_set] = self.produce_pre_compute_rhos(
            q_mcmc, seq_length_fit, level_const
        )

        index_point = np.linspace(-1, 1, num=n_length_fit)[:, np.newaxis]
        phi_theta_mat = np.eye(2 * n_length_fit)[np.newaxis, ...]
        theta_prior_mean = np.zeros([num_electrode, 2 * n_length_fit, 1])

        # Fix parameter
        eta_fix = np.zeros([num_electrode, seq_length_fit, 1])
        s_theta_sq_fix = np.ones([num_electrode, 2])
        s_theta_sq_fix_mat = self.generate_s_theta_sq_mat(s_theta_sq_fix[:, 0], s_theta_sq_fix[:, 1])

        theta_tar_pres = []
        theta_ntar_pres = []
        theta_prior_pres = []

        for e in range(num_electrode):
            _, theta_tar_pres_iter_e = self.create_kernel_mat_complex(
                s_tar[e], index_point, kernel_option,
                gamma=gamma_val[e], scale_sine=scale_sine[e], periodicity=periodicity[e]
            )
            _, theta_ntar_pres_iter_e = self.create_kernel_mat_complex(
                s_ntar[e], index_point, kernel_option,
                gamma=gamma_val[e], scale_sine=scale_sine[e], periodicity=periodicity[e]
            )
            # print('theta_tar_pres_iter_e has shape {}'.format(theta_tar_pres_iter_e.shape))
            theta_tar_pres.append(theta_tar_pres_iter_e)
            theta_ntar_pres.append(theta_ntar_pres_iter_e)

            theta_prior_pres_iter_e = linalg.block_diag(
                theta_tar_pres_iter_e / var_tar[e],
                theta_ntar_pres_iter_e / var_ntar[e]
            )
            theta_prior_pres.append(theta_prior_pres_iter_e)
        theta_tar_pres = np.stack(theta_tar_pres, axis=0)
        theta_ntar_pres = np.stack(theta_ntar_pres, axis=0)
        theta_prior_pres = np.stack(theta_prior_pres, axis=0)

        mcmc_hyper_param_dict = {
            's_zeta_sq': s_zeta_sq,
            'mu_zeta_prior': mu_zeta_prior,
            'lower': lower,
            'upper': upper,
            'rho_set': rho_set,
            'rho_level': rho_level,
            'pres_chky_t_set': pres_chky_t_set,
            'pres_chky_t_set_init': pres_chky_t_set_init,
            'logdet_pres_chky_set': logdet_pres_chky_set,
            'eta_fix': eta_fix,
            's_theta_sq_fix': s_theta_sq_fix,
            's_theta_sq_fix_mat': s_theta_sq_fix_mat,
            'phi_theta_mat': phi_theta_mat,
            'theta_prior_mean': theta_prior_mean,
            'theta_tar_pres': theta_tar_pres,
            'theta_ntar_pres': theta_ntar_pres,
            'theta_prior_pres': theta_prior_pres
        }

        return mcmc_hyper_param_dict

    def standard_mcmc_one_step(
            self, zeta_iter, s_x_sq_iter, pres_chky_t_iter,
            signal_x, var_tar, var_ntar,
            s_theta_sq_fix_mat,
            theta_tar_pres, theta_ntar_pres, theta_prior_pres,
            dt_mat, phi_theta_mat, theta_prior_mean,
            lower, upper, burn_in_samples, s_zeta_sq, mu_zeta_prior,
            pres_chky_t_set, logdet_pres_chky_set, pres_chky_t_set_init,
            s_x_sq_a, s_x_sq_b,
            letter_dim_fit, repet_num, seq_length_fit, n_length_fit, seq_bool
    ):

        # Here, we don't use Mercer's Theorem so s_theta_sq_fix is 1 by default.
        s_theta_sq_fix = np.ones([self.num_electrode, 2])

        # Update theta (beta in full kernel matrix)
        theta_iter = self.update_theta(
            zeta_iter, s_theta_sq_fix_mat, s_x_sq_iter,
            pres_chky_t_iter, theta_prior_pres,
            signal_x, dt_mat, phi_theta_mat, theta_prior_mean
        )

        # Update zeta
        zeta_iter, _, _ = self.update_zeta(
            theta_iter, s_theta_sq_fix_mat, s_x_sq_iter,
            pres_chky_t_iter,
            signal_x, dt_mat, phi_theta_mat, lower, upper,
            burn_in_samples, s_zeta_sq, mu_zeta_prior
        )

        # Update alpha and beta curves
        # Since theta_iter is always updated, we don't place bool variable here
        alpha_iter, beta_iter = self.compute_beta_select_from_theta(
            theta_iter, s_theta_sq_fix, zeta_iter, phi_theta_mat
        )
        alpha_tar_iter, alpha_ntar_iter = np.split(alpha_iter, [n_length_fit], axis=1)

        # Update rho and corr matrix
        # if arg_rho_bool:
        arg_rho_iter = self.update_rho_t(
            beta_iter, s_x_sq_iter, pres_chky_t_set, logdet_pres_chky_set,
            signal_x, dt_mat, letter_dim_fit, repet_num, seq_bool
        )
        pres_chky_t_iter = pres_chky_t_set_init[arg_rho_iter, ...]

        # Update sigma_x (with closed form)
        # if s_x_sq_bool:
        s_x_sq_iter = self.update_s_x_sq(
            beta_iter, pres_chky_t_iter,
            dt_mat, signal_x, letter_dim_fit, 1,
            s_x_sq_a, s_x_sq_b, seq_bool, seq_length_fit
        )

        var_tar = var_tar[:, np.newaxis, np.newaxis]
        var_ntar = var_ntar[:, np.newaxis, np.newaxis]
        log_lkd_tar = np.linalg.slogdet(theta_tar_pres / var_tar)[1] / 2 - 1 / (2 * var_tar) * \
                      np.matmul(np.transpose(alpha_tar_iter, axes=[0, 2, 1]),
                                np.matmul(theta_tar_pres, alpha_tar_iter))
        log_lkd_ntar = np.linalg.slogdet(theta_ntar_pres / var_ntar)[1] / 2 - 1 / (2 * var_ntar) * \
                       np.matmul(np.transpose(alpha_ntar_iter, axes=[0, 2, 1]),
                                 np.matmul(theta_ntar_pres, alpha_ntar_iter))

        x_diff_iter = signal_x - np.matmul(dt_mat, beta_iter[:, np.newaxis, ...])
        pr_x_diff = np.matmul(pres_chky_t_iter[:, np.newaxis, ...], x_diff_iter)
        q_sum = - np.sum(
            np.linalg.norm(pr_x_diff, ord=2, axis=2, keepdims=True) ** 2, axis=(-3, -2, -1)
        ) / (2 * s_x_sq_iter)
        logdet_pres_chky_iter = np.linalg.slogdet(pres_chky_t_iter)[1] / np.sqrt(s_x_sq_iter)

        log_lkd_iter = letter_dim_fit * logdet_pres_chky_iter + q_sum + \
                       np.squeeze(log_lkd_tar + log_lkd_ntar, axis=(-2, -1))

        return [theta_iter, zeta_iter, alpha_iter, beta_iter,
                arg_rho_iter, s_x_sq_iter, log_lkd_iter]

    def standard_mcmc_one_step_multi(
            self, zeta_iter, sigma_s_sq_iter, corr_s_iter, arg_rho_iter,
            cov_t_set, signal_x, dt_mat,
            theta_prior_pres, lower, upper, burn_in_samples,
            zeta_prior_mean, zeta_prior_pres, alpha_s, beta_s,
            corr_s_fix=None
    ):
        channel_dim, seq_num, seq_length, _ = signal_x.shape
        cov_t_iter = cov_t_set[arg_rho_iter]
        pres_t_iter = self.compute_matrix_inv_cholesky(cov_t_iter)
        cov_s_iter = sigma_s_sq_iter * corr_s_iter
        pres_s_iter = self.compute_matrix_inv_cholesky(cov_s_iter)
        pres_s_t_iter = np.kron(pres_s_iter, pres_t_iter)

        alpha_iter = self.update_theta_multi(
            zeta_iter, pres_s_t_iter, signal_x, dt_mat,
            theta_prior_pres, 2 * self.n_length
        )

        zeta_iter, _, _ = self.update_zeta_multi(
            alpha_iter, pres_s_t_iter, signal_x, dt_mat, zeta_prior_pres,
            zeta_prior_mean, lower, upper, burn_in_samples
        )

        # start_t = timer()
        zeta_mat_iter = self.create_select_mat_gp(zeta_iter, channel_dim)
        beta_iter = np.matmul(zeta_mat_iter, alpha_iter)

        if corr_s_fix is None:
            corr_s_iter = self.update_corr_s_cs(
                beta_iter, sigma_s_sq_iter, arg_rho_iter,
                cov_t_set, signal_x, dt_mat
            )
        else:
            # corr_s_iter = np.copy(corr_s_fix)
            corr_s_iter = self.update_corr_s_cs_general(
                beta_iter, sigma_s_sq_iter, arg_rho_iter, cov_t_set,
                signal_x, dt_mat, corr_s_fix
            )

        sigma_s_sq_iter = self.update_sigma_s_sq(
            beta_iter, corr_s_iter, cov_t_iter, signal_x, dt_mat, alpha_s, beta_s
        )
        cov_s_iter = sigma_s_sq_iter * corr_s_iter

        arg_rho_iter, log_lkd_iter = self.update_rho_t_multi(
            alpha_iter, zeta_iter, beta_iter, cov_s_iter, cov_t_set, signal_x, dt_mat
        )
        # end_t = timer()
        # print(end_t - start_t)  # Time in seconds

        return [beta_iter, zeta_iter, sigma_s_sq_iter, corr_s_iter, arg_rho_iter, log_lkd_iter]

    def standard_mcmc_summary(
            self, beta_mcmc, zeta_mcmc, arg_rho_mcmc, s_x_sq_mcmc, post_log_lkd_mcmc,
            s_tar, s_ntar, var_tar, var_ntar,
            eeg_dat_type, method_name, n_length_fit, repet_num_fit,
            channel_scenario, dec_factor, eeg_file_suffix, sim_dat_bool,
            cont_fit_bool, zeta_binary_threshold=None,
            beta_tar_true=None, beta_ntar_true=None, zeta_true=None, **kwargs
    ):
        mcmc_axis = 0
        if sim_dat_bool:
            sub_name = channel_scenario
        else:
            if isinstance(channel_scenario, int):
                sub_name = 'channel_' + str(channel_scenario + 1)  # use python index
            elif len(channel_scenario) == 1:
                sub_name = 'channel_' + str(channel_scenario[0] + 1)  # use python index

            else:
                channel_id_str = [str(e + 1) for e in channel_scenario]
                if len(channel_scenario) == 16:
                    sub_name = 'all_channels'
                else:
                    sub_name = 'channel_' + '_'.join(channel_id_str)

        self.create_method_folder(method_name, sub_name, sim_dat_bool)
        beta_tar_output, beta_ntar_output = np.split(beta_mcmc, [n_length_fit], axis=2)

        if cont_fit_bool:
            zeta_output = np.round(np.median(zeta_mcmc, axis=mcmc_axis), decimals=2)
            print('zeta_median = \n {}'.format(zeta_output))
            job_id_dec = 'continuous_down_{}_{}'.format(dec_factor, eeg_file_suffix)
        else:
            zeta_output = np.round(np.median(zeta_mcmc, axis=mcmc_axis), decimals=2)
            job_id_dec = 'binary_down_{}_{}_zeta_{}'.format(
                dec_factor, eeg_file_suffix, zeta_binary_threshold
            )

        # plot the mean curves
        self.visualize_selection_indicator(
            beta_tar_true, beta_ntar_true,
            beta_tar_output, beta_ntar_output,
            zeta_output, repet_num_fit, None,
            method_name, eeg_dat_type, sub_name,
            job_id_dec, sim_dat_bool, **kwargs
        )
        self.save_mcmc(
            s_x_sq_mcmc, arg_rho_mcmc,
            zeta_mcmc, zeta_true,
            beta_tar_output, beta_ntar_output,
            np.concatenate([s_tar, s_ntar]),
            np.concatenate([var_tar, var_ntar]),
            post_log_lkd_mcmc, repet_num_fit, method_name, eeg_dat_type,
            sub_name, job_id_dec, sim_dat_bool, **kwargs
        )

        return 'standard MCMC summary is done!'

    def standard_mcmc_binary_prepare(
            self, eeg_dat_type, method_name, repet_num_fit,
            channel_scenario, dec_factor, eeg_file_suffix, sim_dat_bool, **kwargs
    ):
        job_id_dec = 'continuous_down_{}_{}'.format(dec_factor, eeg_file_suffix)

        if sim_dat_bool:
            sub_name = channel_scenario
            [s_x_sq_mcmc, arg_rho_mcmc, zeta_mcmc, _,
             beta_tar_mcmc, beta_ntar_mcmc, _, _, log_lkd_mcmc] = self.import_mcmc(
                eeg_dat_type, method_name, repet_num_fit, sub_name, job_id_dec, sim_dat_bool, **kwargs
            )
        else:
            if isinstance(channel_scenario, int):
                sub_name = 'channel_' + str(channel_scenario + 1)
            elif len(channel_scenario) == 1:
                sub_name = 'channel_' + str(channel_scenario + 1)
            elif len(channel_scenario) == 16:
                sub_name = 'all_channels'
            else:
                sub_name = 'channel_' + '_'.join([str(e+1) for e in channel_scenario])
            [s_x_sq_mcmc, arg_rho_mcmc, zeta_mcmc,
             beta_tar_mcmc, beta_ntar_mcmc, _, _, log_lkd_mcmc] = self.import_mcmc(
                eeg_dat_type, method_name, repet_num_fit, sub_name, job_id_dec, sim_dat_bool, **kwargs
            )

        # mcmc_axis = 0
        # zeta_median = np.median(zeta_mcmc, axis=mcmc_axis)
        # print('zeta_median = \n {}'.format(np.round(zeta_median, decimals=3)))
        # zeta_binary_fix = (zeta_median >= zeta_binary_threshold) * 1
        # print('zeta_binary_fix = \n {}, {} selected'.format(
        #     zeta_binary_fix, np.sum(zeta_binary_fix, axis=1))
        # )
        # s_x_sq_fix = np.mean(s_x_sq_mcmc, axis=mcmc_axis)
        # arg_rho_fix = stats.mode(arg_rho_mcmc, axis=mcmc_axis)[0][0, :]
        # pres_chky_t_fix = pres_chky_t_set_init[arg_rho_fix, ...]

        return [arg_rho_mcmc, beta_tar_mcmc, beta_ntar_mcmc,
                s_x_sq_mcmc, zeta_mcmc, log_lkd_mcmc]

    def standard_mcmc_screen_train_single_seq(
            self, eta_fix, beta_comb_mcmc, s_x_sq_mcmc, arg_rho_mcmc, r_mat_neghalf,
            eeg_type_fit, signal_x_fit, pres_chky_t_set_init, d_x_mat, mcmc_iter_num, thining,
            repet_num_fit, reshape_option, normal_bool, t_df=None
    ):
        letter_dim_fit = 1  # by default in single_seq calculation
        eta_fix = eta_fix[:, np.newaxis, ...]
        super_seq_len = signal_x_fit.shape[2]

        t_mat_fit = self.create_transform_mat(
            eeg_type_fit, letter_dim_fit, repet_num_fit, reshape_option
        )
        dt_mat_fit = (d_x_mat @ t_mat_fit)[np.newaxis, ...]
        log_lkd_fit = []
        for i in range(0, mcmc_iter_num, thining):
            pres_chky_t_iter = pres_chky_t_set_init[arg_rho_mcmc[i, :], ...]
            beta_comb_iter = beta_comb_mcmc[i, :, np.newaxis, ...]
            s_x_sq_iter = s_x_sq_mcmc[i, :]
            pr_x_mat_diff_fit_iter = pres_chky_t_iter @ (r_mat_neghalf * (signal_x_fit - dt_mat_fit @ beta_comb_iter - eta_fix))
            log_quad_sum_fit_iter = np.sum(np.linalg.norm(pr_x_mat_diff_fit_iter, ord=2, axis=-2, keepdims=True) ** 2,
                                      axis=(-2, -1))  # (channel_dim, letter_dim)
            if normal_bool:
                log_quad_sum_fit_iter = np.sum(-1 / (2 * s_x_sq_iter[:, np.newaxis]) * log_quad_sum_fit_iter)
            else:
                log_quad_sum_fit_iter = -(t_df + super_seq_len) / 2 * np.sum(np.log(1 + 1 / t_df * log_quad_sum_fit_iter))
            log_lkd_fit.append(log_quad_sum_fit_iter)

        log_lkd_fit = np.stack(log_lkd_fit, axis=0)
        log_lkd_fit_mean = np.mean(log_lkd_fit, axis=0)

        return log_lkd_fit_mean

    def standard_mcmc_screen(
            self, signal_x_fit, eeg_type_fit, eeg_code_fit,
            eta_fix, beta_mcmc, s_x_sq_mcmc, arg_rho_mcmc,
            r_mat_neghalf_fix, pres_chky_t_set_init, d_mat, mcmc_sample_num,
            quantile_val, thinning, letter_dim, repet_num_fit, dec_factor,
            eeg_dat_type, method_name, scenario_name, eeg_file_suffix, normal_bool
    ):
        # for single-seq fit with screening
        # letter_dim = original letter_dim * original repet_num_fit
        # repet_num_fit = 1 by default
        log_lkd_single_seq = []
        eeg_type_fit_2d = np.reshape(eeg_type_fit, [letter_dim, self.num_rep])
        eeg_code_fit_2d = np.reshape(eeg_code_fit, [letter_dim, self.num_rep])
        for seq_id in range(letter_dim):
            log_lkd_seq_id = self.standard_mcmc_screen_train_single_seq(
                eta_fix, beta_mcmc, s_x_sq_mcmc, arg_rho_mcmc, r_mat_neghalf_fix,
                eeg_type_fit_2d[seq_id, :], signal_x_fit[:, seq_id, np.newaxis, ...],
                pres_chky_t_set_init, d_mat, mcmc_sample_num, thinning,
                repet_num_fit, eeg_dat_type, normal_bool
            )
            log_lkd_single_seq.append(log_lkd_seq_id)
        log_lkd_single_seq = np.stack(log_lkd_single_seq, axis=0)
        # print('log_lkd_single_seq = {}'.format(log_lkd_single_seq))
        low_quantile_value = np.round(np.quantile(log_lkd_single_seq, quantile_val, axis=0), decimals=3)
        print('low_quantile_{}_value = {}'.format(quantile_val, low_quantile_value))

        eeg_type_fit_screen = eeg_type_fit_2d[log_lkd_single_seq > low_quantile_value, :]
        eeg_code_fit_screen = eeg_code_fit_2d[log_lkd_single_seq > low_quantile_value, :]
        signal_x_fit_screen = signal_x_fit[:, log_lkd_single_seq > low_quantile_value, ...]

        # save the new dataset
        screen_mat_dir = '{}/{}/{}/{}_eeg_dat_down_{}_from_{}_screen.mat'.format(
            self.parent_eeg_output_path, method_name, scenario_name,
            self.sub_folder_name, dec_factor, eeg_file_suffix
        )

        sio.savemat(screen_mat_dir,
                    {
                        'eeg_signals': signal_x_fit_screen,
                        'eeg_code': eeg_code_fit_screen,
                        'eeg_type': eeg_type_fit_screen,
                        'log_lkd_single_seq': log_lkd_single_seq,
                        'low_quantile_value': low_quantile_value
                    })
        print('eeg_signals_screen has shape {}'.format(signal_x_fit_screen.shape))
        print('eeg_code_screen has shape {}'.format(eeg_code_fit_screen.shape))
        print('eeg_type_screen has shape {}'.format(eeg_type_fit_screen.shape))

        return [signal_x_fit_screen, eeg_type_fit_screen, eeg_code_fit_screen,
                log_lkd_single_seq, low_quantile_value]

    def standard_mcmc(
            self, signal_x_fit, eeg_type_fit,
            s_tar, s_ntar, var_tar, var_ntar,
            mcmc_hyper_params, s_x_sq_a, s_x_sq_b, mcmc_iter_num, burn_in_num,
            letter_dim_fit, repet_num_fit, n_length_fit, seq_length_fit,
            channel_scenario, num_electrode, eeg_dat_type,
            dec_factor, method_name, eeg_file_suffix,
            seq_bool, sim_dat_bool, cont_fit_bool, zeta_0,
            beta_tar_true=None, beta_ntar_true=None, zeta_true=None
    ):

        # import hyper-params
        s_zeta_sq = mcmc_hyper_params['s_zeta_sq']
        mu_zeta_prior = mcmc_hyper_params['mu_zeta_prior']
        lower = mcmc_hyper_params['lower']
        upper = mcmc_hyper_params['upper']
        rho_set = mcmc_hyper_params['rho_set']
        rho_level = mcmc_hyper_params['rho_level']
        pres_chky_t_set = mcmc_hyper_params['pres_chky_t_set']
        pres_chky_t_set_init = mcmc_hyper_params['pres_chky_t_set_init']
        logdet_pres_chky_set = mcmc_hyper_params['logdet_pres_chky_set']
        # eta_fix = mcmc_hyper_params['eta_fix']
        s_theta_sq_fix_mat = mcmc_hyper_params['s_theta_sq_fix_mat']
        phi_theta_mat = mcmc_hyper_params['phi_theta_mat']
        theta_prior_mean = mcmc_hyper_params['theta_prior_mean']
        theta_tar_pres = mcmc_hyper_params['theta_tar_pres']
        theta_ntar_pres = mcmc_hyper_params['theta_ntar_pres']
        theta_prior_pres = mcmc_hyper_params['theta_prior_pres']

        d_mat = self.create_design_mat_gen_bayes_seq(repet_num_fit)
        t_mat = self.create_transform_mat(
            eeg_type_fit, letter_dim_fit, repet_num_fit, eeg_dat_type
        )
        dt_mat = d_mat @ t_mat
        dt_mat = dt_mat[np.newaxis, ...]
        mcmc_axis = 0

        # used for zeta iteration, large number may result in long converging time.
        burn_in_samples = 0
        num_interval = int(mcmc_iter_num / 10)
        # letter_dim_fit/repet_num_fit are for super-seq based, letter_dim_single_seq = letter_dim_fit * repet_num_fit
        theta_mcmc = []
        zeta_mcmc = []
        alpha_mcmc = []
        beta_mcmc = []
        s_x_sq_mcmc = []
        arg_rho_mcmc = []
        log_lkd_mcmc = []

        zeta_iter = np.round(np.random.uniform(
            low=0.2, high=0.8, size=(num_electrode, n_length_fit)), decimals=3
        )
        s_x_sq_iter = np.random.gamma(shape=1, scale=1, size=num_electrode)
        arg_rho_iter = np.random.randint(low=0, high=rho_level, size=num_electrode)
        pres_chky_t_iter = pres_chky_t_set_init[arg_rho_iter, ...]

        if cont_fit_bool:
            print('First-stage MCMC begins.')
            for k in range(mcmc_iter_num):

                [theta_iter, zeta_iter, alpha_iter, beta_iter,
                 arg_rho_iter, s_x_sq_iter, log_lkd_iter] = self.standard_mcmc_one_step(
                    zeta_iter, s_x_sq_iter, pres_chky_t_iter,
                    signal_x_fit, var_tar, var_ntar,
                    s_theta_sq_fix_mat,
                    theta_tar_pres, theta_ntar_pres, theta_prior_pres,
                    dt_mat, phi_theta_mat, theta_prior_mean,
                    lower, upper, burn_in_samples, s_zeta_sq, mu_zeta_prior,
                    pres_chky_t_set, logdet_pres_chky_set, pres_chky_t_set_init,
                    s_x_sq_a, s_x_sq_b, letter_dim_fit, repet_num_fit,
                    seq_length_fit, n_length_fit, seq_bool
                )

                if k % num_interval == 0:
                    print('gibbs index = {}'.format(k + 1))
                    print('zeta_post = \n {}'.format(np.round(zeta_iter, decimals=3)))
                    print('s_x_sq = {}'.format(s_x_sq_iter))
                    print('rho = {}'.format([rho_set[i] for i in arg_rho_iter]))
                    beta_tar_iter, beta_ntar_iter = np.split(beta_iter, [n_length_fit], axis=1)
                    print('beta_tar_iter = {}, \n beta_ntar_iter = {} \n'.format(
                        np.round(beta_tar_iter[0, :, 0], decimals=3),
                        np.round(beta_ntar_iter[0, :, 0], decimals=3))
                    )
                    print('log_lkd_iter = {}'.format(log_lkd_iter))

                theta_mcmc.append(theta_iter)
                zeta_mcmc.append(zeta_iter)
                alpha_mcmc.append(alpha_iter)
                beta_mcmc.append(beta_iter)
                arg_rho_mcmc.append(arg_rho_iter)
                s_x_sq_mcmc.append(s_x_sq_iter)
                log_lkd_mcmc.append(log_lkd_iter)

            beta_mcmc = np.stack(beta_mcmc, axis=mcmc_axis)[burn_in_num:, ...]
            log_lkd_mcmc = np.stack(log_lkd_mcmc, axis=mcmc_axis)[burn_in_num:, :]
            zeta_mcmc = np.stack(zeta_mcmc, axis=mcmc_axis)[burn_in_num:, ...]
            arg_rho_mcmc = np.stack(arg_rho_mcmc, axis=mcmc_axis)[burn_in_num:, :]
            s_x_sq_mcmc = np.stack(s_x_sq_mcmc, axis=mcmc_axis)[burn_in_num:, :]

        else:
            print('Second-stage MCMC begins')
            # Still use the MCMC from the model fit of continuous zetas
            [arg_rho_mcmc, beta_tar_mcmc, beta_ntar_mcmc,
             s_x_sq_mcmc, zeta_mcmc, log_lkd_mcmc] = self.standard_mcmc_binary_prepare(
                eeg_dat_type, method_name, repet_num_fit,
                # zeta_0, pres_chky_t_set_init,
                channel_scenario, dec_factor, eeg_file_suffix, sim_dat_bool
            )
            print('beta_tar_mcmc and beta_ntar_mcmc have shape {}, {}'.format(
                beta_tar_mcmc.shape, beta_ntar_mcmc.shape)
            )
            print('zeta_mcmc has shape {}'.format(zeta_mcmc.shape))
            beta_tar_mcmc_2 = np.copy(beta_tar_mcmc)
            beta_ntar_mcmc_2 = np.copy(beta_ntar_mcmc)
            for mcmc_id in range(mcmc_iter_num - burn_in_num):
                for tau in range(n_length_fit):
                    if zeta_mcmc[mcmc_id, 0, tau] < zeta_0:
                        beta_mean_iter = (beta_tar_mcmc[mcmc_id, :, tau, :] + beta_ntar_mcmc[mcmc_id, :, tau, :]) / 2
                        beta_tar_mcmc_2[mcmc_id, :, tau, :] = np.copy(beta_mean_iter)
                        beta_ntar_mcmc_2[mcmc_id, :, tau, :] = np.copy(beta_mean_iter)

            beta_mcmc = np.concatenate([beta_tar_mcmc_2, beta_ntar_mcmc_2], axis=2)

        self.standard_mcmc_summary(
            beta_mcmc, zeta_mcmc, arg_rho_mcmc, s_x_sq_mcmc, log_lkd_mcmc,
            s_tar, s_ntar, var_tar, var_ntar,
            eeg_dat_type, method_name,
            n_length_fit, repet_num_fit, channel_scenario, dec_factor, eeg_file_suffix,
            sim_dat_bool, cont_fit_bool, zeta_0,
            beta_tar_true, beta_ntar_true, zeta_true
        )

        return 'Entire MCMC is done!'

    def standard_mcmc_multi(
            self, signal_x_fit, eeg_type_fit,
            s_tar, s_ntar, var_tar, var_ntar,  # gamma_val, s_sine, periodicity,
            mcmc_iter_num, burn_in_num, num_interval,
            eeg_dat_type, rho_level_num, q_mcmc,
            theta_prior_pres, lower, upper,
            zeta_prior_mean, zeta_prior_pres,
            alpha_s, beta_s, repet_num_fit, channel_scenario,
            dec_factor, method_name, eeg_file_suffix,
            cont_fit_bool, sim_dat_bool, zeta_0,
            beta_tar_true=None, beta_ntar_true=None, zeta_true=None,
            signal_x_corr_s=None, **kwargs
    ):
        channel_dim, seq_num, seq_length, _ = signal_x_fit.shape
        d_mat_fit = self.create_design_mat_gen_bayes_seq(repet_num_fit)
        t_mat_fit = self.create_transform_mat(
            eeg_type_fit, seq_num, repet_num_fit, eeg_dat_type
        )
        dt_mat_fit = np.matmul(d_mat_fit, t_mat_fit)
        dt_mat_fit = dt_mat_fit[np.newaxis, ...]
        rho_set, rho_level, _, _, _ = self.produce_pre_compute_rhos(
            q_mcmc, seq_length, rho_level_num
        )
        cov_t_set = [self.create_ar2_pres_mat(1, rho_set[i], seq_length)[0]
                     for i in np.arange(rho_level)]
        burn_in_samples = 0

        # initialize parameters
        zeta_mcmc = []
        beta_mcmc = []
        cov_s_mcmc = []
        arg_rho_mcmc = []
        log_lkd_mcmc = []

        zeta_iter = np.random.uniform(
            low=0, high=1, size=(channel_dim, self.n_length)
        )
        corr_s_iter = np.eye(channel_dim) * 10
        sigma_s_sq_iter = 1 / np.random.gamma(shape=alpha_s, scale=1/beta_s, size=1)
        arg_rho_iter = 0
        mcmc_axis = 0

        if cont_fit_bool:
            print('First-stage MCMC begins.')

            for k in range(mcmc_iter_num):

                [beta_iter, zeta_iter, sigma_s_sq_iter,
                 corr_s_iter, arg_rho_iter, log_lkd_iter] = self.standard_mcmc_one_step_multi(
                    zeta_iter, sigma_s_sq_iter, corr_s_iter, arg_rho_iter, cov_t_set,
                    signal_x_fit, dt_mat_fit, theta_prior_pres, lower, upper,
                    burn_in_samples, zeta_prior_mean, zeta_prior_pres,
                    alpha_s, beta_s, corr_s_fix=signal_x_corr_s
                )
                cov_s_iter = sigma_s_sq_iter * corr_s_iter
                if k % num_interval == 0:
                    print('gibbs index = {}'.format(k + 1))
                    print('zeta_post = \n {}'.format(np.round(zeta_iter, decimals=2)))
                    print('rho = {}'.format(rho_set[arg_rho_iter]))
                    print('corr_s_iter = {}, s_s_sq = {}'.format(corr_s_iter, sigma_s_sq_iter))
                    beta_tar_iter, beta_ntar_iter = np.split(beta_iter, [self.n_length], axis=1)
                    print('beta_tar_iter = {}, \n beta_ntar_iter = {} \n'.format(
                        np.round(beta_tar_iter[..., 0], decimals=2),
                        np.round(beta_ntar_iter[..., 0], decimals=2))
                    )
                    print('log_lkd_iter = {}'.format(log_lkd_iter))

                zeta_mcmc.append(zeta_iter)
                beta_mcmc.append(beta_iter)
                arg_rho_mcmc.append(arg_rho_iter)
                cov_s_mcmc.append(cov_s_iter)
                log_lkd_mcmc.append(log_lkd_iter)

            beta_mcmc = np.stack(beta_mcmc, axis=mcmc_axis)[burn_in_num:, ...]
            log_lkd_mcmc = np.stack(log_lkd_mcmc, axis=mcmc_axis)[burn_in_num:]
            zeta_mcmc = np.stack(zeta_mcmc, axis=mcmc_axis)[burn_in_num:, ...]
            arg_rho_mcmc = np.stack(arg_rho_mcmc, axis=mcmc_axis)[burn_in_num:]
            cov_s_mcmc = np.stack(cov_s_mcmc, axis=mcmc_axis)[burn_in_num:, ...]

        else:
            print('Second-stage MCMC begins')

            [arg_rho_mcmc, beta_tar_mcmc, beta_ntar_mcmc,
             cov_s_mcmc, zeta_mcmc, log_lkd_mcmc] = self.standard_mcmc_binary_prepare(
                eeg_dat_type, method_name, repet_num_fit,
                channel_scenario, dec_factor, eeg_file_suffix, sim_dat_bool, **kwargs
            )
            beta_tar_mcmc_2 = np.copy(beta_tar_mcmc)
            beta_ntar_mcmc_2 = np.copy(beta_ntar_mcmc)
            for mcmc_id in range(mcmc_iter_num - burn_in_num):
                for tau in range(self.n_length):
                    if zeta_mcmc[mcmc_id, 0, tau] < zeta_0:
                        beta_mean_iter = (beta_tar_mcmc[mcmc_id, :, tau, :] + beta_ntar_mcmc[mcmc_id, :, tau, :]) / 2
                        beta_tar_mcmc_2[mcmc_id, :, tau, :] = np.copy(beta_mean_iter)
                        beta_ntar_mcmc_2[mcmc_id, :, tau, :] = np.copy(beta_mean_iter)

            beta_mcmc = np.concatenate([beta_tar_mcmc_2, beta_ntar_mcmc_2], axis=2)

        # self.standard_mcmc_summary(
        #     beta_mcmc, zeta_mcmc, arg_rho_mcmc, cov_s_mcmc, log_lkd_mcmc,
        #     s_tar, s_ntar, var_tar, var_ntar,
        #     eeg_dat_type, method_name,
        #     self.n_length, repet_num_fit, channel_scenario, dec_factor, eeg_file_suffix,
        #     sim_dat_bool, cont_fit_bool, zeta_0,
        #     beta_tar_true, beta_ntar_true, zeta_true, **kwargs
        # )

        return 'Entire MCMC is done!'

    def compute_single_letter_single_seq_log_lkd(
            self, beta_comb_mcmc, s_x_sq_mcmc, pres_chky_t_fix,
            eeg_code_3d, signal_x, letter_dim, repet_num_pred, d_mat, mcmc_num, thinning,
            eeg_dat_type, normal_noise_bool, log_prior_prob
    ):
        r"""
        :param beta_comb_mcmc: (mcmc_num, channel_dim, 2 * n_length_fit, 1)
        :param s_x_sq_mcmc:
        :param pres_chky_t_fix:
        :param eeg_code_3d: (letter_dim, repet_num_pred, 12)
        :param signal_x: (1, letter_dim, repet_num_pred, single_seq_length, 1)
        :param letter_dim: integer
        :param repet_num_pred: integer
        :param d_mat:
        :param mcmc_num: integer
        :param thinning:
        :param eeg_dat_type:
        :param normal_noise_bool:
        :param log_prior_prob:
        :return:
            log_lkd_pred_36, (letter_dim, len(repet_pred_ids), channel_dim, mcmc_num, 36)
        """

        single_letter_dim = 1
        single_rep_dim = 1
        channel_dim = beta_comb_mcmc.shape[1]
        mcmc_thin_num = len(np.arange(0, mcmc_num, thinning))
        log_lkd_pred_36 = np.zeros([letter_dim, repet_num_pred,
                                    channel_dim, mcmc_thin_num,
                                    self.letter_table_sum])

        for letter_id in range(letter_dim):
            print('letter_id = {}'.format(letter_id + 1))
            for repet_num_id in range(repet_num_pred):
                for i in range(0, mcmc_num, thinning):
                    log_lkd_test_iter = self.bayes_generate_pred_latency_full(
                        beta_comb_mcmc[i, ...], s_x_sq_mcmc[i, :], pres_chky_t_fix,
                        eeg_code_3d[letter_id, np.newaxis, repet_num_id, :],
                        signal_x[:, letter_id, np.newaxis, repet_num_id, :, :],
                        d_mat, single_letter_dim, single_rep_dim,
                        eeg_dat_type, normal_noise_bool, 0, log_prior_prob
                    )
                    log_lkd_pred_36[letter_id, repet_num_id, :, int(i / thinning), :] = log_lkd_test_iter

        return log_lkd_pred_36

    def compute_single_letter_single_seq_log_lkd_multi(
            self, beta_comb_mcmc, cov_s_mcmc, arg_rho_mcmc, cov_t_set,
            eeg_code_3d, signal_x, letter_dim, repet_num_pred, d_mat, mcmc_num, thinning,
            eeg_dat_type, log_prior_prob
    ):
        channel_dim, _, _, seq_length, _ = signal_x.shape
        mcmc_thin_num = len(np.arange(0, mcmc_num, thinning))
        log_lkd_pred_36 = np.zeros([letter_dim, repet_num_pred, mcmc_thin_num, self.letter_table_sum])

        for letter_id in range(letter_dim):
            print('letter_id = {}'.format(letter_id+1))
            for repet_num_id in range(repet_num_pred):
                for i in range(0, mcmc_num, thinning):
                    eeg_code_sub = eeg_code_3d[letter_id, repet_num_id, :]
                    signal_x_sub = signal_x[:, letter_id, repet_num_id, np.newaxis, :, :]
                    log_lkd_pred_36_sub = self.bayes_generate_pred_latency_full_multi(
                        beta_comb_mcmc[i, ...], cov_s_mcmc[i, ...], arg_rho_mcmc[i], cov_t_set,
                        eeg_code_sub, signal_x_sub, d_mat, eeg_dat_type, log_prior_prob
                    )
                    log_lkd_pred_36[letter_id, repet_num_id, int(i/thinning), :] = log_lkd_pred_36_sub

        return log_lkd_pred_36

    def standard_compute_pred_single_seq(
            self, beta_comb_mcmc, s_x_sq_mcmc, pres_chky_t_fix,
            eeg_code_pred_3d, signal_x_pred, d_mat,
            letter_dim, repet_num_pred, repet_num_fit, mcmc_num, thinning,
            eeg_dat_type, method_name, channel_name, file_subscript, pred_train_bool
    ):
        normal_noise_bool = True
        log_prior_prob = np.log(1 / self.letter_table_sum * np.ones([self.letter_table_sum]))
        # equal prior probability

        log_lkd_pred_36 = self.compute_single_letter_single_seq_log_lkd(
            beta_comb_mcmc, s_x_sq_mcmc, pres_chky_t_fix,
            eeg_code_pred_3d, signal_x_pred, letter_dim, repet_num_pred, d_mat,
            mcmc_num, thinning, eeg_dat_type, normal_noise_bool, log_prior_prob,

        )
        print('log_lkd_pred_36 has shape {}'.format(log_lkd_pred_36.shape))

        if pred_train_bool:
            train_test_label = 'train'
        else:
            train_test_label = 'test'

        self.save_bayes_single_seq_log_lkd(
            repet_num_fit, repet_num_pred, train_test_label,
            method_name + 'Pred', channel_name,
            eeg_dat_type, file_subscript, 'single_seq_log_lkd', log_lkd_pred_36
        )

        '''
        # 5 out of 7 or 5 out of 8
        _, _, channel_dim, _, _ = log_lkd_pred_36.shape
        seq_fix = 5
        n_choose_k = np.array(list(itl.product(*[(0, 1) for i in range(repet_num_pred)])))
        n_choose_k = np.copy(n_choose_k[np.sum(n_choose_k, axis=-1) == seq_fix, :])  # (21, 7) or (56, 8)

        n_k_val, _ = n_choose_k.shape
        # print('n_choose_k array has shape {}'.format(n_choose_k.shape))
        log_lkd_cum = np.zeros([letter_dim, n_k_val, channel_dim, mcmc_thin_num, self.letter_table_sum])
        for n_k_id in range(n_k_val):
            log_lkd_cum[:, n_k_id, ...] = np.sum(log_lkd_pred_36[:, np.where(n_choose_k[n_k_id, :] == 1)[0], ...], axis=1)

        prob_cum = self.exp_normalize(log_lkd_cum)
        prob_cum = np.sum(prob_cum, axis=2)  # (letter_dim, repet_num_pred, mcmc_thin_num, 36)
        prob_cum_letter_arg = np.argsort(prob_cum, axis=-1)[:, :, :, ::-1]

        prob_cum_rs = np.reshape(prob_cum, [letter_dim, n_k_val, mcmc_thin_num, 6, 6])
        prob_cum_row = np.sum(prob_cum_rs, axis=-1)  # (letter_dim, repet_num_pred, mcmc_thin_num, 6)
        prob_cum_col = np.sum(prob_cum_rs, axis=-2)  # (letter_dim, repet_num_pred, mcmc_thin_num, 6)
        prob_cum_row_arg = np.argsort(prob_cum_row, axis=-1)[:, :, :, ::-1]
        prob_cum_col_arg = np.argsort(prob_cum_col, axis=-1)[:, :, :, ::-1]

        bayes_result_dict = {
            'prob_cum': prob_cum,
            'prob_cum_row': prob_cum_row,
            'prob_cum_col': prob_cum_col,
            'letter_cum_rank': prob_cum_letter_arg,
            'row_cum_rank': prob_cum_row_arg,
            'col_cum_rank': prob_cum_col_arg,
            'scale': scale_opt,
            'var': var_opt,
            'sample_num': mcmc_thin_num
        }

        if pred_train_bool:
            self.save_bayes_results(
                bayes_result_dict, repet_num_fit, n_k_val, method_name,
                'single_seq_train', target_letters, target_letter_rows, target_letter_cols,
                channel_name, file_subscript, sim_dat_bool
            )
        else:
            self.save_bayes_results(
                bayes_result_dict, repet_num_fit, n_k_val, method_name,
                'single_seq_test', target_letters, target_letter_rows, target_letter_cols,
                channel_name, file_subscript, sim_dat_bool
            )
        '''

        return 'Single Log Likelihood Has Been Saved!'

    def standard_pred_cum_seq(
            self, scale_opt, var_opt,
            letter_dim, repet_num_pred, repet_num_fit, mcmc_num, thinning,
            eeg_dat_type, target_letters, target_letter_rows, target_letter_cols,
            method_name, channel_name, file_subscript, lkd_name_subscript,
            exclude_first_bool, pred_train_bool, sim_dat_bool, **kwargs
    ):
        if pred_train_bool:
            train_test_label = 'train'
        else:
            train_test_label = 'test'

        mcmc_thin_num = len(np.arange(0, mcmc_num, thinning))
        log_lkd = self.import_bayes_single_seq_log_lkd(
            repet_num_fit, repet_num_pred, train_test_label,
            method_name + 'Pred', channel_name,
            eeg_dat_type, file_subscript, lkd_name_subscript, sim_dat_bool,
            **kwargs
        )
        log_lkd_pred_36 = log_lkd['log_lkd']
        # print('temp has shape {}'.format(log_lkd_pred_36.shape))

        if exclude_first_bool:
            log_lkd_pred_36 = log_lkd_pred_36[:, 1:, ...]
            repet_num_pred = repet_num_pred - 1

        if log_lkd_pred_36.shape[2] == 1:
            # collapse the channel dimension
            log_lkd_cum = np.squeeze(np.cumsum(log_lkd_pred_36, axis=1), axis=2)
        else:
            log_lkd_cum = np.cumsum(log_lkd_pred_36, axis=1)

        print('log_lkd_cum has shape {}'.format(log_lkd_cum.shape))
        prob_cum = self.exp_normalize(log_lkd_cum)
        prob_cum_letter_arg = np.argsort(prob_cum, axis=-1)[:, :, :, ::-1]
        prob_cum_rs = np.reshape(prob_cum, [letter_dim, repet_num_pred, mcmc_thin_num, 6, 6])
        prob_cum_row = np.sum(prob_cum_rs, axis=-1)  # (letter_dim, repet_num_pred, mcmc_thin_num, 6)
        prob_cum_col = np.sum(prob_cum_rs, axis=-2)  # (letter_dim, repet_num_pred, mcmc_thin_num, 6)
        prob_cum_row_arg = np.argsort(prob_cum_row, axis=-1)[:, :, :, ::-1]
        prob_cum_col_arg = np.argsort(prob_cum_col, axis=-1)[:, :, :, ::-1]

        bayes_result_dict = {
            'prob_cum': prob_cum,
            'prob_cum_row': prob_cum_row,
            'prob_cum_col': prob_cum_col,
            'letter_cum_rank': prob_cum_letter_arg,
            'row_cum_rank': prob_cum_row_arg,
            'col_cum_rank': prob_cum_col_arg,
            'scale': scale_opt,
            'var': var_opt,
            'sample_num': mcmc_thin_num
        }

        if pred_train_bool:
            middle_name = 'single_seq_train'
        else:
            middle_name = 'single_seq_test'

        self.save_bayes_results(
            bayes_result_dict, repet_num_fit, repet_num_pred, method_name,
            middle_name, target_letters, target_letter_rows, target_letter_cols,
            channel_name, file_subscript, sim_dat_bool, **kwargs
        )

        return bayes_result_dict

    def standard_compute_pred_single_seq_multi(
            self, beta_comb_mcmc, cov_s_mcmc, arg_rho_mcmc, cov_t_set,
            eeg_code_pred_3d, signal_x_pred, d_mat,
            letter_dim, repet_num_pred, repet_num_fit, mcmc_num, thinning,
            eeg_dat_type, method_name, channel_name,
            file_subscript, pred_train_bool, sim_dat_bool, **kwargs
    ):
        log_prior_prob = np.log(1 / self.letter_table_sum * np.ones([self.letter_table_sum]))

        log_lkd_pred_36 = self.compute_single_letter_single_seq_log_lkd_multi(
            beta_comb_mcmc, cov_s_mcmc, arg_rho_mcmc, cov_t_set,
            eeg_code_pred_3d, signal_x_pred, letter_dim, repet_num_pred, d_mat, mcmc_num, thinning,
            eeg_dat_type, log_prior_prob
        )
        print('log_lkd_pred_36 has shape {}'.format(log_lkd_pred_36.shape))

        if pred_train_bool:
            train_test_label = 'train'
        else:
            train_test_label = 'test'

        self.save_bayes_single_seq_log_lkd(
            repet_num_fit, repet_num_pred, train_test_label,
            method_name + 'Pred', channel_name,
            eeg_dat_type, file_subscript, 'single_seq_log_lkd_multi',
            log_lkd_pred_36, sim_dat_bool, **kwargs
        )

        return 'Single Log Likelihood Multi Channel Has Been Saved!'


