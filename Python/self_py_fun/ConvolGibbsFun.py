import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.ConvolFun import *
from scipy import stats, linalg
plt.style.use('ggplot')
sns.set_context('notebook')


class ConvolGibbs:
    # Within-class global constant
    num_rep = 12
    flash_sum = 2
    non_flash_sum = 10
    DAT_TYPE = 'float32'

    def __init__(self, n_multiple, num_electrode, flash_and_pause_length,
                 mu_1_delta, mu_0_delta, sigma_sq_delta,
                 nu_1, nu_0, scale_1, scale_0,
                 alpha, beta, kappa,
                 letter_dim, trn_repetition):
        self.n_multiple = n_multiple
        self.num_electrode = num_electrode
        self.flash_and_pause_length = flash_and_pause_length
        self.mu_1_delta = mu_1_delta.astype(self.DAT_TYPE)
        self.mu_0_delta = mu_0_delta.astype(self.DAT_TYPE)
        self.sigma_sq_delta = sigma_sq_delta
        self.nu_1 = nu_1
        self.nu_0 = nu_0
        self.scale_1 = scale_1.astype(self.DAT_TYPE)
        self.scale_0 = scale_0.astype(self.DAT_TYPE)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n_length = int(self.n_multiple*self.flash_and_pause_length)
        self.identity = np.tile(np.eye(self.n_length, self.n_length)[np.newaxis, ...],
                                [self.num_electrode, 1, 1]).astype(self.DAT_TYPE)
        self.letter_dim = letter_dim
        self.trn_repetition = trn_repetition
        self.batch_dim_1 = int(self.letter_dim * self.trn_repetition * self.flash_sum)
        self.batch_dim_0 = int(self.letter_dim * self.trn_repetition * self.non_flash_sum)
        self.rearrange = ReArrangeBetaSigma(
            n_multiple=self.n_multiple,
            num_electrode=self.num_electrode,
            flash_and_pause_length=self.flash_and_pause_length)

    def initialize_parameters(self, eeg_type):
        delta_1_mcmc = np.stack([
            stats.multivariate_normal(
                mean=self.mu_1_delta[i, :],
                cov=self.sigma_sq_delta
            ).rvs(1) for i in range(self.num_electrode)], axis=0)

        delta_0_mcmc = np.stack([
            stats.multivariate_normal(
                mean=self.mu_0_delta[i, :],
                cov=self.sigma_sq_delta
            ).rvs(1) for i in range(self.num_electrode)], axis=0)

        pi_1_mcmc = 0.5 * np.ones([1, self.num_electrode, self.n_length])
        pi_0_mcmc = 0.5 * np.ones([1, self.num_electrode, self.n_length])

        gamma_1_mcmc = np.random.binomial(
            n=1, p=0.5, size=[self.num_electrode, self.n_length])
        gamma_0_mcmc = np.random.binomial(
            n=1, p=0.5, size=[self.num_electrode, self.n_length])

        pres_mat_1_mcmc = stats.wishart(
            df=self.n_length+1,
            scale=np.eye(self.n_length)
        ).rvs(self.num_electrode)[np.newaxis, ...]
        pres_mat_0_mcmc = stats.wishart(
            df=self.n_length+1,
            scale=np.eye(self.n_length)
        ).rvs(self.num_electrode)[np.newaxis, ...]

        beta_1_mcmc = np.stack([
            stats.multivariate_normal(
                mean=delta_1_mcmc[i, :], cov=1.0
            ).rvs(self.letter_dim*self.trn_repetition*self.flash_sum)
            for i in range(self.num_electrode)], axis=1)[..., np.newaxis]
        beta_0_mcmc = np.stack([
            stats.multivariate_normal(
                mean=delta_0_mcmc[i, :], cov=1.0
            ).rvs(self.letter_dim*self.trn_repetition*self.non_flash_sum)
            for i in range(self.num_electrode)], axis=1)[..., np.newaxis]

        delta_1_mcmc = delta_1_mcmc[np.newaxis, ..., np.newaxis]
        delta_0_mcmc = delta_0_mcmc[np.newaxis, ..., np.newaxis]
        beta_mcmc = np.concatenate([beta_1_mcmc, beta_0_mcmc], axis=0)
        id_beta = self.rearrange.create_permute_beta_id(
            self.letter_dim, self.trn_repetition, eeg_type=eeg_type)
        beta_iter_k = self.permute_beta_by_type(
            self.letter_dim, self.trn_repetition, beta_mcmc, id_beta)
        print('beta_iter_k has shape {}'.format(beta_iter_k.shape))
        pres_lambda_mcmc = stats.gamma(
            a=self.alpha
        ).rvs(self.num_electrode)[np.newaxis, :, np.newaxis]
        return [
            delta_1_mcmc, delta_0_mcmc,
            pi_1_mcmc, pi_0_mcmc,
            gamma_1_mcmc, gamma_0_mcmc,
            pres_mat_1_mcmc, pres_mat_0_mcmc,
            beta_iter_k,
            pres_lambda_mcmc
        ]

    @staticmethod
    def recover_normal_canonical_form(lambda_mat, eta_vec):

        lambda_mat_shape = lambda_mat.shape
        assert len(lambda_mat_shape) >=2, print('lambda_mat input should be at least a matrix!')

        batch_dim = np.prod(lambda_mat_shape[:-2])
        lambda_mat = np.reshape(lambda_mat, [batch_dim,
                                             lambda_mat_shape[-1],
                                             lambda_mat_shape[-1]])
        eta_vec = np.reshape(eta_vec, [batch_dim,
                                       lambda_mat_shape[-1]])
        # Compute lambda_mat inverse by Chol_factor
        lambda_mat_chol = np.linalg.cholesky(lambda_mat)
        sigma_mat_half = np.transpose(np.linalg.inv(lambda_mat_chol), [0, 2, 1])
        mean_vec = np.array([
            linalg.cho_solve((lambda_mat_chol[i, ...], True), eta_vec[i, :])
            for i in range(batch_dim[0])])
        sigma_mat_half = np.reshape(sigma_mat_half, lambda_mat_shape)
        mean_vec = np.reshape(mean_vec, lambda_mat_shape[:-1])

        return [mean_vec, sigma_mat_half]

    def update_delta_post(
            self, pres_1, pres_0,
            gamma_1, gamma_0,
            beta_1_sum, beta_0_sum):

        assert beta_1_sum.shape == (self.num_electrode, self.n_length, 1)
        assert beta_0_sum.shape == (self.num_electrode, self.n_length, 1)

        # Convert gamma_1/0 to diagonal matrices
        gamma_1 = np.stack([np.diag(gamma_1[i, :]) for i in range(self.num_electrode)], axis=0)
        gamma_0 = np.stack([np.diag(gamma_0[i, :]) for i in range(self.num_electrode)], axis=0)

        lambda_mat_1 = self.identity / self.sigma_sq_delta \
                       + self.batch_dim_1 * np.matmul(np.matmul(gamma_1, pres_1), gamma_1)
        eta_1 = self.mu_1_delta / self.sigma_sq_delta + \
                np.matmul(gamma_1, np.matmul(pres_1, beta_1_sum))

        lambda_mat_0 = self.identity / self.sigma_sq_delta \
                       + self.batch_dim_0 * np.matmul(np.matmul(gamma_0, pres_0), gamma_1)
        eta_0 = self.mu_0_delta / self.sigma_sq_delta + \
                np.matmul(gamma_0, np.matmul(pres_0, beta_0_sum))

        [post_mu_delta_1, sigma_mat_1_half] = self.recover_normal_canonical_form(
            lambda_mat_1, eta_1)
        [post_mu_delta_0, sigma_mat_0_half] = self.recover_normal_canonical_form(
            lambda_mat_0, eta_0)

        std_mvn_rv = stats.multivariate_normal(
            mean=np.zeros([self.n_length]),
            cov=1.0)
        delta_1_post = std_mvn_rv.rvs(size=self.num_electrode)[..., np.newaxis]
        delta_1_post = np.matmul(sigma_mat_1_half, delta_1_post) + post_mu_delta_1

        delta_0_post = std_mvn_rv.rvs(size=self.num_electrode)[..., np.newaxis]
        delta_0_post = np.matmul(sigma_mat_0_half, delta_0_post) + post_mu_delta_0

        return delta_1_post, delta_0_post

    def compute_stratified_beta_sum(self, beta_mat, beta_1_indices, beta_0_indices):

        total_beta_num = int(self.batch_dim_1 + self.batch_dim_0)
        assert beta_mat.shape == (total_beta_num, self.num_electrode, self.n_length, 1)

        # beta_mat_1 = tf.gather(beta_mat, beta_1_indices, axis=0)
        # beta_mat_0 = tf.gather(beta_mat, beta_0_indices, axis=0)

        beta_mat_1 = beta_mat[beta_1_indices, ...]
        beta_mat_0 = beta_mat[beta_0_indices, ...]

        beta_sum_1 = np.sum(beta_mat_1, axis=0)
        beta_sum_0 = np.sum(beta_mat_0, axis=0)

        return beta_sum_1, beta_sum_0

    def update_pi_post(self, gamma_1, gamma_0, radius):

        # To simplify the computation, we use equal weights (sum up to 1)
        append_zeros = np.zeros([self.num_electrode, radius])
        gamma_1 = np.concatenate([append_zeros, gamma_1, append_zeros], axis=-1)
        gamma_0 = np.concatenate([append_zeros, gamma_0, append_zeros], axis=-1)

        wts_vec = np.ones([2*radius+1, 1]) / (2*radius+1)

        pi_1_post = []
        pi_0_post = []

        for i in range(radius, radius+self.n_length):
            pi_1_post.append(np.matmul(gamma_1[:, (i-radius):(i+radius+1)], wts_vec))
            pi_0_post.append(np.matmul(gamma_0[:, (i-radius):(i+radius+1)], wts_vec))

        pi_1_post = np.concatenate(pi_1_post, axis=-1)
        pi_0_post = np.concatenate(pi_0_post, axis=-1)

        return [pi_1_post, pi_0_post]

    def update_gamma_post(self, pi_1, pi_0):
        assert pi_1.shape == pi_0.shape == (self.num_electrode, self.n_length), \
            print('Wrong pi matrix shape!')

        temp_dim = self.num_electrode * self.n_length
        pi_1 = np.reshape(pi_1, [temp_dim])
        pi_0 = np.reshape(pi_0, [temp_dim])

        gamma_1_post = [np.random.binomial(1, pi_1[i], size=1) for i in range(temp_dim)]
        gamma_0_post = [np.random.binomial(1, pi_0[i], size=1) for i in range(temp_dim)]

        gamma_1_post = np.concatenate(gamma_1_post, axis=0)
        gamma_0_post = np.concatenate(gamma_0_post, axis=0)

        gamma_1_post = np.reshape(gamma_1_post, [self.num_electrode, self.n_length])
        gamma_0_post = np.reshape(gamma_0_post, [self.num_electrode, self.n_length])

        return [gamma_1_post, gamma_0_post]

    def update_precision_matrix_post(
            self, outer_prod_1, outer_prod_0,
    ):

        post_df_1 = self.nu_1 + self.batch_dim_1
        post_df_0 = self.nu_0 + self.batch_dim_0

        post_scale_1_inv = np.linalg.inv(self.scale_1) + outer_prod_1
        post_scale_0_inv = np.linalg.inv(self.scale_0) + outer_prod_0

        post_scale_1_tran = np.linalg.inv(np.linalg.cholesky(post_scale_1_inv))
        post_scale_0_tran = np.linalg.inv(np.linalg.cholesky(post_scale_0_inv))

        post_scale_1 = np.matmul(np.transpose(post_scale_1_tran, [0, 2, 1]), post_scale_1_tran)
        post_scale_0 = np.matmul(np.transpose(post_scale_0_tran, [0, 2, 1]), post_scale_0_tran)

        post_pres_1 = np.stack([stats.wishart(df=post_df_1, scale=post_scale_1[i, ...]).rvs(1).astype(self.DAT_TYPE)
                                for i in range(self.num_electrode)], axis=0)
        post_pres_0 = np.stack([stats.wishart(df=post_df_0, scale=post_scale_0[i, ...]).rvs(1).astype(self.DAT_TYPE)
                                for i in range(self.num_electrode)], axis=0)

        return post_pres_1, post_pres_0

    def compute_outer_prod(
            self, beta_mat, delta_1, delta_0,
            gamma_1, gamma_0,
            beta_1_indices, beta_0_indices):

        # Convert gamma_1/0 to diagonal matrices
        gamma_1 = np.stack([np.diag(gamma_1[i, :]) for i in range(self.num_electrode)], axis=0)
        gamma_0 = np.stack([np.diag(gamma_0[i, :]) for i in range(self.num_electrode)], axis=0)

        beta_mat_1 = beta_mat[beta_1_indices, ...]
        beta_mat_0 = beta_mat[beta_0_indices, ...]

        beta_mat_delta_1 = beta_mat_1 - np.matmul(gamma_1, delta_1)[np.newaxis, ...]
        beta_mat_delta_0 = beta_mat_0 - np.matmul(gamma_0, delta_0)[np.newaxis, ...]

        out_prod_1 = np.sum(np.matmul(
            beta_mat_delta_1, np.transpose(beta_mat_delta_1, [0, 1, 3, 2])), axis=0)
        out_prod_0 = np.sum(np.matmul(
            beta_mat_delta_0, np.transpose(beta_mat_delta_0, [0, 1, 3, 2])), axis=0)
        return out_prod_1, out_prod_0

    # The way to generate predicted signals can be borrowed from ConvolFun.py file.
    def create_design_matrix_bayes(self):
        # Create a zero matrix
        dm_row = (self.trn_repetition * self.num_rep + self.n_multiple - 1) * self.flash_and_pause_length
        dm_col = self.trn_repetition * self.num_rep * self.n_length
        dm = np.zeros([dm_row, dm_col], dtype=self.DAT_TYPE)
        id_block = np.eye(N=self.n_length, M=self.n_length)
        for trial_id in range(self.trn_repetition * self.num_rep):
            row_id_low = trial_id * self.flash_and_pause_length
            row_id_upp = row_id_low + self.n_length
            col_id_low = trial_id * self.n_length
            col_id_upp = col_id_low + self.n_length
            dm[row_id_low:row_id_upp, col_id_low:col_id_upp] = id_block
        dm = np.tile(dm[np.newaxis, np.newaxis, :, :], [self.letter_dim, self.num_electrode, 1, 1])
        # dm should have output shape (19, 16, 920, 4500)
        return dm

    def create_predicted_signals(self, dm, beta_mat):
        # beta_mat should have input shape (19, 180, 16, 25, 1)
        beta_mat = np.reshape(np.transpose(beta_mat, [0, 2, 1, 3, 4]),
                              [self.letter_dim,
                               self.num_electrode,
                               self.trn_repetition*self.num_rep*self.n_length,
                               1])
        return np.matmul(dm, beta_mat)

    def create_partial_design_matrix_bayes(self):

        dm_extract_1 = np.zeros([2 * self.n_multiple - 2, self.n_length, self.n_length])
        dm_transform_1 = np.zeros([2 * self.n_multiple - 2, self.n_length, self.n_length])
        transform_diag = np.zeros([self.n_length])
        for i in range(self.n_multiple-1):

            pre_extract_diag_ele = np.concatenate([np.zeros([(self.n_multiple - 1 - i) * self.flash_and_pause_length]),
                                                   np.ones([(i+1)*self.flash_and_pause_length])], axis=0)
            dm_extract_1[i, ...] = np.copy(np.diag(pre_extract_diag_ele))
            post_extract_diag_ele = np.concatenate([np.ones([(i + 1) * self.flash_and_pause_length]),
                                                    np.zeros([(self.n_multiple - 1 - i)*self.flash_and_pause_length])],
                                                   axis=0)
            dm_extract_1[-i - 1, ...] = np.copy(np.diag(post_extract_diag_ele))

            transform_off_diag = np.ones([(i+1) * self.flash_and_pause_length])
            dm_transform_1[i, ...] = np.copy(np.diag(transform_diag, 0) +
                                             np.diag(transform_off_diag,
                                                     self.n_length - (i+1) * self.flash_and_pause_length))
            dm_transform_1[-i-1, ...] = np.copy(np.diag(transform_diag, 0) +
                                                np.diag(transform_off_diag, (i+1)*self.flash_and_pause_length - self.n_length))

        dm_combined = np.matmul(dm_transform_1, dm_extract_1)
        # 视情况决定要不要tile letter_dim batch以及转换成tensor的output format。
        dm_combined = np.tile(dm_combined[:, np.newaxis, ...], [1, self.num_electrode, 1, 1])

        return dm_combined  # (8, 16, 25, 25)

    @staticmethod
    def compute_mu_conditional_beta(dm, eeg_signal_slice, beta_slice_pre, beta_slice_post):

        # dm should have input shape (8, 16, 25, 25)
        # eeg_signal slice should have input shape (16, 25, 1)
        # beta_slice_pre/post should have input shape (4, 16, 25, 1)
        # Here the pre/post refer to the time rather than the order of mcmc iteration!
        beta_slice = np.concatenate([beta_slice_pre, beta_slice_post], axis=0)
        # print('beta_slice has shape {}'.format(beta_slice.shape))
        # print('dm has shape {}'.format(dm.shape))
        # print('eeg_signal_slice has shape {}'.format(eeg_signal_slice.shape))
        mu_conditional_beta = eeg_signal_slice - np.sum(np.matmul(dm, beta_slice), axis=0)

        return mu_conditional_beta

    def extract_eeg_signal_slice(self, eeg_signals, letter_id, rep_id):
        lower_id = rep_id * self.flash_and_pause_length
        upper_id = lower_id + self.n_length
        eeg_signal_slice = eeg_signals[letter_id, :, lower_id:upper_id, 0]

        return eeg_signal_slice[..., np.newaxis]

    def update_individual_beta(
            self, pres_matrix, pres_lambda, delta, gamma, mu_conditional_beta):

        gamma = np.stack([np.diag(gamma[i, :]) for i in range(self.num_electrode)], axis=0)
        pres_lambda = pres_lambda[:, np.newaxis]

        lambda_beta_mat = np.multiply(pres_lambda, self.identity) + pres_matrix
        eta_beta = np.matmul(pres_matrix, np.matmul(gamma, delta)) + \
                   np.multiply(pres_lambda, mu_conditional_beta)

        '''
        lambda_beta_mat_chol = np.linalg.cholesky(lambda_beta_mat)
        sigma_lambda_mat_half = np.transpose(np.linalg.inv(lambda_beta_mat_chol), [0, 2, 1])
        post_mu_delta = np.stack([linalg.cho_solve((lambda_beta_mat_chol[i, ...], True), eta_beta[i, :])
                                  for i in range(self.num_electrode)], axis=0)
        '''
        [post_mu_delta, sigma_lambda_mat_half] = self.recover_normal_canonical_form(
            lambda_beta_mat, eta_beta)
        print('post_mu_delta has shape {}'.format(post_mu_delta.shape))

        std_mvn_rv = stats.multivariate_normal(
            mean=np.zeros([self.n_length]),
            cov=1.0)

        beta_post_sample = std_mvn_rv.rvs(self.num_electrode)[..., np.newaxis]
        print('beta_post_sample before matmul has shape {}'.format(beta_post_sample.shape))
        beta_post_sample = np.matmul(sigma_lambda_mat_half, beta_post_sample) + post_mu_delta[:, np.newaxis]

        return beta_post_sample

    def update_precision_lambda_post(self, letter_dim, eeg_signals, predicted_signals):

        # notice that eeg_signals and predicted_signals should have input shape
        # dimension (letter_dim, num_electrode, trn_seq_length, 1)
        trn_seq_length = eeg_signals.shape[2]
        alpha_post = letter_dim * trn_seq_length / 2 + self.alpha
        signal_diff = eeg_signals - predicted_signals

        signal_diff_sq = np.sum(np.squeeze(np.matmul(np.transpose(signal_diff, [0, 1, 3, 2]),
                                                     signal_diff), axis=(2, 3)), axis=0)
        # print('signal_diff_sq has shape {}'.format(signal_diff_sq.shape))
        beta_post = signal_diff_sq / 2 + self.beta
        # We need to rescale final = std * scale + loc
        # Notice that beta = 1 / scale
        lambda_post = stats.gamma(a=alpha_post).rvs(size=self.num_electrode).astype(self.DAT_TYPE)
        lambda_post = lambda_post / beta_post

        return lambda_post[:, np.newaxis]

    def permute_beta_by_type(self, letter_dim, repet_dim,
                             beta_combined, id_beta):

        dim_temp = letter_dim * repet_dim * self.num_rep
        # print(beta_combined.shape)
        assert beta_combined.shape == (dim_temp, self.num_electrode, self.n_length, 1), \
            print('beta_combined has wrong input shape!')

        beta_combined = beta_combined[id_beta, ...]

        return beta_combined
