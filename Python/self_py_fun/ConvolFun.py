import tensorflow as tf
import tensorflow_probability as tfp
# from tensorflow_probability import edward2 as ed2
tfd = tfp.distributions
tfb = tfp.bijectors
tfp_kernels = tfp.positive_semidefinite_kernels
import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.PreFun import *
# import scipy as sp
import seaborn as sns
# from scipy.spatial.distance import pdist, squareform
plt.style.use('ggplot')
sns.set_context('notebook')
# https://wookayin.github.io/tensorflow-talk-debugging/#29
# https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/TensorFlow_Probability_Case_Study_Covariance_Estimation.ipynb#scrollTo=znG_AtTR7qob


# Multivariate normal parametrized by loc and Cholesky precision matrix.
class MVNPrecisionCholesky(tfd.TransformedDistribution):

    def __init__(self, loc, precision_cholesky, re_batch_ndims, name=None):
        super(MVNPrecisionCholesky, self).__init__(
            distribution=tfd.Independent(
                tfd.Normal(loc=tf.zeros_like(loc, dtype='float32'),
                           scale=tf.ones_like(loc, dtype='float32')),
                reinterpreted_batch_ndims=re_batch_ndims),
            bijector=tfb.Chain([
                tfb.Affine(shift=loc),
                tfb.Invert(tfb.Affine(scale_tril=precision_cholesky,
                                      adjoint=True)),
            ]), name=name)


class PriorModel:
    num_rep = 12
    DAT_TYPE = 'float32'
    eps_value = tf.keras.backend.epsilon()

    def __init__(self, n_length, num_electrode):
        self.n_length = n_length
        self.num_electrode = num_electrode

    @staticmethod
    def generate_delta(mean_vec, hyper_delta_var, convert_to_numpy=False):
        delta_rv = tfd.MultivariateNormalDiag(
            loc=mean_vec,
            scale_identity_multiplier=hyper_delta_var)
        delta_sample = tf.squeeze(delta_rv.sample(1), axis=0)
        if convert_to_numpy:
            delta_sample = tf.keras.backend.eval(delta_sample)
        return delta_sample

    @staticmethod
    def compute_delta_log_lhd(mean_vec, hyper_delta_var, target_delta_value):
        delta_rv = tfd.MultivariateNormalDiag(
            loc=mean_vec,
            scale_identity_multiplier=hyper_delta_var
        )
        return tf.reduce_sum(delta_rv.log_prob(target_delta_value))

    def generate_pres_chky_matrix(self, df, channel_dim=1, convert_to_numpy=False):
        pres_chky_rv = tfd.Wishart(
            df=df,
            scale=tf.eye(self.n_length, self.n_length, [channel_dim]),
            input_output_cholesky=True,
            validate_args=True
        )
        pres_chky_sample = tf.squeeze(pres_chky_rv.sample(1), axis=0)
        if convert_to_numpy:
            pres_chky_sample = tf.keras.backend.eval(pres_chky_sample)
        return pres_chky_sample

    def compute_pres_chky_log_lhd(self, df, pres_chky_matrix_value, channel_dim=1):
        pres_chky_rv = tfd.Wishart(
            df=df,
            scale=tf.eye(self.n_length, self.n_length, [channel_dim]),
            input_output_cholesky=True)
        return tf.reduce_sum(pres_chky_rv.log_prob(pres_chky_matrix_value))

    # https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
    # Potential reference:
    # https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1966.10502018#.Xc12Cy2ZNgc
    def generate_pres_chky_2(self, sigma_r_sq, channel_dim=1, convert_to_numpy=False):
        uni_normal_rv = tfd.MultivariateNormalDiag(
            loc=tf.zeros([channel_dim, self.n_length*(self.n_length+1)/2]),
            scale_identity_multiplier=sigma_r_sq
        )
        # Need to modify the main-diagonal term with another chi-square random variable w/
        # df n-i+1.
        # Convert multi-1d array to an upper-triangular matrix (with channel_dim batch)
        df_vec = tf.range(self.n_length, dtype=self.DAT_TYPE)+1
        df_vec = df_vec[::-1]
        df_vec = tf.tile(df_vec[tf.newaxis, :], [channel_dim, 1])
        diag_chisq_rv = tfd.Chi2(df=df_vec)

        uni_normal_sample = tf.squeeze(uni_normal_rv.sample(1), axis=0)
        chisq_sample = tf.squeeze(diag_chisq_rv.sample(1), axis=0)
        if convert_to_numpy:
            uni_normal_sample = tf.keras.backend.eval(uni_normal_sample)
            chisq_sample = tf.keras.backend.eval(chisq_sample)
        return uni_normal_sample, chisq_sample

    @staticmethod
    def convert_1d_array_to_upper_triangular(uni_normal_sample, chisq_sample,
                                             convert_to_numpy=False):
        uni_normal_sample = tfd.fill_triangular(uni_normal_sample, upper=True)
        uni_normal_sample = tf.linalg.set_diag(uni_normal_sample, tf.sqrt(chisq_sample))
        if convert_to_numpy:
            uni_normal_sample = tf.keras.backend.eval(uni_normal_sample)
        return uni_normal_sample

    @staticmethod
    def compute_pres_upper_tri(cov_matrix, convert_to_numpy=False):
        pres_matrix = tf.linalg.inv(cov_matrix)
        upper_tri_sample = tf.transpose(tf.linalg.cholesky(pres_matrix), [0, 2, 1])
        if convert_to_numpy:
            upper_tri_sample = tf.keras.backend.eval(upper_tri_sample)
        return upper_tri_sample

    @staticmethod
    def convert_upper_triangular_to_1d_array(upper_tri_sample, convert_to_numpy=False):
        upper_tri_sample = tfd.fill_triangular_inverse(upper_tri_sample, upper=True)
        if convert_to_numpy:
            upper_tri_sample = tf.keras.backend.eval(upper_tri_sample)
        return upper_tri_sample

    def compute_pres_chky_log_lhd_2(self, sigma_r_sq, uni_normal_sample, chisq_sample,
                                    channel_dim=1):
        # Notice that uni_normal_sample is long_array
        uni_normal_rv = tfd.MultivariateNormalDiag(
            loc=tf.zeros([channel_dim, self.n_length*(self.n_length+1)/2]),
            scale_identity_multiplier=sigma_r_sq
        )
        normal_log_lhd = tf.reduce_sum(uni_normal_rv.log_prob(uni_normal_sample))

        df_vec = tf.range(self.n_length, dtype=self.DAT_TYPE) + 1
        df_vec = df_vec[::-1]
        df_vec = tf.tile(df_vec[tf.newaxis, :], [channel_dim, 1])
        diag_chisq_rv = tfd.Chi2(df=df_vec)
        chisq_log_lhd = tf.reduce_sum(diag_chisq_rv.log_prob(chisq_sample))

        return normal_log_lhd + chisq_log_lhd

    def generate_eta_and_compute_log_lhd(
            self, pres_chky, letter_dim, rep_dim, flash_num,
            convert_to_numpy=False, channel_dim=1):
        std_normal_rv = tfd.MultivariateNormalDiag(
            loc=tf.zeros([channel_dim, self.n_length]),
            scale_identity_multiplier=tf.ones([]))
        dim_temp = letter_dim * rep_dim * flash_num
        eta = std_normal_rv.sample([dim_temp])

        log_lhd_xi = tf.reduce_sum(std_normal_rv.log_prob(eta))
        log_abs_diag_value = tf.reduce_sum(tf.linalg.logdet(pres_chky))
        # log_abs_diag_value = tf.reduce_sum(tf.math.log(self.eps_value + tf.abs(tf.linalg.diag_part(pres_chky))))
        log_lhd_pres_chky = dim_temp * log_abs_diag_value
        # Solve eta from pres_chky_1 * eta = normal_vector

        eta = tf.transpose(eta, [1, 0, 2])[..., tf.newaxis]
        # eta should have shape (channel_dim, ..., n_length, 1)
        # pres_chky should have shape (channel_dim, n_length, n_length)
        # print('eta has shape {}'.format(eta.shape))
        # print('pres_chky has shape {}'.format(pres_chky.shape))

        def _solve_eta_per_channel(elems):
            pres_chky_chan = elems[0]
            eta_chan = elems[1]
            pres_chky_chan = tf.tile(pres_chky_chan[tf.newaxis, ...], [dim_temp, 1, 1])
            eta_chan = tf.linalg.triangular_solve(pres_chky_chan, eta_chan, lower=False)
            return eta_chan

        elems = (pres_chky, eta)
        eta_solve = tf.map_fn(_solve_eta_per_channel, elems, dtype=self.DAT_TYPE)
        eta_solve = tf.transpose(tf.squeeze(eta_solve, axis=-1), [1, 0, 2])

        if convert_to_numpy:
            eta_solve = tf.keras.backend.eval(eta_solve)
        return eta_solve, log_lhd_xi+log_lhd_pres_chky

    @staticmethod
    def generate_pres_e(alpha, beta, convert_to_numpy=False, channel_dim=1):
        pres_e_rv = tfd.Gamma(
            concentration=alpha*tf.ones([channel_dim]),
            rate=beta*tf.ones([channel_dim])
        )
        pres_e = tf.squeeze(pres_e_rv.sample(1), axis=0)
        if convert_to_numpy:
            pres_e = tf.keras.backend.eval(pres_e)

        return pres_e

    @staticmethod
    def compute_pres_e_log_lhd(alpha, beta, pres_e, channel_dim=1):
        pres_e_rv = tfd.Gamma(
            concentration=alpha*tf.ones([channel_dim]),
            rate=beta*tf.ones([channel_dim])
        )

        return tf.reduce_sum(pres_e_rv.log_prob(pres_e))


class ReArrangeBetaSigma:
    # Global constants:
    num_rep = 12
    f_sum = 2
    nf_sum = num_rep - f_sum
    DAT_TYPE = 'float32'

    def __init__(self, n_multiple, num_electrode, flash_and_pause_length):
        self.n_multiple = n_multiple
        self.num_electrode = num_electrode
        self.flash_and_pause_length = flash_and_pause_length
        self.n_length = int(n_multiple * flash_and_pause_length)

    def tile_and_combine_delta(self, letter_dim, repet_dim, delta_1, delta_0, channel_dim=1):

        assert delta_1.shape == (channel_dim, self.n_length), \
            print('delta_1 has wrong input shape!')
        assert delta_0.shape == (channel_dim, self.n_length), \
            print('delta_0 has wrong input shape!')
        delta_1 = tf.convert_to_tensor(delta_1)[tf.newaxis, ...]
        delta_0 = tf.convert_to_tensor(delta_0)[tf.newaxis, ...]
        dim_1 = letter_dim * repet_dim * self.f_sum
        dim_0 = letter_dim * repet_dim * self.nf_sum
        delta_1 = tf.tile(delta_1, multiples=[dim_1, 1, 1])
        delta_1 = tf.reshape(tf.transpose(delta_1, perm=[1, 0, 2]),
                            shape=[channel_dim, dim_1, self.n_length])
        delta_0 = tf.tile(delta_0, multiples=[dim_0, 1, 1])
        delta_0 = tf.reshape(tf.transpose(delta_0, perm=[1, 0, 2]),
                            shape=[channel_dim, dim_0, self.n_length])
        delta_combined = tf.concat([delta_1, delta_0], axis=1)

        return delta_combined

    # Need to absorb this numpy function within tensorflow graph (especially for prediction)
    def create_permute_beta_id(self, letter_dim, repet_dim, eeg_type):

        dim_temp = letter_dim * repet_dim * self.num_rep
        assert eeg_type.shape == (dim_temp,), print('eeg_type has wrong input shape!')
        id_beta = np.zeros([dim_temp]) - 1
        id_beta[eeg_type == 1] = np.arange(self.f_sum*letter_dim*repet_dim)
        id_beta[eeg_type != 1] = np.arange(self.f_sum*letter_dim*repet_dim, dim_temp)

        return id_beta.astype('int32')

    # This function requires further editing in terms of the dimension rearrangement.
    def permute_beta_by_type(self, letter_dim, repet_dim,
                             beta_combined, id_beta, channel_dim=1):

        dim_temp = letter_dim * repet_dim * self.num_rep
        assert beta_combined.shape == (dim_temp, channel_dim, self.n_length), \
            print('beta_combined has wrong input shape!')
        # 2280, 16, 25
        # beta_combined = tf.reshape(beta_combined, [channel_dim,
        #                                            dim_temp,
        #                                            self.n_length])
        beta_combined = tf.gather(beta_combined, id_beta,
                                  axis=0, name='beta_permuted')
        beta_combined = tf.reshape(beta_combined, [letter_dim,
                                                   repet_dim*self.num_rep,
                                                   channel_dim,
                                                   self.n_length])
        beta_combined = tf.transpose(beta_combined, [0, 2, 1, 3])
        # print('beta_combined has shape {}'.format(beta_combined.shape))
        beta_combined = tf.reshape(beta_combined, [letter_dim,
                                                   channel_dim,
                                                   repet_dim*self.num_rep*self.n_length,
                                                   1])
        return beta_combined

    # The following functions are based on Bayesian generative model
    # which may apply tensorflow and tensorflow-probability.
    # Notice that the design matrix can be automatically broadcast
    # w.r.t channel batch and letter batch.
    def create_design_mat_gen_bayes_seq(self, repetition_dim):
        r"""

        :param repetition_dim: integer
        :return: design_x, with output shape
            [(num_rep*num_repetition+n_multiple-1)*flash_and_pause_length, num_rep*num_repetition*n_length]

        """
        # Create a zero matrix
        dm_row = (repetition_dim*self.num_rep + self.n_multiple - 1) * self.flash_and_pause_length
        dm_col = repetition_dim*self.num_rep*self.n_length
        dm = np.zeros([dm_row, dm_col])
        id_block = np.eye(self.n_length)

        for trial_id in range(repetition_dim*self.num_rep):
            row_id_low = trial_id * self.flash_and_pause_length
            row_id_upp = row_id_low + self.n_length
            col_id_low = trial_id * self.n_length
            col_id_upp = col_id_low + self.n_length
            dm[row_id_low:row_id_upp, col_id_low:col_id_upp] = id_block

        return dm

    def create_joint_beta_tilta(
            self, letter_dim, repet_dim, beta_combined, id_beta, design_x, channel_dim=1
    ):
        r"""
        :param letter_dim: integer
        :param repet_dim: integer
        :param beta_combined: array_like, (channe_dim, letter_dim, noise_size*n_length, 1)
        :param id_beta: array_like, or None
        :param design_x: array_like, (channel_dim, letter_dim, seq_length, noise_size*n_length)
        :param channel_dim: array_like
        :return:
                For design_x, we can ignore the outer-2 batch as long as the rightmost 2 dimensions are correct.
                For id_beta, if none, it implies that the beta_combined has already been permuted.
        """

        if id_beta is not None:
            beta_combined = self.permute_beta_by_type(
                letter_dim, repet_dim, beta_combined, id_beta, channel_dim)

        seq_x = design_x @ beta_combined

        return seq_x

    @staticmethod
    def convert_from_pres_to_cov(pres_chky_mat, convert_to_numpy=False):
        pres_chky_mat_inv = tf.linalg.inv(pres_chky_mat)
        cov_mat = tf.matmul(pres_chky_mat_inv, tf.transpose(pres_chky_mat_inv, [0, 2, 1]))
        if convert_to_numpy:
            cov_mat = tf.keras.backend.eval(cov_mat)
        return cov_mat

    @staticmethod
    def convert_from_cov_to_pres(cov_mat, convert_to_numpy=False):
        prec_chky_mat = tf.linalg.cholesky(tf.linalg.inv(cov_mat))
        if convert_to_numpy:
            prec_chky_mat = tf.keras.backend.eval(prec_chky_mat)
        return prec_chky_mat

    def create_trial_specific_cov_fn(self, trial_i, unit_pres_mat_value, rep_dim, channel_dim=1):
        assert unit_pres_mat_value.shape == (channel_dim, self.n_length, self.n_length)
        unit_cov_fn_value = self.convert_from_pres_to_cov(unit_pres_mat_value)
        large_cov_size = (self.num_rep * rep_dim + self.n_multiple - 1) * self.flash_and_pause_length
        upp_left = trial_i*self.flash_and_pause_length
        low_right = large_cov_size - trial_i*self.flash_and_pause_length - self.n_length
        paddings = tf.constant([[0, 0], [upp_left, low_right], [upp_left, low_right]])
        large_cov_value = tf.pad(unit_cov_fn_value, paddings, "constant")
        return large_cov_value

    def create_collapsed_cov_fn(self, letter_dim, rep_dim, unit_target_pres_value,
                                unit_nt_pres_value, eeg_type, channel_dim=1):
        trial_sum = self.num_rep * rep_dim
        trn_total_seq = (trial_sum+self.n_multiple-1) * self.flash_and_pause_length
        eeg_type = np.reshape(eeg_type, [letter_dim, trial_sum])
        collapsed_cov_fn = tf.zeros([channel_dim, trn_total_seq, trn_total_seq])

        for letter_id in range(letter_dim):
            collapsed_cov_fn_letter = tf.zeros([channel_dim, trn_total_seq, trn_total_seq])
            for trial_i in range(trial_sum):
                if eeg_type[letter_id, trial_i] == 1:
                    collapsed_cov_fn_letter += self.create_trial_specific_cov_fn(
                        trial_i, unit_target_pres_value, rep_dim)
                else:
                    collapsed_cov_fn_letter += self.create_trial_specific_cov_fn(
                        trial_i, unit_nt_pres_value, rep_dim)
            collapsed_cov_fn = tf.concat([collapsed_cov_fn, collapsed_cov_fn_letter], axis=0)
        collapsed_cov_fn = tf.reshape(collapsed_cov_fn[channel_dim:, ...], [letter_dim, self.num_electrode, trn_total_seq, trn_total_seq])
        # should have shape (19, 16, 920, 920)
        return collapsed_cov_fn


class MVNJointModel:

    def __init__(self, n_length, num_electrode, re_batch_ndims=2):
        self.n_length = n_length
        self.num_electrode = num_electrode
        self.re_batch_ndims = re_batch_ndims

    def generate_pseudo_signals(self, y_tilta, prec_chky_tilta=None,
                                cov_tilta=None, sample_batch=1,
                                convert_to_numpy=False, channel_dim=1):
        assert prec_chky_tilta is not None or cov_tilta is not None, \
            print('Missing covariance component.')
        if prec_chky_tilta is None:
            prec_chky_tilta = tf.linalg.cholesky(tf.linalg.inv(cov_tilta))
        jitter = 0.1 * tf.eye(self.n_length, self.n_length, [channel_dim])
        mvn_rv = MVNPrecisionCholesky(
            loc=y_tilta,
            precision_cholesky=prec_chky_tilta+jitter,
            re_batch_ndims=self.re_batch_ndims)
        pseudo_signal = mvn_rv.sample(sample_batch)
        if sample_batch == 1:
            pseudo_signal = tf.squeeze(pseudo_signal, axis=0)
        if convert_to_numpy:
            pseudo_signal = tf.keras.backend.eval(pseudo_signal)
        return pseudo_signal

    def compute_log_lhd(self, y_tilta, y_value,
                        prec_chky_tilta=None, cov_tilta=None,
                        sum_over_letter=True):
        assert prec_chky_tilta is not None or cov_tilta is not None, \
            print('Missing covariance component.')
        if prec_chky_tilta is None:
            prec_chky_tilta = tf.linalg.cholesky(tf.linalg.inv(cov_tilta))
        mvn_rv = MVNPrecisionCholesky(
            loc=y_tilta,
            precision_cholesky=prec_chky_tilta,
            re_batch_ndims=self.re_batch_ndims)
        mvn_log_prob = mvn_rv.log_prob(y_value)
        if sum_over_letter:
            mvn_log_prob = tf.reduce_sum(mvn_log_prob, axis=0)
        return mvn_log_prob


class WLSOpt(EEGPreFun):
    # class-level global constant

    def __init__(self, *args, **kwargs):
        super(WLSOpt, self).__init__(*args, **kwargs)
        # Create the bijector for chky matrices
        self.unconstrained_to_precison_chky = tfb.Chain([
            # Step 2: Exponentiate the diagonals
            tfb.TransformDiagonal(tfb.Exp(validate_args=self.VALIDATE_ARGS)),
            # Step 1: Expand the vector to a lower triangular matrix
            tfb.FillTriangular(validate_args=self.VALIDATE_ARGS),
        ])
        self.prior = PriorModel(
            n_length=self.n_length, num_electrode=self.num_electrode)
        self.rearrange = ReArrangeBetaSigma(
            n_multiple=self.n_multiple,
            num_electrode=self.num_electrode,
            flash_and_pause_length=self.flash_and_pause_length)

    @staticmethod
    def session_options(enable_gpu_ram_resizing=False):
        """Convenience function which sets a common 'tf.Session' options."""
        config = tf.ConfigProto()
        if enable_gpu_ram_resizing:
            config.gpu_options.allow_growth = True
        return config

    def reset_sess(self, config=None):
        # Convenience function to create TF graph and session or reset them.
        if config is None:
            config = self.session_options()
        tf.reset_default_graph()
        global sess
        # noinspection PyBroadException
        try:
            sess.close()
        except:
            pass
        sess = tf.InteractiveSession(config=config)

    def print_test_info(self, test_repetition):
        print('This is subject {}.'.format(self.sub_folder_name))
        print('We are predicting {} repetitions for testing purpose.'.format(test_repetition))

    # Import datafiles with WLS specific requirement:
    def import_eeg_processed_dat_wls(self, file_subscript,
                                     letter_dim=None, trn_repetition=None,
                                     reshape_to_1d=True):
        [eeg_signals, eeg_code, eeg_type] = \
            self.import_eeg_processed_dat(file_subscript, reshape_1d_bool=False)
        shape1, shape2, _ = eeg_type.shape
        if letter_dim is not None:
            assert letter_dim <= shape1, 'Incorrect letter dimension, ' \
                                         'should not be greater than {}.'.format(shape1)
        else:
            letter_dim = shape1
        if trn_repetition is not None:
            assert trn_repetition <= shape2, 'Incorrect repetition dimension, ' \
                                             'should not be greater than {}.'.format(shape2)
        else:
            trn_repetition = shape2

        # eeg_signals = eeg_signals / 10

        eeg_signals_trun, _ = self.create_truncate_segment_batch(
            eeg_signals, eeg_type, letter_dim, trn_repetition)
        trn_total_seq_length = (trn_repetition*self.num_rep+self.n_multiple-1)*self.flash_and_pause_length
        eeg_signals = np.transpose(eeg_signals[:letter_dim, :trn_total_seq_length, :],
                                   [0, 2, 1])
        if reshape_to_1d:
            eeg_code = np.reshape(eeg_code[:letter_dim, :trn_repetition, :],
                                  [letter_dim*trn_repetition*self.num_rep])
            eeg_type = np.reshape(eeg_type[:letter_dim, :trn_repetition, :],
                                  [letter_dim*trn_repetition*self.num_rep])

        return [eeg_signals.astype(self.DAT_TYPE),
                eeg_signals_trun.astype(self.DAT_TYPE),
                eeg_code.astype(self.DAT_TYPE),
                eeg_type.astype(self.DAT_TYPE)]

    # Construct design matrix X (letter-specific, intercept excluded)
    def construct_design_matrix_per_letter(
            self, total_seq_length, eeg_code, target_row_col):
        # Assume no letter effect, nor row/column effect
        params_type_num = 2
        z = np.zeros([params_type_num, self.n_length], dtype=np.int32)
        z[0, :] = np.arange(1, self.n_length+1)  # Non-target
        z[1, :] = np.arange(self.n_length+1, 2*self.n_length+1)  # Target
        # print('can multiple assignment')
        total_seq_num = int((total_seq_length - self.n_length) / self.flash_and_pause_length + 1)
        bool_index = np.in1d(eeg_code, target_row_col) * 1
        design_x = np.zeros([total_seq_num, total_seq_length], dtype=np.int32)
        for i in range(total_seq_num):
            low_num = self.flash_and_pause_length * i
            upp_num = self.flash_and_pause_length * i + self.n_length
            design_x[i, low_num:upp_num] = z[bool_index[i], :]

        design_x0 = np.zeros([total_seq_length, self.n_length*2])
        # print('design_x0 done!')
        for i in range(total_seq_length):
            for j in range(total_seq_num):
                if design_x[j, i] > 0:
                    design_x0[i, design_x[j, i]-1] = 1

        return design_x0.astype(self.DAT_TYPE)

    def create_penalty_fn(self):
        # Create the second-order diff matrix object
        # as well as smoothing around zero matrix
        P1 = np.eye(N=self.n_length, M=self.n_length, dtype=np.int32)
        P1 = P1[1:, :] - P1[:-1, :]
        P1 = P1[1:, :] - P1[:-1, :]
        P1 = np.matmul(P1.T, P1)
        P_smooth = np.eye(N=self.n_length * 2, M=self.n_length * 2, dtype=np.int32)
        P_smooth[:self.n_length, :self.n_length] = np.copy(P1)
        P_smooth[self.n_length:, self.n_length:] = np.copy(P1)
        P_zero = np.zeros([self.n_length * 2, self.n_length * 2], dtype=np.int32)
        P_zero[self.n_length:, :self.n_length:] = np.copy(np.eye(N=self.n_length, M=self.n_length,
                                                                 dtype=self.DAT_TYPE))
        return [P_smooth, P_zero]

    def from_weights_to_beta(self, design_x0, eeg_signals, l_cholesky_inv,
                             lambda_s, lambda_0, P_smooth, P_zero):

        # X^t W X = (L^-1X)^t (L^-1X)
        l_cholesky_inv = np.tile(l_cholesky_inv[:, np.newaxis, :, :],
                                 reps=[1, self.num_letter, 1, 1])
        l_inv_X = np.matmul(l_cholesky_inv, design_x0)
        l_inv_X_t = np.transpose(l_inv_X, axes=(0, 1, 3, 2))
        XtWX = np.sum(np.matmul(l_inv_X_t, l_inv_X), axis=1)
        # X^t W Y = (L^-1X)^t (L^-1Y)
        XtWY = np.sum(np.matmul(l_inv_X_t, np.matmul(l_cholesky_inv, eeg_signals)), axis=1)

        # Use cholesky decomposition to solve beta_mle
        # XtWX beta = XtWY
        # Step 1: Obtain Cholesky decomposition of XtWX with the penalty term
        l_XWX = np.linalg.cholesky(XtWX + lambda_s * P_smooth + lambda_0 * P_zero)
        l_XWX_inv = np.linalg.inv(l_XWX)
        # l_XWX @ l_XWX^t beta = XtWY
        # Step 2: Solve l_XWX theta = XtWY
        theta = np.matmul(l_XWX_inv, XtWY)
        # Step 3: Solve l_XWX^t beta = theta
        beta_mle = np.matmul(np.transpose(l_XWX_inv, axes=(0, 2, 1)), theta)
        return beta_mle

    # MLE iterations for WLS algorithm
    def from_beta_to_weights(self, design_x0, eeg_signals, beta_mle, jitter, trn_repetition=-1):
        _, x_dim, _ = design_x0.shape
        x0_beta = np.matmul(design_x0,
                            np.tile(beta_mle[:, np.newaxis, :, :],
                                    reps=[1, self.num_letter, 1, 1]))
        w_mle_inv = np.matmul(eeg_signals - x0_beta, np.transpose(eeg_signals - x0_beta, axes=(0, 1, 3, 2))) \
                    / self.num_letter
        w_mle_inv = np.sum(w_mle_inv, axis=1)
        # Place different structure on w_mle_inv to simplify the training model
        w_mle_inv = self.block_diagonal_weights(w_mle_inv, jitter, x_dim, trn_repetition)
        # Apply cholesky decomposition
        l_cholesky = np.linalg.cholesky(w_mle_inv)
        l_cholesky_inv = np.linalg.inv(l_cholesky)

        return l_cholesky_inv

    @staticmethod
    # Assume different across channels
    def unstructured_weights(w_mle_inv, jitter, x_dim):
        assert len(w_mle_inv.shape) == 3
        w_mle_inv += jitter * np.eye(N=x_dim, M=x_dim)
        return w_mle_inv

    # Assume different across channels
    def block_diagonal_weights(self, w_mle_inv, jitter, x_dim, trn_repetition, channel_dim=1):
        assert len(w_mle_inv.shape) == 3
        w_mle_inv_block_diag = np.zeros([channel_dim, x_dim, x_dim])
        for i in range(trn_repetition):
            block_low = i * self.flash_and_pause_length * self.num_rep
            block_upp = (i + 1) * self.flash_and_pause_length * self.num_rep
            w_mle_inv_block_diag[:, block_low:block_upp, block_low:block_upp] = \
                w_mle_inv[:, block_low:block_upp, block_low:block_upp]
        # The last smaller block
        block_low_2 = trn_repetition * self.num_rep * self.flash_and_pause_length
        w_mle_inv_block_diag[block_low_2:, block_low_2:] = \
            w_mle_inv[block_low_2:, block_low_2:]
        # Add jitter to the diagonal to make it more pdf
        w_mle_inv_block_diag = self.unstructured_weights(w_mle_inv_block_diag, jitter, x_dim)
        return w_mle_inv_block_diag

    @staticmethod
    # Assume different across channels
    def multi_diagonal_weights(w_mle_inv, jitter, x_dim, max_lag, channel_dim=1):
        assert len(w_mle_inv.shape) == 3
        # Add jitter and digonal term together
        w_mle_inv_md = np.diagonal(w_mle_inv, offset=0, axis1=1, axis2=2)[:, :, np.newaxis] \
                       * np.eye(N=x_dim, M=x_dim)[np.newaxis, :, :]
        w_mle_inv_md += jitter * np.eye(N=x_dim, M=x_dim)[np.newaxis, :, :]
        for i in range(1, max_lag):
            off_diagonal_val = np.diagonal(w_mle_inv, offset=i, axis1=1, axis2=2)
            # print('off_diagonal_val has shape {}'.format(off_diagonal_val.shape))
            for j in range(channel_dim):
                np.fill_diagonal(w_mle_inv_md[j, i:, :], off_diagonal_val[j, :])
                np.fill_diagonal(w_mle_inv_md[j, :, i:], off_diagonal_val[j, :])
        return w_mle_inv_md

    # Compute the loss function
    def compute_mahalanobis_dist_sq(self, design_x0, eeg_signals, beta_mle, l_cholesky_inv):

        Xbeta_l = np.matmul(design_x0, np.tile(beta_mle[:, np.newaxis, :, :],
                                               reps=[1, self.num_letter, 1, 1]))
        # print('Xbeta_l has shape {}'.format(Xbeta_l.shape))
        Y_Xb = eeg_signals - Xbeta_l
        # print('Y_Xb has shape {}'.format(Y_Xb.shape))
        l_cholesky_inv = np.tile(l_cholesky_inv[:, np.newaxis, :, :],
                                 reps=[1, self.num_letter, 1, 1])
        L_inv_res = np.matmul(l_cholesky_inv, Y_Xb)
        # print('L^-1 (Y-Xb) has shape {}'.format(L_inv_res.shape))
        L_inv_res_sq = np.sum(np.matmul(np.transpose(L_inv_res, axes=[0, 1, 3, 2]),
                                        L_inv_res), axis=0)

        return np.squeeze(L_inv_res_sq)

    # def get_log_prob_eeg_convol_fn(
    #         self, letter_dim, repet_dim,
    #         hyper_param_dict, eeg_signals, id_beta, design_x):
    #
    #     mean_vec_1 = hyper_param_dict['mean_vec_1']
    #     mean_vec_0 = hyper_param_dict['mean_vec_0']
    #     hyper_delta_var = hyper_param_dict['hyper_delta_var']
    #     hyper_sigma_r_sq = hyper_param_dict['hyper_sigma_r_sq']
    #
    #     def _log_prob_eeg_convol_fn(delta_1_value,
    #                                 delta_0_value,
    #                                 pres_array_1_value,
    #                                 pres_array_0_value):
    #
    #         # def _print_precision(pres_chky_1, pres_chky_0):
    #         #     print('precision_chky_1:\n {}'.format(pres_chky_1))
    #         #     print('precision_chky_0:\n {}'.format(pres_chky_0))
    #         #     return False  # operations must return something!
    #         # # Turn our method into a tensorflow operation
    #         # prec_chky_op = tf.numpy_function(_print_precision, [pres_chky_1_value, pres_chky_0_value], tf.bool)
    #         #
    #         # assertion_op_1 = tf.compat.v1.assert_equal(
    #         #     tf.reduce_sum(tf.linalg.band_part(pres_chky_1_value, -1, 0)), tf.cast(0, dtype=self.DAT_TYPE),
    #         #     message='Not lower triangular for pres', summarize=4, name='low-tri-check-1'
    #         # )
    #         #
    #         # assertion_op_0 = tf.assert_equal(
    #         #     tf.reduce_sum(tf.linalg.band_part(pres_chky_0_value, -1, 0)), tf.cast(0, dtype=self.DAT_TYPE),
    #         #     message='Not symmetrical', summarize=4, name='low-tri-check-0'
    #         # )
    #
    #         delta_1_log_prob = self.prior.compute_delta_log_lhd(
    #             mean_vec_1, hyper_delta_var, delta_1_value)
    #         delta_0_log_prob = self.prior.compute_delta_log_lhd(
    #             mean_vec_0, hyper_delta_var, delta_0_value)
    #
    #         pres_chky_1_log_prob = self.prior.compute_pres_chky_log_lhd_2(
    #             hyper_sigma_r_sq, pres_array_1_value)
    #         pres_chky_0_log_prob = self.prior.compute_pres_chky_log_lhd_2(
    #             hyper_sigma_r_sq, pres_array_0_value)
    #         # Intermediate variable eta
    #         # Convert pres_chky_1/0 to upper-triangular matrices
    #         pres_array_1_value = self.prior.convert_1d_array_to_upper_triangular(pres_array_1_value)
    #         pres_array_0_value = self.prior.convert_1d_array_to_upper_triangular(pres_array_0_value)
    #
    #         eta_1, eta_1_log_prob = self.prior.generate_eta_and_compute_log_lhd(
    #             pres_array_1_value, letter_dim, repet_dim, self.flash_sum)
    #         eta_0, eta_0_log_prob = self.prior.generate_eta_and_compute_log_lhd(
    #             pres_array_0_value, letter_dim, repet_dim, self.non_flash_sum)
    #
    #         delta_combined = self.rearrange.tile_and_combine_delta(
    #             letter_dim, repet_dim, delta_1_value, delta_0_value)
    #         eta_combined = tf.transpose(tf.concat([eta_1, eta_0], axis=0),
    #                                     perm=[1, 0, 2])
    #         beta_combined = delta_combined + eta_combined
    #         beta_tilta = self.rearrange.create_joint_beta_tilta(
    #             letter_dim, repet_dim, beta_combined, id_beta, design_x)
    #         beta_tilta = tf.squeeze(beta_tilta, axis=-1)
    #
    #         residuals = eeg_signals - beta_tilta
    #         eeg_signals_log_prob = -0.5 * tf.reduce_sum(tf.pow(tf.linalg.norm(
    #             residuals, ord='fro', axis=[-2, -1]), 2))
    #
    #         total_log_prob = delta_1_log_prob + delta_0_log_prob \
    #                          + pres_chky_1_log_prob + pres_chky_0_log_prob \
    #                          + eta_1_log_prob + eta_0_log_prob \
    #                          + eeg_signals_log_prob
    #
    #         return total_log_prob
    #
    #     return _log_prob_eeg_convol_fn

    # Tenga un cuenta que 'eeg_t_mean_init' y 'eeg_nt_mean_init' son al azar.
    def create_initial_chain(self,
                             eeg_t_mean_init, eeg_nt_mean_init,
                             eeg_t_cov, eeg_nt_cov):
        upper_tri_1 = self.prior.compute_pres_upper_tri(eeg_t_cov) + \
                      tf.eye(self.n_length, self.n_length, [1])
        upper_tri_0 = self.prior.compute_pres_upper_tri(eeg_nt_cov) + \
                      tf.eye(self.n_length, self.n_length, [1])
        upper_array_1 = self.prior.convert_upper_triangular_to_1d_array(upper_tri_1)
        upper_array_0 = self.prior.convert_upper_triangular_to_1d_array(upper_tri_0)
        initial_chain_states = [
            tf.random.normal(mean=eeg_t_mean_init, shape=[], dtype=self.DAT_TYPE),
            tf.random.normal(mean=eeg_nt_mean_init, shape=[], dtype=self.DAT_TYPE),
            # tf.random.normal(shape=[self.num_electrode,
            #                         int(self.n_length * (1 + self.n_length) / 2)],
            #                  dtype=self.DAT_TYPE),
            # tf.random.normal(shape=[self.num_electrode,
            #                         int(self.n_length * (1 + self.n_length) / 2)],
            #                  dtype=self.DAT_TYPE)
            upper_array_1,
            upper_array_0
        ]
        return initial_chain_states

    @staticmethod
    def _mala_kernel(target_log_prob_fn, step_size_init):
        kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size_init)
        return kernel

    @staticmethod
    def _random_walk_kernel(target_log_prob_fn, scale):
        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale)
        )
        return kernel

    # Add transformed kernel and bijector
    def _ttk_hmc_kernel(self, target_log_prob_fn,
                        num_burnin_steps, num_leapfrog_steps,
                        step_size_init, target_accept_prob):
        step_size_init = tf.convert_to_tensor(step_size_init, dtype=self.DAT_TYPE)
        target_accept_prob = tf.convert_to_tensor(target_accept_prob, dtype=self.DAT_TYPE)

        # ttk = tfp.mcmc.TransformedTransitionKernel(
        # inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        #     target_log_prob_fn=target_log_prob_fn,
        #     num_leapfrog_steps=num_leapfrog_steps,
        #     step_size=step_size_init),
        # bijector = self.create_bijectors()

        ttk = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size_init),
            num_adaptation_steps=int(num_burnin_steps * 0.8),
            target_accept_prob=target_accept_prob
        )
        return ttk

    def create_bijectors(self, channel_dim=1):
        # tfb.Softplus(), tfb.Softplus() for univariate variance
        return [tfb.Identity() for i in range(channel_dim*2)] + \
               [self.unconstrained_to_precison_chky for i in range(channel_dim*2)]

    @staticmethod
    def trace_log_accept_ratio(states, previous_kernel_results):
        return previous_kernel_results.log_accept_ratio

    @staticmethod
    def trace_everything(states, previous_kernel_results):
        return previous_kernel_results

    def mcmc_sample_chain(self, target_log_prob_fn,
                          t_mean_init, nt_mean_init,
                          eeg_t_cov, eeg_nt_cov,
                          n_samples, n_burn_in,
                          num_steps_between_results, step_size_init,
                          target_accept_prob, num_leapfrog_steps):
        # Create initial states
        initial_chain_states = self.create_initial_chain(
            t_mean_init, nt_mean_init, eeg_t_cov, eeg_nt_cov)

        para_mcmc = tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=n_burn_in,
            num_steps_between_results=num_steps_between_results,  # large value leads to large memory
            current_state=initial_chain_states,
            # kernel=self._ttk_hmc_kernel(
            #     target_log_prob_fn=target_log_prob_fn,
            #     num_burnin_steps=n_burn_in,
            #     num_leapfrog_steps=num_leapfrog_steps,
            #     step_size_init=step_size_init,
            #     target_accept_prob=target_accept_prob),
            # kernel=self._mala_kernel(
            #     target_log_prob_fn,
            #     step_size_init),
            kernel=self._random_walk_kernel(target_log_prob_fn, step_size_init),
            parallel_iterations=10,  # large iteration number leads to large memory
            return_final_kernel_results=False,
            trace_fn=None,
        )
        return para_mcmc

    @staticmethod
    def provide_hyper_params(mean_vec_1, mean_vec_0, hyper_delta_var, hyper_sigma_r_sq):

        hyper_param_dict = {
            'mean_vec_1': mean_vec_1,
            'mean_vec_0': mean_vec_0,
            'hyper_delta_var': hyper_delta_var,
            'hyper_sigma_r_sq': hyper_sigma_r_sq
        }

        return hyper_param_dict

    def save_hmc_params_est(self, params_list, message):
        if not os.path.exists('{}/{}/{}/convol_python'.format(
                self.parent_path,
                self.data_type,
                self.sub_folder_name[:4])):
            os.mkdir('{}/{}/{}/convol_python'
                     .format(self.parent_path,
                             self.data_type,
                             self.sub_folder_name[:4]))
        print(message)
        sio.savemat('{}/{}/{}/convol_python/hmc_param_est.mat'.
                    format(self.parent_path, self.data_type,
                           self.sub_folder_name[:4]),
                    {
                        'mcmc_message': message,
                        'delta_1': params_list[0],
                        'delta_0': params_list[1],
                        'pres_chky_1': params_list[2],
                        'pres_chky_0': params_list[3],
                        'pres': params_list[4]
                    })

    def save_gibbs_params_est(self, params_list, message):
        if not os.path.exists('{}/{}/{}/convol_python'.format(
                self.parent_path,
                self.data_type,
                self.sub_folder_name[:4])):
            os.mkdir('{}/{}/{}/convol_python'
                     .format(self.parent_path,
                             self.data_type,
                             self.sub_folder_name[:4]))
        print(message)
        sio.savemat('{}/{}/{}/convol_python/gibbs_param_est.mat'.
                    format(self.parent_path, self.data_type,
                           self.sub_folder_name[:4]),
                    {
                        'mcmc_message': message,
                        'delta_1': params_list[0],
                        'delta_0': params_list[1],
                        'pres_chky_1': params_list[2],
                        'pres_chky_0': params_list[3],
                        'pres': params_list[4]
                    })

    def save_hmc_params_est_sim(self, sim_folder_name, message, params_list):
        if not os.path.exists('{}/SIM_files/{}/convol_python'.format(
                self.parent_path, sim_folder_name)):
            os.mkdir('{}/SIM_files/{}/convol_python'.format(self.parent_path, sim_folder_name))
        sio.savemat('{}/SIM_files/{}/convol_python/hmc_param_est.mat'.
                    format(self.parent_path, sim_folder_name),
                    {
                        'message': message,
                        'delta_1': params_list[0],
                        'delta_0': params_list[1],
                        'pres_chky_1': params_list[2],
                        'pres_chky_0': params_list[3],
                        'pres': params_list[4]
                    })

    def save_opt_params_est_sim(self, sim_folder_name, message, params_list):
        if not os.path.exists('{}/SIM_files/{}/convol_python'.format(
            self.parent_path, sim_folder_name
        )):
            os.mkdir('{}/SIM_files/{}/convol_python'.format(self.parent_path, sim_folder_name))
        sio.savemat('{}/SIM_files/{}/convol_python/opt_param_est.mat'.
                    format(self.parent_path, sim_folder_name),
                    {
                        'message': message,
                        'delta_1_opt': params_list[0],
                        'delta_0_opt': params_list[1],
                        'pres_chky_1_opt': params_list[2],
                        'pres_chky_0_opt': params_list[3],
                        'pres_opt': params_list[4],
                        'chisq_1_opt': params_list[5],
                        'chisq_0_opt': params_list[6]
                    })

    def import_hmc_params_est(self):

        mcmc_dat = sio.loadmat('{}/{}/{}/convol_python/hmc_param_est.mat'
                               .format(self.parent_path,
                                       self.data_type,
                                       self.sub_folder_name[:4]))

        mcmc_keys, _ = zip(*mcmc_dat.items())
        # print(mcmc_keys[3:])
        # mcmc_keys = list(mcmc_dat)
        return mcmc_dat

    def import_hmc_params_est_sim(self, sim_folder_name):

        mcmc_dat = sio.loadmat('{}/SIM_files/{}/convol_python/hmc_param_est.mat'
                               .format(self.parent_path, sim_folder_name))
        mcmc_keys, _ = zip(*mcmc_dat.items())
        # print(mcmc_keys[3:])
        # mcmc_keys = list(mcmc_dat)
        return mcmc_dat

    def import_opt_params_est_sim(self, sim_folder_name):

        opt_dat = sio.loadmat('{}/SIM_files/{}/convol_python/opt_param_est.mat'
                               .format(self.parent_path, sim_folder_name))
        opt_keys, _ = zip(*opt_dat.items())
        print(opt_keys[3:])

        return opt_dat

    def import_gibbs_params_est(self):

        gibbs_dat = sio.loadmat('{}/{}/{}/convol_python/gibbs_param_est.mat'
                                .format(self.parent_path,
                                        self.data_type,
                                        self.sub_folder_name[:4]))
        gibbs_keys, _ = zip(*gibbs_dat.items())
        print(gibbs_keys[3:])

        return gibbs_dat

    def partial_log_prob_eeg_conol_test_fn(
            self, delta_1_value, delta_0_value,
            pres_chky_1_value, pres_chky_0_value,
            letter_dim_test, repet_dim_test,
            eeg_signals_test, eeg_code_test
    ):
        design_x = self.rearrange.create_design_mat_gen_bayes_seq(repet_dim_test)
        # Generate delta_combined
        delta_combined = self.rearrange.tile_and_combine_delta(
            letter_dim_test, repet_dim_test, delta_1_value, delta_0_value)

        pres_chky_1_value = self.prior.convert_1d_array_to_upper_triangular(pres_chky_1_value)
        pres_chky_0_value = self.prior.convert_1d_array_to_upper_triangular(pres_chky_0_value)
        eta_1_value, _ = self.prior.generate_eta_and_compute_log_lhd(
            pres_chky_1_value, letter_dim_test, repet_dim_test, self.flash_sum)
        eta_0_value, _ = self.prior.generate_eta_and_compute_log_lhd(
            pres_chky_0_value, letter_dim_test, repet_dim_test, self.non_flash_sum)

        eta_combined = tf.transpose(tf.concat([eta_1_value, eta_0_value], axis=0),
                                    perm=[1, 0, 2])
        beta_combined = delta_combined + eta_combined
        # Use eeg_code_test to generate 36 different eeg_types
        dim_temp = letter_dim_test*repet_dim_test*self.num_rep
        eeg_code_test = np.reshape(eeg_code_test, [dim_temp])
        mvn_log_lhd = tf.zeros([1, letter_dim_test, repet_dim_test], dtype=self.DAT_TYPE)

        for _, target_letter in enumerate(self.letter_table):

            eeg_type_test = self.generate_eeg_type_from_letter_eeg_code(
                eeg_code_test, target_letter)
            id_beta = self.rearrange.create_permute_beta_id(letter_dim_test, repet_dim_test, eeg_type_test)
            beta_tilta = self.rearrange.create_joint_beta_tilta(
                letter_dim_test, repet_dim_test, beta_combined, id_beta, design_x)
            beta_tilta = tf.squeeze(beta_tilta, axis=-1)

            log_lhd_per_letter = tf.math.cumsum(
                tf.reduce_sum((eeg_signals_test-beta_tilta)**2, axis=1), axis=-1)

            indices = tf.cast(tf.linspace(1., repet_dim_test, repet_dim_test), dtype='int32')
            indices = (indices*self.num_rep+self.n_multiple-1)*self.flash_and_pause_length-1
            log_lhd_per_letter = tf.gather(log_lhd_per_letter, indices, axis=1)
            # print('log_lhd_per_letter has shape {}'.format(log_lhd_per_letter.shape))

            mvn_log_lhd = tf.concat([mvn_log_lhd, log_lhd_per_letter[tf.newaxis, ...]], axis=0)
        # print('mvn_log_lhd has shape {}'.format(mvn_log_lhd.shape))
        eeg_mvn_log_lhd = mvn_log_lhd[1:, ...]
        # print('eeg_mvn_arg_max_ has shape {}'.format(eeg_mvn_arg_max.shape))
        return eeg_mvn_log_lhd

    def log_prob_eeg_convol_pred_i(
            self, delta_1, delta_0,
            pres_mat_1, pres_mat_0,
            pres_lambda, eeg_signals_test,
            letter_dim_test, repet_dim_test,
            eeg_code_test, design_x
    ):

        batch_dim_1 = letter_dim_test * repet_dim_test * self.flash_sum
        batch_dim_0 = letter_dim_test * repet_dim_test * self.non_flash_sum

        # Generate intermediate beta
        std_mvn_rv = tfd.MultivariateNormalDiag(
            loc=tf.zeros([self.num_electrode, self.n_length]),
            scale_identity_multiplier=tf.ones([]))

        beta_1 = std_mvn_rv.sample(batch_dim_1)[..., tf.newaxis]
        beta_0 = std_mvn_rv.sample(batch_dim_0)[..., tf.newaxis]

        pres_mat_1_chol = tf.linalg.cholesky(pres_mat_1)
        cov_mat_1_half = tf.transpose(tf.linalg.inv(pres_mat_1_chol), [0, 2, 1])
        pres_mat_0_chol = tf.linalg.cholesky(pres_mat_0)
        cov_mat_0_half = tf.transpose(tf.linalg.inv(pres_mat_0_chol), [0, 2, 1])

        beta_1 = tf.matmul(cov_mat_1_half[tf.newaxis, ...], beta_1) + delta_1[tf.newaxis, ..., tf.newaxis]
        beta_0 = tf.matmul(cov_mat_0_half[tf.newaxis, ...], beta_0) + delta_0[tf.newaxis, ..., tf.newaxis]

        beta_combined = tf.squeeze(tf.concat([beta_1, beta_0], axis=0), axis=-1)
        # print('beta_combined has shape {}'.format(beta_combined.shape))
        mvn_log_lhd = tf.zeros([1, letter_dim_test, repet_dim_test], dtype=self.DAT_TYPE)

        for _, target_letter in enumerate(self.letter_table):
            eeg_type_test = self.generate_eeg_type_from_letter_eeg_code(
                eeg_code_test, target_letter)
            id_beta = self.rearrange.create_permute_beta_id(letter_dim_test, repet_dim_test, eeg_type_test)
            beta_tilta = self.rearrange.create_joint_beta_tilta(
                letter_dim_test, repet_dim_test, beta_combined, id_beta, design_x,
                channel_dim=self.num_electrode)
            beta_tilta = tf.squeeze(beta_tilta, axis=-1)
            # print('beta_tilta has shape {}'.format(beta_tilta.shape))

            log_lhd_per_letter = tf.math.cumsum(
                tf.reduce_sum(-pres_lambda[tf.newaxis, ...]/2*(eeg_signals_test - beta_tilta)**2, axis=1), axis=-1)
            # print('log_lhd_per_letter has shape {}'.format(log_lhd_per_letter.shape))

            indices = tf.cast(tf.linspace(1., repet_dim_test, repet_dim_test), dtype='int32')
            indices = (indices * self.num_rep + self.n_multiple - 1) * self.flash_and_pause_length - 1
            log_lhd_per_letter = tf.gather(log_lhd_per_letter, indices, axis=1)

            mvn_log_lhd = tf.concat([mvn_log_lhd, log_lhd_per_letter[tf.newaxis, ...]], axis=0)
        eeg_mvn_log_lhd = mvn_log_lhd[1:, ...]
        # print('eeg_mvn_arg_max_ has shape {}'.format(eeg_mvn_arg_max.shape))
        return eeg_mvn_log_lhd