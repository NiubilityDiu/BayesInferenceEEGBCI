import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.EEGPreFun import *
plt.style.use('ggplot')
import csv
# import tensorflow as tf
# import tensorflow_probability as tfp
import seaborn as sns
# tfd = tfp.distributions
sns.set_context('notebook')
# If running this file on the cluster, pay attention to the tensorflow comment section!


class SWLDAPred(EEGPreFun):

    def __init__(self, *args, **kwargs):
        super(SWLDAPred, self).__init__(*args, **kwargs)

    def import_matlab_swlda_wts_trn_repetition(
            self, trn_rep_dim, file_subscript):
        folder_dir = '{}/{}/{}/swlda_matlab/{}_swlda_wts_trn_rep_{}_{}.mat'\
            .format(self.parent_path,
                    self.data_type,
                    self.sub_folder_name[:4],
                    self.sub_folder_name,
                    trn_rep_dim,
                    file_subscript)

        swlda_wts = sio.loadmat(folder_dir)
        swlda_wts_keys, _ = zip(*swlda_wts.items())
        # print(swlda_wts_keys)

        return swlda_wts

    def swlda_pred(self, b, inmodel,
                   trn_rep_dim, eeg_signals_trun, eeg_code):

        r"""

        b:
        inmodel:
        trn_rep_dim:
        eeg_signals_trun:
        eeg_code:

        return:
        """

        b_inmodel = np.multiply(np.transpose(inmodel), b)
        predict_y = np.matmul(eeg_signals_trun, b_inmodel)
        eeg_code = np.reshape(eeg_code, [self.num_letter * self.num_repetition * self.num_rep])

        cumulative_score_row_code = np.zeros([int(self.num_rep / 2), self.num_letter * self.num_repetition])
        cumulative_score_col_code = np.zeros([int(self.num_rep / 2), self.num_letter * self.num_repetition])
        for i in range(int(self.num_rep / 2)):
            cumulative_score_row_code[i, :] = predict_y[np.where(eeg_code == i + 1)[0], 0]
            cumulative_score_col_code[i, :] = predict_y[np.where(eeg_code == i + self.row_column_length + 1)[0], 0]

        cumulative_score_row_code = np.reshape(cumulative_score_row_code,
                                               [int(self.num_rep / 2),
                                                self.num_letter,
                                                self.num_repetition])

        cumulative_score_col_code = np.reshape(cumulative_score_col_code,
                                               [int(self.num_rep / 2),
                                                self.num_letter,
                                                self.num_repetition])

        # Split the cumsum by the training/testing set
        [cumulative_score_row_train,
         cumulative_score_row_test] = np.split(cumulative_score_row_code,
                                               [trn_rep_dim], axis=-1)
        [cumulative_score_col_train,
         cumulative_score_col_test] = np.split(cumulative_score_col_code,
                                               [trn_rep_dim], axis=-1)

        cumulative_score_row_train = np.cumsum(cumulative_score_row_train, axis=-1)
        cumulative_score_col_train = np.cumsum(cumulative_score_col_train, axis=-1)

        argmax_row_train = np.argmax(cumulative_score_row_train, axis=0) + 1
        argmax_col_train = np.argmax(cumulative_score_col_train, axis=0) + self.row_column_length + 1

        cumulative_score_row_test = np.cumsum(cumulative_score_row_test, axis=2)
        cumulative_score_col_test = np.cumsum(cumulative_score_col_test, axis=2)

        argmax_row_test = np.argmax(cumulative_score_row_test, axis=0) + 1
        argmax_col_test = np.argmax(cumulative_score_col_test, axis=0) + self.row_column_length + 1

        argmax_row = np.concatenate([argmax_row_train, argmax_row_test], axis=-1)
        argmax_col = np.concatenate([argmax_col_train, argmax_col_test], axis=-1)

        pred_letter_mat = []
        for i in range(self.num_letter):
            for j in range(self.num_repetition):
                pred_letter_mat.append(self.determine_letter(argmax_row[i, j], argmax_col[i, j]))

        pred_letter_mat = np.array(pred_letter_mat)
        pred_letter_mat = np.reshape(pred_letter_mat, [self.num_letter, self.num_repetition])
        return pred_letter_mat

    def compute_selected_sample_stats(
            self, eeg_signals_trun_sub, eeg_type_sub, inmodel
    ):
        r"""

        eeg_signals_trun_sub:
        eeg_type_sub:
        inmodel:
        return:

        """
        section_num, _, _ = eeg_signals_trun_sub.shape
        eeg_signals_trun_sub = np.reshape(eeg_signals_trun_sub,
                                          [section_num, self.num_electrode * self.n_length])
        eeg_signals_trun_sub = eeg_signals_trun_sub[:, inmodel[0, :] == 1]

        eeg_signals_mean_1_sub = np.mean(eeg_signals_trun_sub[eeg_type_sub == 1, :], axis=0)
        eeg_signals_mean_0_sub = np.mean(eeg_signals_trun_sub[eeg_type_sub == 0, :], axis=0)
        eeg_signals_cov_sub = np.cov(eeg_signals_trun_sub, rowvar=False)

        return [
            eeg_signals_mean_1_sub.astype(self.DAT_TYPE),
            eeg_signals_mean_0_sub.astype(self.DAT_TYPE),
            eeg_signals_cov_sub.astype(self.DAT_TYPE)
        ]
    '''
    def swlda_produce_two_step_estimation(
            self, eeg_signals_trun_sub, eeg_code,
            eeg_signals_trun_t_mean, eeg_signals_trun_nt_mean,
            eeg_signals_trun_cov,
            trn_repetition):

        r"""
        eeg_signals_trun_sub: array_like
            the truncated segments of eeg signals,
            should have dimension
            (letter_dim * repet_dim * self.num_rep, number of features)
            For swlda, we don't consider the num_electrode dimension, need reshape after inputing the raw data.
        eeg_code: 3d-array like
            the eeg_code of the entire training set,
            should have dimension (self.num_letter, self.num_repetition, self.num_rep)
            For swlda, we need to reshape it to the 1d-array.
        eeg_signals_trun_t_mean: 1d-array like
        eeg_signals_trun_nt_mean: 1d-array like
        trn_repetition: integer
            the number of training sequences

        return:
            an array containing the prediction letter matrix,
            should have the dimension (self.num_letter, self.num_repetition)
        """

        mvn_1_obj = tfd.MultivariateNormalFullCovariance(
            loc=eeg_signals_trun_t_mean,
            covariance_matrix=eeg_signals_trun_cov,
            name='mvn_1_obj')

        mvn_0_obj = tfd.MultivariateNormalFullCovariance(
            loc=eeg_signals_trun_nt_mean,
            covariance_matrix=eeg_signals_trun_cov,
            name='mvn_0_obj')

        l_mvn_1_value = mvn_1_obj.log_prob(eeg_signals_trun_sub)
        l_mvn_0_value = mvn_0_obj.log_prob(eeg_signals_trun_sub)

        # Two-step Estimation:
        l_mvn_1_ordered = []
        l_mvn_0_ordered = []
        eeg_code_flat = np.reshape(eeg_code, [self.num_letter * self.num_repetition * self.num_rep])

        with tf.compat.v1.Session() as sess:
            [l_mvn_1_value_, l_mvn_0_value_] = sess.run([l_mvn_1_value, l_mvn_0_value])

        for i in range(self.num_rep):
            l_mvn_1_ordered.append(l_mvn_1_value_[eeg_code_flat == i + 1])
            l_mvn_0_ordered.append(l_mvn_0_value_[eeg_code_flat == i + 1])

        l_mvn_1_ordered = np.stack(l_mvn_1_ordered, axis=1)
        l_mvn_0_ordered = np.stack(l_mvn_0_ordered, axis=1)
        l_mvn_1_ordered = np.reshape(l_mvn_1_ordered, [self.num_letter, self.num_repetition, self.num_rep])
        l_mvn_0_ordered = np.reshape(l_mvn_0_ordered, [self.num_letter, self.num_repetition, self.num_rep])

        log_lhd_row = np.zeros([self.num_letter, self.num_repetition, int(self.num_rep / 2)], dtype=self.DAT_TYPE)
        log_lhd_col = np.zeros([self.num_letter, self.num_repetition, int(self.num_rep / 2)], dtype=self.DAT_TYPE)
        row_indices = np.arange(1, self.row_column_length + 1)
        col_indices = np.arange(self.row_column_length + 1, self.num_rep + 1)
        for i in range(1, self.row_column_length + 1):
            row_not_i = np.setdiff1d(row_indices, i)
            log_lhd_row[:, :, i - 1] = l_mvn_1_ordered[:, :, i - 1] + \
                                       np.sum(l_mvn_0_ordered[:, :, row_not_i - 1], axis=2)

        for j in range(self.row_column_length + 1, self.num_rep + 1):
            col_not_j = np.setdiff1d(col_indices, j)
            log_lhd_col[:, :, j - self.row_column_length - 1] = l_mvn_1_ordered[:, :, j - 1] + \
                                       np.sum(l_mvn_0_ordered[:, :, col_not_j - 1], axis=2)

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
        argmax_col_id += self.row_column_length + 1
        argmax_row_col_id = np.stack([argmax_row_id, argmax_col_id], axis=2)

        letter_pred_matrix = []
        for trn_letter_id in range(self.num_letter):
            letter_pred_matrix_l = []
            for rep_id in range(self.num_repetition):
                letter_pred_matrix_l.append(
                    self.determine_letter(*argmax_row_col_id[trn_letter_id, rep_id, :]))
            letter_pred_matrix_l = np.stack(letter_pred_matrix_l, axis=0)
            letter_pred_matrix.append(letter_pred_matrix_l)
        letter_pred_matrix = np.stack(letter_pred_matrix, axis=0)

        return letter_pred_matrix
    '''

    def save_swlda_pred_results(self, new_swlda_result,
                                sub_folder_name, trn_rep_dim):
        r'''

        :param new_swlda_result:
        :param sub_folder_name:
        :param trn_rep_dim:
        :return:
        '''

        file_dir = "{}/EEGswLDA/{}_swlda_pred.csv".format(self.parent_path, sub_folder_name)
        assert 1 <= trn_rep_dim <= self.num_repetition, print('wrong training repetition dim!')
        if trn_rep_dim == 1:
            task = 'w'
        else:
            task = 'a'
        with open(file_dir, task) as f:
            f_writer = csv.writer(f)
            if task == "w":
                l0 = ['Testing sequence starts from column {}, '
                      'no testing for the last row.'
                      .format(trn_rep_dim+1)]
                f_writer.writerow(l0)
                l1 = ['# of training sequence']
                l1.extend([i for i in range(1, self.num_repetition + 1)])
                f_writer.writerow(l1)

                l2 = [trn_rep_dim]
                l2.extend(new_swlda_result)
                f_writer.writerow(l2)
            else:
                l2 = [trn_rep_dim]
                l2.extend(new_swlda_result)
                f_writer.writerow(l2)
