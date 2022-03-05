# import sys
# sys.path.insert(0, './self_py_fun')
from self_py_fun.PreFun import *
plt.style.use('ggplot')
import csv
import seaborn as sns
sns.set_context('notebook')
# If running this file on the cluster, comment out the tensorflow-related functions and imports.


class ExistMLPred(EEGPreFun):

    def __init__(self, *args, **kwargs):
        super(ExistMLPred, self).__init__(*args, **kwargs)

    def import_sim_matlab_swlda_wts_train(
            self, trn_rep_dim, file_subscript, scenario_name
    ):

        folder_dir = '{}/swLDA/{}/sim_swlda_wts_train_{}_{}.mat'.format(
            self.parent_sim_output_path,
            scenario_name, trn_rep_dim, file_subscript
        )
        # print(folder_dir)

        swlda_wts = sio.loadmat(folder_dir)
        swlda_wts_keys, _ = zip(*swlda_wts.items())
        # print(swlda_wts_keys)

        return swlda_wts

    def import_eeg_matlab_swlda_wts_train(
            self, file_subscript, scenario_name, eeg_file_suffix,
            channel_dim=1, channel_id=1
    ):

        if channel_dim == 1:
            folder_dir = '{}/swLDA/{}/{}_swlda_wts_train_{}_channel_{}_{}.mat'\
                .format(self.parent_eeg_output_path, scenario_name,
                        self.sub_folder_name, file_subscript, channel_id, eeg_file_suffix)
        else:
            folder_dir = '{}/swLDA/{}/{}_swlda_wts_train_{}_{}_{}.mat'\
                .format(self.parent_eeg_output_path, scenario_name,
                        self.sub_folder_name, file_subscript, scenario_name, eeg_file_suffix)

        print(folder_dir)
        swlda_wts = sio.loadmat(folder_dir)
        swlda_wts_keys, _ = zip(*swlda_wts.items())
        # print(swlda_wts_keys)

        return swlda_wts

    @staticmethod
    def swlda_predict_y_prob(
            b, in_model, eeg_signals_trun
    ):
        b_inmodel = np.multiply(np.transpose(in_model), b)
        pred_prob = np.matmul(eeg_signals_trun, b_inmodel)
        return pred_prob

    def ml_predict(
            self, predict_prob, eeg_code,
            letter_dim, repetition_pred
    ):
        r"""
        :param predict_prob: 2d-array probability, (feature_vector_dim, 1)
        :param eeg_code: ultimately converted to 1d-array
        :param letter_dim: integer
        :param repetition_pred: integer
        :return:
        """
        assert predict_prob.shape == (letter_dim * repetition_pred * self.num_rep, 1), \
            print('Inconsistent dimension of predict_prob.')

        eeg_code = np.reshape(eeg_code, [self.num_letter * repetition_pred * self.num_rep])
        single_score_row = np.zeros([int(self.num_rep / 2), self.num_letter * repetition_pred])
        single_score_col = np.zeros([int(self.num_rep / 2), self.num_letter * repetition_pred])

        for i in range(int(self.num_rep / 2)):
            single_score_row[i, :] = predict_prob[np.where(eeg_code == i + 1)[0], 0]
            single_score_col[i, :] = predict_prob[np.where(eeg_code == i + self.row_column_length + 1)[0], 0]

        single_score_row = np.reshape(
            single_score_row, [int(self.num_rep / 2), self.num_letter, repetition_pred]
        )
        single_score_col = np.reshape(
            single_score_col, [int(self.num_rep / 2), self.num_letter, repetition_pred]
        )

        print('single_score_row has shape {}'.format(single_score_row.shape))
        # Compute the prediction based on single seq (row + col)
        arg_max_single_row = np.argmax(single_score_row, axis=0) + 1
        arg_max_single_col = np.argmax(single_score_col, axis=0) + self.row_column_length + 1

        # cumulative
        cum_score_row = np.cumsum(single_score_row, axis=-1)
        cum_score_col = np.cumsum(single_score_col, axis=-1)

        '''
        # 5 out of 7 or 5 out of 8
        _, num_letter_temp, _ = single_score_row.shape
        seq_fix = 5
        n_choose_k = np.array(list(itl.product(*[(0, 1) for i in range(repetition_pred)])))
        n_choose_k = np.copy(n_choose_k[np.sum(n_choose_k, axis=-1) == seq_fix, :])  # (21, 7) or (56, 8)
        n_k_val, _ = n_choose_k.shape
        cum_score_row = np.zeros([6, num_letter_temp, n_k_val])
        cum_score_col = np.zeros([6, num_letter_temp, n_k_val])
        for n_k_id in range(n_k_val):
            cum_score_row[..., n_k_id] = np.sum(single_score_row[..., np.where(n_choose_k[n_k_id, :] == 1)[0]], axis=-1)
            cum_score_col[..., n_k_id] = np.sum(single_score_col[..., np.where(n_choose_k[n_k_id, :] == 1)[0]], axis=-1)
        '''

        arg_max_cum_row = np.argmax(cum_score_row, axis=0) + 1
        arg_max_cum_col = np.argmax(cum_score_col, axis=0) + self.row_column_length + 1

        letter_single_mat = np.zeros([letter_dim, repetition_pred]).astype('<U1')
        letter_cum_mat = np.zeros([letter_dim, repetition_pred]).astype('<U1')

        for i in range(letter_dim):
            for j in range(repetition_pred):
                letter_single_mat[i, j] = self.determine_letter(arg_max_single_row[i, j], arg_max_single_col[i, j])
                letter_cum_mat[i, j] = self.determine_letter(arg_max_cum_row[i, j], arg_max_cum_col[i, j])

        ml_pred_dict = {
            "single": letter_single_mat,
            "cum": letter_cum_mat
        }
        return ml_pred_dict

    def ml_predict_entropy(
            self, predict_prob, eeg_code, screen_ids,
            letter_dim, repetition_pred
    ):
        assert predict_prob.shape == (letter_dim * repetition_pred * self.num_rep, 1), \
            print('Inconsistent dimension of predict_prob.')
        assert screen_ids.shape == (letter_dim, repetition_pred, repetition_pred), \
            print('Inconsistent dimension of screen indicator matrix.')

        eeg_code = np.reshape(eeg_code, [self.num_letter * repetition_pred * self.num_rep])
        single_score_row = np.zeros([int(self.num_rep / 2), self.num_letter * repetition_pred])
        single_score_col = np.zeros([int(self.num_rep / 2), self.num_letter * repetition_pred])

        for i in range(int(self.num_rep / 2)):
            single_score_row[i, :] = predict_prob[np.where(eeg_code == i + 1)[0], 0]
            single_score_col[i, :] = predict_prob[np.where(eeg_code == i + self.row_column_length + 1)[0], 0]

        single_score_row = np.reshape(
            single_score_row, [int(self.num_rep / 2), self.num_letter, repetition_pred]
        )  # (6, 19, 8)
        single_score_col = np.reshape(
            single_score_col, [int(self.num_rep / 2), self.num_letter, repetition_pred]
        )

        arg_max_single_row = np.argmax(single_score_row, axis=0) + 1
        arg_max_single_col = np.argmax(single_score_col, axis=0) + self.row_column_length + 1

        entropy_score_row = np.zeros_like(single_score_row)
        entropy_score_col = np.zeros_like(single_score_col)

        for l_id in range(letter_dim):
            for rep_id in range(repetition_pred):
                for rep_select_id in range(repetition_pred):
                    if screen_ids[l_id, rep_id, rep_select_id] == 1:
                        entropy_score_row[:, l_id, rep_id] = entropy_score_row[:, l_id, rep_id] + single_score_row[:, l_id, rep_select_id]
                        entropy_score_col[:, l_id, rep_id] = entropy_score_col[:, l_id, rep_id] + single_score_col[:, l_id, rep_select_id]
        arg_max_entropy_row = np.argmax(entropy_score_row, axis=0) + 1
        arg_max_entropy_col = np.argmax(entropy_score_col, axis=0) + self.row_column_length + 1

        letter_single_mat = np.zeros([letter_dim, repetition_pred]).astype('<U1')
        letter_entropy_mat = np.zeros([letter_dim, repetition_pred]).astype('<U1')

        for i in range(letter_dim):
            for j in range(repetition_pred):
                letter_single_mat[i, j] = self.determine_letter(arg_max_single_row[i, j], arg_max_single_col[i, j])
                letter_entropy_mat[i, j] = self.determine_letter(arg_max_entropy_row[i, j], arg_max_entropy_col[i, j])

        ml_prediction_dict = {
            "single": letter_single_mat,
            "cum": letter_entropy_mat
        }

        return ml_prediction_dict

    @staticmethod
    def compute_selected_sample_stats(
            signals_trun_sub, type_sub, in_model
    ):
        r"""
        :param signals_trun_sub: (sample_row, feature_col)
        :param type_sub: (sample_row,)
        :param in_model: (1, feature_col)
        :return:
        """
        signals_trun_sub = signals_trun_sub[:, in_model[0, :] == 1]

        signals_mean_1_sub = np.mean(signals_trun_sub[type_sub == 1], axis=0)
        signals_mean_0_sub = np.mean(signals_trun_sub[type_sub == 0], axis=0)
        signals_cov_sub = np.cov(signals_trun_sub, rowvar=False)

        return [signals_mean_1_sub, signals_mean_0_sub,
                signals_cov_sub, signals_trun_sub]

    def plot_swlda_select_feature(
            self, in_model, trn_repetition, scenario_name, sim_dat, channel_ids=None
    ):
        r"""
        :param inmodel: array_like, (1, num_electrode * n_length)
        :param sim_folder_name: string
        :param trn_repetition: integer or string
        :param scenario_name: string
        :param sim_dat: bool_like
        :param channel_ids: array_like
        :return: plot
        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        in_model = np.reshape(in_model, [channel_dim, self.n_length])
        if sim_dat:
            plot_pdf_dir = '{}/swLDA/{}/{}_swlda_select_train_{}.pdf'.format(
                self.parent_sim_output_path, scenario_name,
                self.sub_folder_name, trn_repetition
            )
        else:
            plot_pdf_dir = '{}/swLDA/{}/{}_swlda_select_train_{}.pdf'.format(
                self.parent_eeg_output_path, scenario_name,
                self.sub_folder_name, trn_repetition
            )
        plot_pdf = bpdf.PdfPages(plot_pdf_dir)
        # print(plot_pdf_dir)
        print('The number of features selected by swLDA is {}.'.format(np.sum(in_model)))

        # log-likelihood trace-plot and mean selection rate
        for i, e_id in enumerate(channel_ids):
            fig_alt = plt.figure(figsize=(10, 10))
            plt.plot(np.arange(self.n_length), in_model[i, :])
            plt.ylim(-0.1, 1.1)
            plt.title('channel ' + str(e_id + 1))
            plt.close()
            # plt.show()
            plot_pdf.savefig(fig_alt)
        plot_pdf.close()
        return 'plots done!'

    def save_exist_ml_pred_results(
            self, exist_ml_pred_dict, repet_num_fit, repet_num_pred,
            target_letter, exist_ml_name, scenario_name, file_subscript,
            train_bool=True, sim_dat_bool=True
    ):
        r"""
        :param exist_ml_pred_dict:
        :param repet_num_fit:
        :param repet_num_pred:
        :param target_letter:
        :param exist_ml_name:
        :param scenario_name:
        :param file_subscript:
        :param train_bool:
        :param sim_dat_bool:
        :return:
        """
        method_dir = "{}/{}/{}".format(
            self.parent_path,
            exist_ml_name,
            self.sub_name_short
        )
        try:
            os.mkdir(method_dir)
            print('Directory', method_dir, ' is created.')
        except FileExistsError:
            print('Directory ', method_dir, ' already exists.')

        if sim_dat_bool:
            method_dir = '{}/{}'.format(method_dir, self.sub_folder_name)
            try:
                os.mkdir(method_dir)
                print('Directory', method_dir, ' is created.')
            except FileExistsError:
                print('Directory ', method_dir, ' already exists.')

        method_dir = '{}/{}'.format(method_dir, scenario_name)
        try:
            os.mkdir(method_dir)
            print('Directory', method_dir, ' is created.')
        except FileExistsError:
            print('Directory ', method_dir, ' already exists.')

        if train_bool:
            file_dir = "{}/{}_train_{}_pred_train_{}_{}.csv".format(
                method_dir, self.sub_folder_name,
                repet_num_fit, repet_num_pred, file_subscript
            )
        else:
            file_dir = "{}/{}_train_{}_pred_test_{}_{}.csv".format(
                method_dir, self.sub_folder_name,
                repet_num_fit, repet_num_pred, file_subscript
            )

        single_pred = exist_ml_pred_dict['single']
        cum_pred = exist_ml_pred_dict['cum']

        task = 'w'
        with open(file_dir, task) as f:
            f_writer = csv.writer(f)
            l0 = ['trn_repetition', str(repet_num_fit)]
            f_writer.writerow(l0)
            l1 = ['Sequence id']
            l1.extend(list(np.arange(1, repet_num_pred+1)))
            f_writer.writerow(l1)

            f_writer.writerow(['Single sequence prediction:'])
            for i, letter_i in enumerate(target_letter):
                l_ij = ['Letter {}'.format(letter_i)]
                for j in range(repet_num_pred):
                    l_ij.append(single_pred[i, j])
                f_writer.writerow(l_ij)

            f_writer.writerow([' '])

            f_writer.writerow(['Cumulative sequence prediction:'])
            for i, letter_i in enumerate(target_letter):
                l_ij = ['Letter {}'.format(letter_i)]
                for j in range(repet_num_pred):
                    l_ij.append(cum_pred[i, j])
                f_writer.writerow(l_ij)

        return 'Saving results done!'

    def split_trunc_train_set_odd_even(
            self, eeg_signals_trun, eeg_type_3d, eeg_code_3d, rep_train_id, rep_test_id
    ):
        eeg_signals_trun_2 = np.reshape(
            eeg_signals_trun, [self.num_electrode,
                               self.num_letter,
                               self.num_repetition,
                               self.num_rep,
                               self.n_length]
        )
        # We pick the odd sequence for training and even sequence for testing (within TRN_file)
        eeg_signals_trun_2_odd = eeg_signals_trun_2[..., rep_train_id - 1, :, :]
        eeg_signals_trun_2_even = eeg_signals_trun_2[..., rep_test_id - 1, :, :]
        repet_num_train = len(rep_train_id)
        repet_num_test = len(rep_test_id)
        eeg_signals_trun_2_odd = np.reshape(
            eeg_signals_trun_2_odd, [self.num_electrode,
                                     self.num_letter * repet_num_train * self.num_rep,
                                     self.n_length]
        )
        eeg_signals_trun_2_odd = np.transpose(eeg_signals_trun_2_odd, [1, 0, 2])
        eeg_signals_trun_2_even = np.reshape(
            eeg_signals_trun_2_even, [self.num_electrode,
                                      self.num_letter * repet_num_test * self.num_rep,
                                      self.n_length]
        )
        eeg_signals_trun_2_even = np.transpose(eeg_signals_trun_2_even, [1, 0, 2])

        eeg_type_odd = eeg_type_3d[:, rep_train_id-1, :]
        eeg_type_even = eeg_type_3d[:, rep_test_id-1, :]
        eeg_code_odd = eeg_code_3d[:, rep_train_id-1, :]
        eeg_code_even = eeg_code_3d[:, rep_test_id-1, :]

        return [eeg_signals_trun_2_odd, eeg_type_odd, eeg_code_odd,
                eeg_signals_trun_2_even, eeg_type_even, eeg_code_even]

    def exist_ml_fit_predict(
            self, ml_obj, signals_trun, eeg_code_1d,
            target_letters, repet_num_fit, repet_num_pred,
            method_name, scenario_name, file_subscript,
            train_bool, sim_dat_bool
    ):
        r"""
        :param ml_obj:
        :param signals_trun:
        :param eeg_type_1d: (feature length, 1)
        :param eeg_code_1d: (feature length, 1)
        :param target_letters:
        :param repet_num_fit:
        :param repet_num_pred:
        :param method_name:
        :param scenario_name:
        :param file_subscript:
        :param train_bool:
        :param sim_dat_bool:
        :return:
        """

        letter_dim = len(target_letters)
        # ml_obj.fit(signals_trun, eeg_type_1d[0, :])
        y_pred_train = ml_obj.predict_proba(signals_trun)[:, 1, np.newaxis]

        exist_ml_dict = self.ml_predict(
            y_pred_train, eeg_code_1d[0, :], letter_dim, repet_num_pred
        )
        self.save_exist_ml_pred_results(
            exist_ml_dict, repet_num_fit, repet_num_pred,
            target_letters, method_name, scenario_name, file_subscript,
            train_bool, sim_dat_bool
        )
        print('Exist ML results have been saved.')
        return exist_ml_dict

