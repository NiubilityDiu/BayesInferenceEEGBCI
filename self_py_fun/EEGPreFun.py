import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.EEGGenFun import *
import matplotlib.backends.backend_pdf as bpdf


class EEGPreFun(EEGGeneralFun):

    # data_type can take values from TRN_files, NOC_files, or TEST_files
    def __init__(self, data_type, sub_folder_name,
                 *args, **kwargs):
        super(EEGPreFun, self).__init__(*args, **kwargs)

        self.Fs = self.sampling_rate
        self.sub_folder_name = sub_folder_name
        self.eeg_file_name = self.sub_folder_name + '_16_electrodes'
        self.data_type = data_type

    def print_sub_trn_info(self, trn_repetition):
        # Print out the subject we are making inference.
        print('This is subject {}.'.format(self.sub_folder_name))
        print('We are using {} repetitions for training purpose.'.format(trn_repetition))

    def import_eeg_dat(self):
        eeg_dat = sio.loadmat('{}/{}/{}/{}.mat'
                              .format(self.parent_path, self.data_type,
                                      self.sub_folder_name[:4], self.eeg_file_name))

        eeg_keys, _ = zip(*eeg_dat.items())
        # print(eeg_keys)
        eeg_signals = eeg_dat['signal']
        eeg_code = eeg_dat['Stim_Code']
        eeg_type = eeg_dat['Stim_Type']
        eeg_pis = eeg_dat['Stim_PIS'][:, 0]

        return [eeg_signals, eeg_code, eeg_type, eeg_pis]

    # It seems that I can import and extract specific features from raw dataset
    # via python only, no need to use MATLAB any more.
    def import_frt_file(self):
        eeg_frt = sio.loadmat('{}/{}/{}/{}.mat'
                              .format(self.parent_path, self.data_type,
                                      self.sub_folder_name[:4], self.eeg_file_name))
        # First level dict.keys()
        frt_signals = eeg_frt['signal']
        frt_code = eeg_frt['Stim_Code']
        frt_type = eeg_frt['Stim_Type']
        frt_pis = eeg_frt['Stim_PIS']
        frt_letters = list(eeg_frt['frt_letters'][:,0])
        ls_weights = eeg_frt['ls_weights']

        return [frt_signals, frt_code, frt_type, frt_pis, frt_letters, ls_weights]

    # def compute_trn_seq_length(self, trn_repetition):
    #     return int((self.num_rep*trn_repetition+self.n_multiple-1)*self.flash_and_pause_length)

    # Determine eeg_code_subset, eeg_type_subset from this function
    def truncate_raw_sequence(self, eeg_pis, eeg_signal, eeg_code, eeg_type):
        pis_1_num, _ = eeg_signal[eeg_pis == 1, :].shape
        print('eeg_pis = 1 has average length {}'.format(int(pis_1_num / self.num_letter)))

        pis_2_num, _ = eeg_signal[eeg_pis == 2, :].shape
        print('eeg_pis = 2 has average length {}'.format(int(pis_2_num / self.num_letter)))

        pis_3_num, _ = eeg_signal[eeg_pis == 3, :].shape
        print('eeg_pis = 3 has average length {}'.format(int(pis_3_num / self.num_letter)))

        eeg_signals_subset = eeg_signal[np.logical_or(eeg_pis == 2, eeg_pis == 3), :]
        eeg_code_subset = eeg_code[eeg_pis == 2, :]
        eeg_type_subset = eeg_type[eeg_pis == 2, :]
        # print('eeg_code_subset has shape {}'.format(eeg_code_subset.shape))
        row_num, num_electrode = eeg_signals_subset.shape
        print('eeg_signals after pis == 2 & pis == 3 has shape {}'.format(eeg_signals_subset.shape))
        single_seq_length = int(row_num / self.num_letter)
        print('single_seq_length for eeg signals only = {}'.format(single_seq_length))
        eeg_signals_subset = np.reshape(eeg_signals_subset, [self.num_letter, single_seq_length, num_electrode])

        # row_num_2 and single_seq_length_2 are valid for code and type without any additional noise.
        single_seq_length_2 = int(pis_2_num / self.num_letter)
        eeg_code_subset = np.reshape(eeg_code_subset,
                                     [self.num_letter, self.num_repetition,
                                      self.num_rep, self.flash_and_pause_length])
        eeg_code_subset = eeg_code_subset[:, :, :, 0]
        eeg_type_subset = np.reshape(eeg_type_subset,
                                     [self.num_letter, self.num_repetition,
                                      self.num_rep, self.flash_and_pause_length])
        eeg_type_subset = eeg_type_subset[:, :, :, 0]
        print('eeg_signals_subset prior to truncation has shape {}'.format(eeg_signals_subset.shape))
        print('eeg_code_subset has shape {}'.format(eeg_code_subset.shape))
        print('eeg_type_subset has shape {}'.format(eeg_type_subset.shape))

        num_repetition_val = int(single_seq_length_2 / self.flash_and_pause_length / self.num_rep)
        print('manual num_repetition = {}'.format(num_repetition_val))
        last_3_index = single_seq_length_2 + 160
        # This value has fixed formula as long as the number of 3 per letter epoch is smaller than 192.
        eeg_signals_subset = eeg_signals_subset[:, :last_3_index, :]
        return eeg_signals_subset, eeg_code_subset, eeg_type_subset

    # First, we apply moving average window filter and then down sample by the equivalent values
    # Or the decimation factor is the same as the window length
    def moving_average_decimate(self, eeg_signals_subset, eeg_code_subset, eeg_type_subset, dec_factor):
        _, seq_length, _ = eeg_signals_subset.shape
        move_window = np.ones([dec_factor]) / dec_factor
        move_average_signal = np.zeros([self.num_letter, self.num_electrode, seq_length])
        for trn_letter_id in range(self.num_letter):
            for ele_id in range(self.num_electrode):
                move_average_signal[trn_letter_id, ele_id, :] = \
                    np.convolve(eeg_signals_subset[trn_letter_id, :, ele_id], move_window, mode="same")
        col_index = np.linspace(start=0, stop=seq_length, num=int(seq_length/dec_factor),
                                endpoint=False, dtype=np.int64)
        move_average_signal_dec = move_average_signal[:, :, col_index]

        return move_average_signal_dec[..., np.newaxis]

    def save_truncated_signal(
            self, eeg_signals_subset, eeg_code_subset,
            eeg_type_subset
    ):
        sio.savemat('{}/{}/{}/{}_eeg_dat_raw_trun.mat'.
                    format(self.parent_path, self.data_type,
                           self.sub_folder_name[:4], self.sub_folder_name),
                    {
                        'eeg_signals': eeg_signals_subset,
                        'eeg_code': eeg_code_subset,
                        'eeg_type': eeg_type_subset
                    })

    def save_down_sample_signal(
            self, eeg_signals_down_sample, eeg_code_subset, eeg_type_subset):
        sio.savemat('{}/{}/{}/{}_eeg_dat_down.mat'.
                    format(self.parent_path, self.data_type,
                           self.sub_folder_name[:4], self.sub_folder_name),
                    {
                        'eeg_signals': eeg_signals_down_sample,
                        'eeg_code': eeg_code_subset,
                        'eeg_type': eeg_type_subset
                    })

    # Export it to MATLAB to perform SWLDA or any other classification method
    def save_trun_signal_1d_label(self, eeg_signals_trun, eeg_type, file_subscript):
        train_length, channel_dim, n_length = eeg_signals_trun.shape

        eeg_signals_trun = np.reshape(
            eeg_signals_trun, [train_length, channel_dim*n_length])
        # eeg_signals_trun = np.transpose(eeg_signals_trun)
        # print('eeg_signals_trun has shape {}'.format(eeg_signals_trun.shape))

        # Convert 3d eeg_type to 1d
        eeg_type = np.reshape(eeg_type, [self.num_letter * self.num_repetition * self.num_rep])
        print(eeg_type[:self.num_rep])

        # Convert 0-1 coding to -1-to-1 coding scheme
        eeg_type = np.copy((eeg_type - 0.5)*2)
        print(eeg_type[:self.num_rep])
        sio.savemat('{}/{}/{}/{}_eeg_dat_matlab_{}.mat'
                    .format(self.parent_path, self.data_type,
                            self.sub_folder_name[:4], self.sub_folder_name,
                            file_subscript),
                    {
                        'signals': eeg_signals_trun,
                        'label': eeg_type
                    })

    # https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
    # https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    # https://towardsdatascience.com/significance-of-acf-and-pacf-plots-in-time-series-analysis-2fa11a5d10a8

    def import_eeg_processed_dat(self, file_subscript, reshape_to_1d):

        r"""
        file_subscript: string
        reshape_to_1d: bool

        return:
            A list of three arrays, including
                eeg_signals, with shape (num_letter, num_electrode, seq_length)
                eeg_code, with shape (num_letter, num_repetition, num_rep)
                eeg_type, with shape (num_letter, num_repetition, num_rep)
        """
        file_path = '{}/{}/{}/{}_eeg_dat_{}.mat'\
            .format(self.parent_path, self.data_type,
                    self.sub_folder_name[:4], self.sub_folder_name,
                    file_subscript)
        print(file_path)
        eeg_dat = sio.loadmat(file_path)

        eeg_keys, _ = zip(*eeg_dat.items())
        # print(eeg_keys)
        eeg_signals = eeg_dat['eeg_signals']
        eeg_code = eeg_dat['eeg_code']
        eeg_type = eeg_dat['eeg_type']

        if reshape_to_1d:
            eeg_code = np.reshape(eeg_code, [self.num_letter*self.num_repetition*self.num_rep])
            eeg_type = np.reshape(eeg_type, [self.num_letter*self.num_repetition*self.num_rep])
        return [eeg_signals, eeg_code, eeg_type]

    def create_truncate_segment(self, eeg_signals_subset, repetition_dim):
        eeg_signals_subset = np.transpose(eeg_signals_subset)  # with shape (16, seq_length)
        # print('eeg_signals_subset has shape {}'.format(eeg_signals_subset.shape))
        eeg_signals_subset_trun = []
        total_z_num = repetition_dim * self.num_rep
        for i in range(total_z_num):
            low_bound = i * int(self.flash_and_pause_length)
            upp_bound = low_bound + self.n_length
            temp_i = eeg_signals_subset[:, low_bound:upp_bound]
            # print('temp_i has shape {}'.format(temp_i.shape))
            eeg_signals_subset_trun.append(temp_i)
        eeg_signals_subset_trun = np.stack(eeg_signals_subset_trun, axis=1)
        return eeg_signals_subset_trun

    def create_truncate_segment_batch(
            self, eeg_signals, eeg_type, letter_dim, trn_repetition):
        r"""
        args:
        -----
            eeg_signals: array_like
                should have the input shape (letter_dim, channel_dim, seq_length)
            eeg_type: None or 3d-array
                with shape (letter_dim, rep_dim, num_rep)
            letter_dim: integer
            trn_repetition: integer
                should not be greater than num_repetition/rep_dim

        return:
        -----
            A tuple containing two elements:
            1. truncated eeg signal segments,
            should have the input shape (trn_repetition * num_rep * letter_dim, channel_dim, n_length)
            2. the 1d-array of eeg_type of the training set.
        """

        total_rep = trn_repetition * self.num_rep
        eeg_signals_trun = []
        for i in range(total_rep):
            low_bound = i * int(self.flash_and_pause_length)
            upp_bound = low_bound + self.n_length
            eeg_signals_subset = eeg_signals[..., low_bound:upp_bound]
            # print('eeg_signals_subset has shape {}'.format(eeg_signals_subset.shape))
            eeg_signals_trun.append(eeg_signals_subset)
        eeg_signals_trun = np.stack(eeg_signals_trun, axis=1)
        # print('eeg_signals_trun has shape {}'.format(eeg_signals_trun.shape))
        eeg_signals_trun = np.reshape(eeg_signals_trun, [self.num_letter * total_rep,
                                                         self.num_electrode,
                                                         self.n_length])
        eeg_type_sub = 0
        if eeg_type is not None:
            eeg_type_sub = np.reshape(eeg_type[:letter_dim, :trn_repetition, :],
                                  [letter_dim * trn_repetition * self.num_rep])

        return eeg_signals_trun, eeg_type_sub

    '''
    @staticmethod
    # Use FullCovariance module to compute log-likelihood
    def compute_log_lhd_sub(mean_fn, full_cov_mat, name, input_data, reduce_sum_axis):
        mvn_sub_obj = tfd.MultivariateNormalFullCovariance(
            loc=mean_fn,
            covariance_matrix=full_cov_mat,
            name=name
        )
        assert reduce_sum_axis == 1 or reduce_sum_axis == 0, \
            print('reduce_sum_axis should only be either 0 or 1!')
        # reduce_sum_axis = 1 implies that the sub_grp_index is the same across all channels
        if reduce_sum_axis == 1:
            l_mvn_sub_value = tf.reduce_sum(mvn_sub_obj.log_prob(input_data),
                                            axis=reduce_sum_axis)
        else:
            l_mvn_sub_value = mvn_sub_obj.log_prob(input_data)
        # print('l_mvn_sub_value has shape {}'.format(l_mvn_sub_value.shape))
        return l_mvn_sub_value

    # Sum up different sections log-likelihood
    # (if direct concatenation leads to failed Cholesky decomposition)
    def sum_log_lhd_sub(self, mean_fn_list, full_cov_mat_list, name, input_data_list, reduce_sum_axis):
        assert len(mean_fn_list) == len(full_cov_mat_list) == len(input_data_list)
        l_mvn_sub_sum = []
        sec_num = len(mean_fn_list)
        for i in range(sec_num):
            l_mvn_sub_i = self.compute_log_lhd_sub(mean_fn_list[i], full_cov_mat_list[i],
                                                   name+str(i+1), input_data_list[i],
                                                   reduce_sum_axis)
            l_mvn_sub_sum.append(l_mvn_sub_i)
        l_mvn_sub_sum = tf.stack(l_mvn_sub_sum, axis=1)
        l_mvn_sub_sum = tf.reduce_sum(l_mvn_sub_sum, axis=1)
        print('l_mvn_sub_sum has shape {}'.format(l_mvn_sub_sum.shape))
        return l_mvn_sub_sum
    '''

    def produce_trun_mean_cov_subset(self, eeg_signals_trun, eeg_type_sub):

        r"""
        args:
        -----
            eeg_signals_trun: 3d-array
                (stimulus_length, channel_dim, n_length)
            eeg_type_sub: 1d-array
                (letter_dim * repet_dim * num_rep,)

        return:
        -----
            A list of 4 arrays including
                target_mean, (num_electrode, n_length)
                non_target_mean, (num_electrode, n_length)
                target_cov, (num_electrode, n_length, n_length)
                non_target_cov, (num_electrode, n_length, n_length)
        """

        eeg_signals_trun_sub_t = eeg_signals_trun[eeg_type_sub == 1, :, :]
        eeg_signals_trun_sub_nt = eeg_signals_trun[eeg_type_sub == 0, :, :]

        # Examine sample mean function (under sub-setting)
        eeg_signals_trun_t_mean = np.mean(eeg_signals_trun_sub_t, axis=0)
        eeg_signals_trun_nt_mean = np.mean(eeg_signals_trun_sub_nt, axis=0)

        # Examine sample covariance matrix
        eeg_signals_trun_t_cov = np.stack([np.cov(eeg_signals_trun_sub_t[:, i, :], rowvar=False)
                                           for i in range(self.num_electrode)], axis=0)
        eeg_signals_trun_nt_cov = np.stack([np.cov(eeg_signals_trun_sub_nt[:, i, :], rowvar=False)
                                           for i in range(self.num_electrode)], axis=0)

        return [eeg_signals_trun_t_mean, eeg_signals_trun_nt_mean,
                eeg_signals_trun_t_cov, eeg_signals_trun_nt_cov]

    def produce_mean_covariance_plots(self, mu_1, mu_0,
                                      cov_1, cov_0,
                                      var_key, file_subscript):
        # Save all mean function together in a pdf
        plot_parental_dir = '{}/{}/{}'.format(self.parent_path,
                                              self.data_type,
                                              self.sub_folder_name[:4])
        if mu_1 is not None and mu_0 is not None:
            mean_fn_pdf = bpdf.PdfPages('{}/{}_mean_{}.pdf'
                                        .format(plot_parental_dir,
                                                var_key,
                                                file_subscript))
            for ele_id in range(self.num_electrode):
                fig = plt.figure(figsize=(12, 10))
                plt.plot(self.time_range, mu_1[ele_id, :], label="target")
                plt.plot(self.time_range, mu_0[ele_id, :], label="non-target")
                plt.hlines(y=0, xmin=self.time_range[0], xmax=self.time_range[-1])
                plt.xlabel('Time (ms)')
                plt.ylabel('Mean Estimation')
                plt.title('{}, Channel {}'.format(self.sub_folder_name, ele_id + 1))
                plt.legend(loc='upper right')
                mean_fn_pdf.savefig(fig)
                # plt.show()
                plt.close()
            mean_fn_pdf.close()

        if cov_1 is not None and cov_0 is not None:

            # Save all sample covariance matrix together in a pdf
            sample_cov_t_pdf = bpdf.PdfPages('{}/{}_cov_target_{}.pdf'
                                             .format(plot_parental_dir,
                                                     var_key,
                                                     file_subscript))
            for i in range(self.num_electrode):
                # Common configuration
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                X, Y = np.meshgrid(self.time_range, self.time_range[::-1])

                fig1 = plt.figure(figsize=(12, 10))
                ax1 = fig1.add_axes([left, bottom, width, height])
                Z1 = cov_1[i, :, :]
                cp1 = plt.contourf(X, Y, Z1)
                fig1.colorbar(cp1)
                ax1.set_title('Target Contour, Channel ' + str(i + 1))
                ax1.set_xlabel('Time (ms)')
                ax1.set_ylabel('Time (ms)')
                sample_cov_t_pdf.savefig(fig1)
                plt.close()
            sample_cov_t_pdf.close()

            sample_cov_nt_pdf = bpdf.PdfPages('{}/{}_cov_non_target_{}.pdf'
                                              .format(plot_parental_dir,
                                                      var_key,
                                                      file_subscript))
            for i in range(self.num_electrode):
                # Common configuration
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                X, Y = np.meshgrid(self.time_range, self.time_range[::-1])

                fig0 = plt.figure(figsize=(7, 6))
                ax0 = fig0.add_axes([left, bottom, width, height])
                Z0 = cov_0[i, :, :]
                cp0 = plt.contourf(X, Y, Z0)
                fig0.colorbar(cp0)

                ax0.set_title('Non-target Contour, Channel ' + str(i + 1))
                ax0.set_xlabel('Time (ms)')
                ax0.set_ylabel('Time (ms)')
                sample_cov_nt_pdf.savefig(fig0)
                plt.close()
            sample_cov_nt_pdf.close()

