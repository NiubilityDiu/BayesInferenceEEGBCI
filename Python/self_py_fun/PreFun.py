import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.GeneralFun import *
import matplotlib.backends.backend_pdf as bpdf
import csv


class EEGPreFun(EEGGeneralFun):

    def __init__(self, data_type, sub_folder_name, sub_name_short='K101',
                 *args, **kwargs):
        super(EEGPreFun, self).__init__(*args, **kwargs)
        self.sub_folder_name = sub_folder_name
        self.sub_name_short = sub_name_short
        # self.eeg_file_name = self.sub_folder_name + '_16_electrodes'
        self.data_type = data_type
        self.parent_sim_output_path = '{}/{}/{}/{}'.format(
            self.parent_path, self.data_type, self.sub_name_short, self.sub_folder_name
        )
        self.parent_eeg_output_path = '{}/{}/{}'.format(
            self.parent_path, self.data_type, self.sub_name_short
        )
        """
        'data_type' takes values from TRN_files, NOC_files, TEST_files or SIM_files
        'sub_folder_name' refers to the official name
        'sub_name_short' refers to 'sim' in specific.
        
         may change in the future, mark here!
        """

    def create_sim_dataset_folder(self, sub_name_short):
        dir_name = '{}/{}/{}'.format(
            self.parent_path, self.data_type, sub_name_short
        )
        try:
            os.mkdir(dir_name)
            print('Directory', dir_name, ' is created.')
        except FileExistsError:
            print('Directory ', dir_name, ' already exists.')

    def create_method_folder(self, method_name, scenario_name=None, sim_dat=True):
        r"""
        :param method_name: string, should follow CamelCase convention
        :param scenario_name: string, scenario_name for simulation study purpose
        :param sim_dat: bool
        :return: create a new directory to save results associated with the method_name
        """
        if sim_dat:
            dir_name = '{}/{}'.format(self.parent_sim_output_path, method_name)

            try:
                os.mkdir(dir_name)
                print('Directory', dir_name, ' is created.')
            except FileExistsError:
                print('Directory ', dir_name, ' already exists.')

            if scenario_name is not None:
                dir_name = '{}/{}/{}'.format(
                    self.parent_sim_output_path, method_name, scenario_name
                )

                try:
                    os.mkdir(dir_name)
                    print('Directory', dir_name, ' is created.')
                except FileExistsError:
                    print('Directory ', dir_name, ' already exists.')
        else:
            dir_name = '{}/{}'.format(self.parent_eeg_output_path, method_name)

            try:
                os.mkdir(dir_name)
                print('Directory', dir_name, ' is created.')
            except FileExistsError:
                print('Directory ', dir_name, ' already exists.')

            if scenario_name is not None:
                dir_name = '{}/{}/{}'.format(
                    self.parent_eeg_output_path, method_name, scenario_name
                )

                try:
                    os.mkdir(dir_name)
                    print('Directory', dir_name, ' is created.')
                except FileExistsError:
                    print('Directory ', dir_name, ' already exists.')

    def import_eeg_dat(self, eeg_file_name, show_dim_bool=True):
        r"""
        :param show_dim_bool: bool_like, whether show the dimension of the raw eeg dataset
        :return: A list of raw signals including eeg_signals, eeg_code, eeg_type, and eeg_pis.

        eeg_signals: (total_seq_length, 16), 16-channel time sequence

        eeg_code: (total_seq_length, 1), flash code pattern in long array format,
        take values between 1 and 12, implying row and column indices

        eeg_type: (total_seq_length, 1), binary flash type in long array format,
        implying target or non-target flashes

        eeg_pis: (total_seq_length,), phase in sequence in long array format, take values in 0-3,
        0: irrelevant (break time), 1: prior to the experiment, 2: during the experiment,
        3: after the experiment. It starts and ends with 0, and 1, 2, 3, show up alternately.
        We only focus on 2 and 3.

        """
        eeg_dir = '{}/{}_{}.mat'.format(
            self.parent_eeg_output_path, self.sub_folder_name, eeg_file_name
        )
        print(eeg_dir)
        eeg_dat = sio.loadmat(eeg_dir)

        eeg_keys, _ = zip(*eeg_dat.items())
        print(eeg_keys)
        if eeg_file_name == 'raw':
            eeg_signals = eeg_dat['signal']
        else:
            eeg_signals = eeg_dat['signal_bp']

        eeg_code = eeg_dat['Stim_Code']
        eeg_type = eeg_dat['Stim_Type']
        eeg_pis = eeg_dat['Stim_PIS'][:, 0]

        if show_dim_bool:
            print('raw eeg time sequence has shape {}'.format(eeg_signals.shape))
            print('raw eeg code has shape {}'.format(eeg_code.shape))
            print('raw eeg type has shape {}'.format(eeg_type.shape))
            print('raw eeg phase-in-sequence has shape {}'.format(eeg_pis.shape))

        if 'FRT' in self.sub_folder_name:
            frt_letters = list(eeg_dat['frt_letters'][0])
            ls_wts = eeg_dat['ls_weights']
            print('frt_letters = {}'.format(frt_letters))
            print('ls_wts has shape {}'.format(ls_wts.shape))
            self._update_num_letter(len(frt_letters))
            print('num_letter = {}'.format(self.num_letter))
            return [eeg_signals, eeg_code, eeg_type, eeg_pis, frt_letters, ls_wts]

        elif 'SLO' in self.sub_folder_name or 'REG' in self.sub_folder_name:
            slo_letters = list(eeg_dat['slo_letters'][0][0][0])
            print('slo_letters = {}'.format(slo_letters))
            self._update_num_letter(len(slo_letters))
            return [eeg_signals, eeg_code, eeg_type, eeg_pis, slo_letters]

        else:
            return [eeg_signals, eeg_code, eeg_type, eeg_pis]

    def import_frt_file(self, eeg_file_name):

        eeg_frt = sio.loadmat('{}/{}/{}/{}.mat'.format(
            self.parent_path, self.data_type,
            self.sub_name_short, eeg_file_name)
        )
        frt_signals = eeg_frt['signal']
        frt_code = eeg_frt['Stim_Code']
        frt_type = eeg_frt['Stim_Type']
        frt_pis = eeg_frt['Stim_PIS']
        frt_letters = list(eeg_frt['frt_letters'][:,0])
        ls_weights = eeg_frt['ls_weights']

        return [frt_signals, frt_code, frt_type, frt_pis, frt_letters, ls_weights]

    # Determine eeg_code_subset, eeg_type_subset from this function
    def truncate_raw_sequence(
            self, eeg_pis, eeg_signal, eeg_code, eeg_type,
            show_dim_bool=True
    ):
        r"""
        :param eeg_pis:
        :param eeg_signal:
        :param eeg_code:
        :param eeg_type:
        :param show_dim_bool: bool, whether to display the shape of truncated raw sequence
        :return:
        """

        pis_1_num, _ = eeg_signal[eeg_pis == 1, :].shape
        print('eeg_pis = 1 has average length {}'.format(int(pis_1_num / self.num_letter)))

        pis_2_num, _ = eeg_signal[eeg_pis == 2, :].shape
        single_seq_length_2 = int(pis_2_num / self.num_letter)
        print('eeg_pis = 2 has average length {}'.format(single_seq_length_2))

        pis_3_num, _ = eeg_signal[eeg_pis == 3, :].shape
        print('eeg_pis = 3 has average length {}'.format(int(pis_3_num / self.num_letter)))

        eeg_code_subset = eeg_code[eeg_pis == 2, :]
        eeg_type_subset = eeg_type[eeg_pis == 2, :]

        if 'SLO' in self.sub_folder_name:
            # we have sufficiently long pause time to cover P300 so that
            # there is no need to include the pis == 3.
            eeg_signals_subset = eeg_signal[eeg_pis == 2, :]
            row_num, num_electrode = eeg_signals_subset.shape
            print('eeg_signals after pis == 2 has shape {}'.format(eeg_signals_subset.shape))
            single_seq_length = int(row_num / self.num_letter)
            print('single_seq_length for eeg signals only = {}'.format(single_seq_length))
            eeg_signals_subset = np.reshape(eeg_signals_subset, [self.num_letter, single_seq_length, num_electrode])

            # Identify the starting point of eeg_pis = 2
            status_2_start = np.where(eeg_pis == 2)[0][0]
            # Identify the flash_length in raw data file
            flash_len = np.where(eeg_code[status_2_start:, 0] == 0)[0][0]
            # Identify the pause_length in raw data file
            pause_len = np.where(eeg_code[status_2_start+flash_len:, 0] != 0)[0][0]
            # Identify the sum of the above two
            self._update_flash_and_pause_length(flash_len, pause_len)
            # Identify the number repetition
            num_repetition = int(single_seq_length_2 / self.flash_and_pause_length / self.num_rep)
            self._update_num_repetition(num_repetition)
            # Update n_length, here it equals to d without overlapping
            self._update_n_length(self.flash_and_pause_length)

            # row_num_2 and single_seq_length_2 are valid for code and type without any additional noise.
            eeg_code_subset = np.reshape(eeg_code_subset,
                                         [self.num_letter, self.num_repetition,
                                          self.num_rep, self.flash_and_pause_length])
            eeg_code_subset = eeg_code_subset[:, :, :, 0]
            eeg_type_subset = np.reshape(eeg_type_subset,
                                         [self.num_letter, self.num_repetition,
                                          self.num_rep, self.flash_and_pause_length])
            eeg_type_subset = eeg_type_subset[:, :, :, 0]

        else:
            eeg_signals_subset = eeg_signal[np.logical_or(eeg_pis == 2, eeg_pis == 3), :]
            row_num, num_electrode = eeg_signals_subset.shape
            print('eeg_signals after pis == 2 & pis == 3 has shape {}'.format(eeg_signals_subset.shape))
            single_seq_length = int(row_num / self.num_letter)
            print('single_seq_length for eeg signals only = {}'.format(single_seq_length))
            eeg_signals_subset = np.reshape(
                eeg_signals_subset,
                [self.num_letter, single_seq_length, num_electrode]
            )

            # row_num_2 and single_seq_length_2 are valid for code and type without any additional noise.
            eeg_code_subset = np.reshape(eeg_code_subset,
                                         [self.num_letter, self.num_repetition,
                                          self.num_rep, self.flash_and_pause_length])
            eeg_code_subset = eeg_code_subset[:, :, :, 0]
            eeg_type_subset = np.reshape(eeg_type_subset,
                                         [self.num_letter, self.num_repetition,
                                          self.num_rep, self.flash_and_pause_length])
            eeg_type_subset = eeg_type_subset[:, :, :, 0]

            num_repetition_val = int(single_seq_length_2 / self.flash_and_pause_length / self.num_rep)
            print('The actual sequence number = {}'.format(num_repetition_val))
            last_3_index = single_seq_length_2 + 160
            # This value has fixed formula as long as the number of 3 per letter epoch is smaller than 192.
            eeg_signals_subset = eeg_signals_subset[:, :last_3_index, :]

        # Transpose for both cases:
        eeg_signals_subset = np.transpose(eeg_signals_subset, [0, 2, 1])

        if show_dim_bool:
            print('eeg_signals_subset has shape {}'.format(eeg_signals_subset.shape))
            print('eeg_code_subset has shape {}'.format(eeg_code_subset.shape))
            print('eeg_type_subset has shape {}'.format(eeg_type_subset.shape))

        return eeg_signals_subset, eeg_code_subset, eeg_type_subset

    def save_truncated_signal(
            self, eeg_signals_subset, eeg_code_subset, eeg_type_subset, eeg_name_suffix
    ):
        r"""
        :param eeg_signals_subset: (letter_dim, channel_dim, single_letter_seq_length)
        :param eeg_code_subset: (letter_dim, seq_num, num_rep)
        :param eeg_type_subset: (letter_dim, seq_num, num_rep)
        :param eeg_name_suffix: string
        :return: save those results to the designated directory
        """
        sio.savemat('{}/{}_eeg_dat_{}_trun.mat'.
                    format(self.parent_eeg_output_path, self.sub_folder_name,
                           eeg_name_suffix),
                    {
                        'eeg_signals': eeg_signals_subset,
                        'eeg_code': eeg_code_subset,
                        'eeg_type': eeg_type_subset
                    })

    def save_eeg_spatial_corr(self, eeg_spatial_corr, eeg_name_suffix):
        sio.savemat('{}/{}_eeg_dat_spatial_corr_{}.mat'.format(
            self.parent_eeg_output_path, self.sub_folder_name, eeg_name_suffix
        ),
                    {
                        'corr': eeg_spatial_corr
                    })
        return 'correlation matrix has been saved.'

    def moving_average_decimate(
            self, eeg_signals_subset, dec_factor, show_dim_bool=True
    ):
        r"""
        :param eeg_signals_subset: (letter_dim, channel_dim, single_letter_seq_length)
        :param dec_factor: decimation number, usually positive integer
        :param show_dim_bool: bool, whether display the shape of output array
        :return: post-processing signals

        First, we apply moving average window filter and then down-sample by the equivalent values

        The decimation factor is the same as the window length
        """
        _, _, seq_length = eeg_signals_subset.shape
        move_window = np.ones([dec_factor]) / dec_factor
        move_aver_signal = np.zeros([
            self.num_letter, self.num_electrode, seq_length]
        )
        for trn_letter_id in range(self.num_letter):
            for ele_id in range(self.num_electrode):
                move_aver_signal[trn_letter_id, ele_id, :] = \
                    np.convolve(eeg_signals_subset[trn_letter_id, ele_id, :], move_window, mode="same")
        col_index = np.linspace(
            start=0, stop=seq_length, num=int(seq_length / dec_factor),
            endpoint=False, dtype=np.int64
        )
        move_aver_signal_dec = move_aver_signal[:, :, col_index]
        move_aver_signal_dec = move_aver_signal_dec[..., np.newaxis]

        if show_dim_bool:
            print('moving average signal has shape {}'.format(move_aver_signal_dec.shape))

        return move_aver_signal_dec

    def save_sample_signal_down(
            self, eeg_signals_down_sample, eeg_code_subset, eeg_type_subset,
            decimate_factor, eeg_name_suffix
    ):
        r"""
        :param eeg_signals_down_sample: array_like, (letter_dim, channel_dim, super_seq_length_down, 1)
        :param eeg_code_subset: (letter_dim, num_repetition, 12)
        :param eeg_type_subset: (letter_dim, num_repetition, 12)
        :param decimate_factor: integer
        :param eeg_name_suffix: string
        :return: save those variables in .mat file
        """
        if 'SLO' in self.sub_folder_name or 'REG' in self.sub_folder_name:
            sio.savemat('{}/{}/{}/{}_eeg_dat_down_{}_{}.mat'.
                        format(self.parent_path, self.data_type,
                               self.sub_name_short, self.sub_folder_name,
                               decimate_factor, eeg_name_suffix),
                        {
                            'eeg_signals': eeg_signals_down_sample,
                            'eeg_code': eeg_code_subset,
                            'eeg_type': eeg_type_subset,
                            'num_letter': self.num_letter,
                            'flash_and_pause_length': self.flash_and_pause_length / decimate_factor
                        })
        else:
            sio.savemat('{}/{}/{}/{}_eeg_dat_down_{}_{}.mat'.
                        format(self.parent_path, self.data_type,
                               self.sub_name_short, self.sub_folder_name,
                               decimate_factor, eeg_name_suffix),
                        {
                            'eeg_signals': eeg_signals_down_sample,
                            'eeg_code': eeg_code_subset,
                            'eeg_type': eeg_type_subset
                        })

    def save_eeg_truncated_seq_signal(
            self, eeg_seq_signal, eeg_type, eeg_code, file_subscript
    ):
        r"""
        :param eeg_seq_signal: array_like, (letter_dim, num_electrode, repet_num, seq_length, 1)
        :param eeg_type: 3d-array_like, (letter_dim, repet_num, num_rep)
        :param eeg_code: 3d-array_like, (letter_dim, repet_num, num_rep)
        :param file_subscript: string_like, 'train' or 'test'
        :return:
        """
        sio.savemat('{}/{}/{}/{}_eeg_dat_super_seq_trun_seq_{}.mat'.
                    format(self.parent_path, self.data_type,
                           self.sub_name_short, self.sub_folder_name,
                           file_subscript),
                    {
                        'eeg_signals': eeg_seq_signal,
                        'eeg_code': eeg_code,
                        'eeg_type': eeg_type
                    })

    def import_eeg_spatial_corr(self, eeg_name_suffix):

        eeg_file_dir = '{}/{}_eeg_dat_spatial_corr_{}.mat'.format(
            self.parent_eeg_output_path, self.sub_folder_name, eeg_name_suffix
        )
        print(eeg_file_dir)
        eeg_dat = sio.loadmat(eeg_file_dir)
        eeg_keys, _ = zip(*eeg_dat.items())
        # print(eeg_keys)

        eeg_spatial_corr = eeg_dat['corr']
        return eeg_spatial_corr

    def import_sim_bayes_gen_dataset(
            self, letter_dim, repet_num, sim_type, reshape_option=1
    ):
        r"""
        :param letter_dim:
        :param repet_num:
        :param sim_type:
        :param reshape_option:
        :return: signals: array_like, (channel_dim, letter_dim, repet_num * 12, n_length, 1)
        """

        file_dir = '{}/SIM_files/{}/{}/sim_dat_{}.mat'.format(
            self.parent_path, self.sub_name_short, self.sub_folder_name, sim_type
        )
        # print(file_dir)

        sim_dat = sio.loadmat(file_dir)
        sim_keys, _ = zip(*sim_dat.items())
        # print(sim_keys)

        signals = sim_dat['signals']
        tar = sim_dat['tar']
        ntar = sim_dat['ntar']
        eeg_code = sim_dat['eeg_code']  # originally 2d-array
        eeg_type = sim_dat['eeg_type']  # originally 2d-array
        rho_s = sim_dat['rho_s'][0, :]
        s_x_sq = sim_dat['s_x_sq'][0, :]
        rho = sim_dat['rho']
        message = sim_dat['message'][0]

        if reshape_option == 3:
            eeg_code = np.reshape(eeg_code, [letter_dim, repet_num, self.num_rep])
            eeg_type = np.reshape(eeg_type, [letter_dim, repet_num, self.num_rep])
        if reshape_option == 1:
            eeg_code = np.reshape(eeg_code, [letter_dim * repet_num * self.num_rep])
            eeg_type = np.reshape(eeg_type, [letter_dim * repet_num * self.num_rep])

        # print('eeg_signals has shape {}'.format(signals.shape))
        # print('eeg_type has shape {}'.format(eeg_type.shape))
        # print('eeg_code has shape {}'.format(eeg_code.shape))

        return [tar, ntar, s_x_sq, rho, rho_s,
                signals, eeg_code, eeg_type, message]

    def import_sim_ml_trunc_dataset(
            self, sim_type,
            letter_dim, repet_num,
            reshape_option
    ):
        r"""
        :param sim_type:
        :param letter_dim:
        :param repet_num:
        :param reshape_option:
        :return: signals: array_like, (# of stimuli, channel_dim * n_length)
                 eeg_code: 1d-array_like, (# of stimuli)
                 eeg_type: 1d-array_like, (# of stimuli)
        """

        file_dir = '{}/SIM_files/Chapter_1/{}/{}/sim_dat_{}.mat'.format(
            self.parent_path, self.sub_name_short,
            self.sub_folder_name, sim_type
        )
        print(file_dir)

        sim_dat = sio.loadmat(file_dir)
        sim_keys, _ = zip(*sim_dat.items())
        # print(sim_keys)

        # Discriminant ML methods
        signals = sim_dat['eeg_signals']
        eeg_type = sim_dat['eeg_type']  # originally 1d-array
        eeg_code = sim_dat['eeg_code']  # originally 1d-array

        if reshape_option == 2:
            eeg_type = np.reshape(eeg_type, [letter_dim, repet_num * self.num_rep])
            eeg_code = np.reshape(eeg_code, [letter_dim, repet_num * self.num_rep])

        if reshape_option == 3:
            eeg_type = np.reshape(eeg_type, [letter_dim, repet_num, self.num_rep])
            eeg_code = np.reshape(eeg_code, [letter_dim, repet_num, self.num_rep])

        return [signals, eeg_type, eeg_code]

    def save_sim_ml_trunc_dataset(
            self, signals_trun, type_indicator, code_indicator, file_subscript
    ):
        print('signals_trun prior to pre-processing has shape {}'.format(signals_trun.shape))
        channel_dim, train_length, n_length = signals_trun.shape
        signals_trun = np.transpose(signals_trun, [1, 0, 2])
        signals_trun = np.reshape(
            signals_trun, [train_length, channel_dim * n_length])
        print('signals prior to swlda has shape {}'.format(signals_trun.shape))
        type_1d = np.reshape(
            type_indicator, [self.num_letter * self.num_repetition * self.num_rep]
        )
        code_1d = np.reshape(
            code_indicator, [self.num_letter * self.num_repetition * self.num_rep]
        )
        # Convert 0-1 coding to -1-to-1 coding scheme
        type_1d = np.copy((type_1d - 0.5) * 2)
        print('type_1d has shape {}'.format(type_1d.shape))
        print('code_1d has shape {}'.format(code_1d.shape))
        print('type_1d top 12 looks like {}'.format(type_1d[:12]))

        file_dir = '{}/{}/{}/{}/sim_dat_ML_{}.mat'.format(
            self.parent_path, self.data_type, self.sub_name_short,
            self.sub_folder_name, file_subscript
        )
        sio.savemat(file_dir,
                    {
                        'eeg_signals': signals_trun,
                        'eeg_type': type_1d,
                        'eeg_code': code_1d
                    })

        return 'save dataset done!\n'

    # Export it to MATLAB to perform SWLDA or any other classification method
    def save_truncate_signal_1d_real(
            self, signals_trun, type_indicator, code_indicator, file_subscript, repet_num, array_3d_bool=True
    ):
        r"""
        :param signals_trun: 3d-array, (letter_dim * num_repetition * num_rep, num_electrode, n_length)
        :param type_indicator: 1d-array, (letter_dim * num_repetition * num_rep)
        :param code_indicator: 1d-array, (letter_dim * num_repetition * num_rep)
        :param file_subscript: string,
        :param repet_num: integer
        :param array_3d_bool: bool whether we want to fit single channel. True to leave signals_trun a 3d-array.
        :return:
        """
        train_length, channel_dim, n_length = signals_trun.shape
        if not array_3d_bool:
            signals_trun = np.reshape(
                signals_trun, [train_length, channel_dim * n_length])
        type_1d = np.reshape(
            type_indicator, [1, self.num_letter * repet_num * self.num_rep]
        )
        code_1d = np.reshape(
            code_indicator, [1, self.num_letter * repet_num * self.num_rep]
        )
        # Convert 0-1 coding to -1-to-1 coding scheme
        type_1d = np.copy((type_1d - 0.5) * 2)
        print('signals_trun has shape {}'.format(signals_trun.shape))
        print('type_1d has shape {}'.format(type_1d.shape))
        print('code_1d has shape {}'.format(code_1d.shape))

        sio.savemat('{}/{}/{}/{}_eeg_dat_ML_{}.mat'
                    .format(self.parent_path,
                            self.data_type,
                            self.sub_name_short,
                            self.sub_folder_name,
                            file_subscript),
                    {
                        'eeg_signals': signals_trun,
                        'eeg_type': type_1d,
                        'eeg_code': code_1d
                    })

    # https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
    # https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    # https://towardsdatascience.com/significance-of-acf-and-pacf-plots-in-time-series-analysis-2fa11a5d10a8

    def import_eeg_processed_dat(
            self, file_subscript, reshape_1d_bool=True, letter_dim=None, num_repetition=None
    ):

        r"""
        file_subscript: string
        reshape_to_1d: bool
        letter_dim: integer
        num_repetition: integer

        return:
            A list of three arrays, including
                eeg_signals, with shape (num_letter, num_electrode, seq_length)
                eeg_code, with shape (num_letter, num_repetition, num_rep)
                eeg_type, with shape (num_letter, num_repetition, num_rep)
        """

        file_path = '{}/{}_eeg_dat_{}.mat'.format(
            self.parent_eeg_output_path,
            self.sub_folder_name,
            file_subscript
        )
        # print(file_path)
        eeg_dat = sio.loadmat(file_path)

        eeg_keys, _ = zip(*eeg_dat.items())
        # print(eeg_keys)
        eeg_signals = eeg_dat['eeg_signals']
        eeg_code = eeg_dat['eeg_code']
        eeg_type = eeg_dat['eeg_type']

        if letter_dim is None:
            letter_dim = self.num_letter

        if num_repetition is None:
            num_repetition = self.num_repetition

        if reshape_1d_bool:
            eeg_code = np.reshape(eeg_code, [1, letter_dim*num_repetition*self.num_rep])
            eeg_type = np.reshape(eeg_type, [1, letter_dim*num_repetition*self.num_rep])

        else:
            eeg_code = np.reshape(eeg_code, [letter_dim, num_repetition, self.num_rep])
            eeg_type = np.reshape(eeg_type, [letter_dim, num_repetition, self.num_rep])

        print('eeg_signal has shape {}'.format(eeg_signals.shape))
        print('eeg_code has shape {}'.format(eeg_code.shape))
        print('eeg_type has shape {}'.format(eeg_type.shape))

        if 'SLO' in self.sub_folder_name or 'REG' in self.sub_folder_name:
            num_letter = int(eeg_dat['num_letter'][0][0])
            flash_and_pause_length = int(eeg_dat['flash_and_pause_length'][0][0])
            return [eeg_signals, eeg_code, eeg_type,
                    num_letter, flash_and_pause_length]

        else:
            return [eeg_signals, eeg_code, eeg_type]

    def import_eeg_odd_even_dat(self, file_subscript):
        file_path = '{}/{}_eeg_dat_{}.mat'.format(
            self.parent_eeg_output_path, self.sub_folder_name, file_subscript
        )
        print(file_path)

        eeg_dat = sio.loadmat(file_path)
        eeg_keys, _ = zip(*eeg_dat.items())
        print(eeg_keys)

        eeg_signals = eeg_dat['eeg_signals']
        eeg_code = eeg_dat['eeg_code']
        eeg_type = eeg_dat['eeg_type']

        return eeg_signals, eeg_code, eeg_type

    def import_eeg_single_channel_screen_dat(
            self, method_name, scenario_name, file_subscript
    ):
        screen_mat_dir = '{}/{}/{}/{}_eeg_dat_{}_screen.mat'.format(
            self.parent_eeg_output_path, method_name, scenario_name,
            self.sub_folder_name, file_subscript
        )
        # print(screen_mat_dir)

        eeg_dat = sio.loadmat(screen_mat_dir)
        eeg_keys, _ = zip(*eeg_dat.items())
        # print(eeg_keys)

        eeg_signals = eeg_dat['eeg_signals']
        eeg_code = eeg_dat['eeg_code']
        eeg_type = eeg_dat['eeg_type']
        eeg_log_lkd = eeg_dat['log_lkd_single_seq']
        low_q_val = eeg_dat['low_quantile_value']

        letter_dim_screen = eeg_code.shape[0]
        eeg_code = np.reshape(eeg_code, [letter_dim_screen * self.num_rep])
        eeg_type = np.reshape(eeg_type, [letter_dim_screen * self.num_rep])

        return eeg_signals, eeg_code, eeg_type, eeg_log_lkd, low_q_val

    def create_truncate_segment(self, eeg_signals_subset, repetition_dim):
        eeg_signals_subset = np.transpose(eeg_signals_subset)  # with shape (16, seq_length)
        eeg_signals_subset_trun = []
        total_z_num = repetition_dim * self.num_rep
        for i in range(total_z_num):
            low_bound = i * int(self.flash_and_pause_length)
            upp_bound = low_bound + self.n_length
            temp_i = eeg_signals_subset[:, low_bound:upp_bound]
            eeg_signals_subset_trun.append(temp_i)
        eeg_signals_subset_trun = np.stack(eeg_signals_subset_trun, axis=1)
        return eeg_signals_subset_trun

    def create_truncate_segment_batch(
            self, eeg_signals, eeg_type, letter_dim, trn_repetition,
            show_dim_bool
    ):
        r"""
        args:
        -----
            eeg_signals: array_like, (channel_dim, letter_dim, super_seq_length, 1)

            eeg_type: None or 3d-array, (letter_dim, rep_dim, num_rep)

            letter_dim: integer

            trn_repetition: integer

            show_dim_bool: bool
        return:
        -----
            A tuple containing two elements:

            1. truncated eeg signal segments,
            (channel_dim, trn_repetition * num_rep * letter_dim, n_length)

            2. the 1d-array of eeg_type of the training set.

        note:
        -----
            This function creates truncated signals by stimulus (with assumed latency period).
        """

        if 'SLO' in self.sub_folder_name:
            print('eeg_signals has shape {}'.format(eeg_signals.shape))
            eeg_signals_trun = np.reshape(
                eeg_signals, [self.num_letter,
                              self.num_electrode,
                              self.num_repetition * self.num_rep,
                              self.n_length]
            )
            eeg_signals_trun = np.transpose(
                eeg_signals_trun, [1, 0, 2, 3]
            )
            eeg_signals_trun = np.reshape(
                eeg_signals_trun, [self.num_electrode,
                                   self.num_letter * self.num_repetition * self.num_rep,
                                   self.n_length]
            )
            print('eeg_signals_trun has shape {}'.format(eeg_signals_trun.shape))

            return eeg_signals_trun

        else:
            total_rep = trn_repetition * self.num_rep
            eeg_signals_trun = []
            for i in range(total_rep):
                low_bound = i * int(self.flash_and_pause_length)
                upp_bound = low_bound + self.n_length
                eeg_signals_subset = eeg_signals[..., low_bound:upp_bound, 0]
                eeg_signals_trun.append(eeg_signals_subset)
            eeg_signals_trun = np.stack(eeg_signals_trun, axis=2)
            print('eeg_signals_trun after stack at 2 has shape {}'.format(eeg_signals_trun.shape))
            eeg_signals_trun = np.reshape(eeg_signals_trun, [self.num_electrode,
                                                             self.num_letter * total_rep,
                                                             self.n_length])
            eeg_type_sub = 0
            if eeg_type is not None:
                eeg_type_sub = np.reshape(
                    eeg_type[:letter_dim, :trn_repetition, :],
                    [letter_dim * trn_repetition * self.num_rep]
                )
            if show_dim_bool:
                print('flash-based truncated signal has shape {}\n'.format(eeg_signals_trun.shape))
                if eeg_type is not None:
                    print('eeg type subset has shape {}\n'.format(eeg_type_sub.shape))
                var_mag = np.var(eeg_signals_trun, axis=(1, 2))
                print('signal variance has sigma_sq per channel \n {}\n'.format(
                    np.round(var_mag, decimals=2))
                )

            return eeg_signals_trun, eeg_type_sub

    def create_truncate_seq_segment_batch(
            self, eeg_signals, letter_dim, repetition_dim
    ):
        r"""
        :param eeg_signals: super-sequence array,
            have shape (channel_dim, letter_dim, super_seq_length, 1)
        :param letter_dim:
        :param repetition_dim: the total number of repetition per letter
        :return: array, (channel_dim, letter_dim, repetiton_dim, seq_length, 1)

        This function creates truncated signals by sequence (known length, fixed by design)
        """
        # seq_length = (self.num_rep + self.n_multiple-1) * self.flash_and_pause_length
        assert eeg_signals.shape[1] == letter_dim, print('incorrect letter dim input!')
        eeg_signals_trun = np.zeros([
            self.num_electrode, letter_dim, repetition_dim, self.seq_length, 1]
        )
        for k in range(repetition_dim):
            k_low = k * self.num_rep * self.flash_and_pause_length
            k_upp = k_low + self.seq_length
            eeg_signals_trun[..., k, :, :] = eeg_signals[..., k_low:k_upp, :]

        return eeg_signals_trun

    def produce_trun_mean_cov_subset(self, signals_trun, type_sub):

        r"""
        args:
        -----
            signals_trun: 3d-array, (channel_dim, stimulus_num, n_length)
            type_sub: 1d-array, (letter_dim * sequence_num * num_rep,)

        return:
        -----
            A list of 4 arrays including
                target_mean, (num_electrode, n_length)
                non_target_mean, (num_electrode, n_length)
                target_cov, (num_electrode, n_length, n_length)
                non_target_cov, (num_electrode, n_length, n_length)

        note:
        -----
            descriptive mean and sample covariance statistics from real data
        """

        signals_trun_sub_t = signals_trun[:, type_sub == 1, :]
        signals_trun_sub_nt = signals_trun[:, type_sub == 0, :]

        # Examine sample mean function (under sub-setting)
        signals_trun_t_mean = np.mean(signals_trun_sub_t, axis=1)
        signals_trun_nt_mean = np.mean(signals_trun_sub_nt, axis=1)

        # Examine sample covariance matrix
        signals_trun_t_cov = np.stack([np.cov(signals_trun_sub_t[i, ...], rowvar=False)
                                       for i in range(self.num_electrode)], axis=0)
        signals_trun_nt_cov = np.stack([np.cov(signals_trun_sub_nt[i, ...], rowvar=False)
                                       for i in range(self.num_electrode)], axis=0)
        return [signals_trun_t_mean, signals_trun_nt_mean,
                signals_trun_t_cov, signals_trun_nt_cov]

    def produce_mean_covariance_plots(
            self, mu_1, mu_0, cov_1, cov_0,
            file_subscript, sim_dat=True
    ):
        # plot_parental_dir = '{}/{}/{}/{}'.format(
        #     self.parent_path, self.data_type, self.sub_name_short, self.sub_folder_name
        # )
        if sim_dat:
            plot_parental_dir = self.parent_sim_output_path
        else:
            plot_parental_dir = self.parent_eeg_output_path
        print(plot_parental_dir)
        trunc_dir = plot_parental_dir + '/trunc_stats/'
        try:
            os.mkdir(trunc_dir)
            print('Directory', trunc_dir, ' is created.')
        except FileExistsError:
            print('Directory ', trunc_dir, ' already exists.')

        if mu_1 is not None and mu_0 is not None:
            mean_fn_pdf = bpdf.PdfPages('{}/{}_sample_mean_{}.pdf'.format(
                trunc_dir, self.sub_folder_name, file_subscript)
            )
            for ele_id in range(self.num_electrode):
                fig = plt.figure(figsize=(12, 10))
                plt.plot(self.time_range, mu_1[ele_id, :], label="target")
                plt.plot(self.time_range, mu_0[ele_id, :], label="non-target")
                plt.hlines(y=0, xmin=self.time_range[0], xmax=self.time_range[-1])
                # plt.ylim(-8.5, 4.5)
                plt.xlabel('Time (ms)')
                plt.ylabel('Mean Estimation')
                plt.title('{}, channel {}'.format(self.sub_folder_name, ele_id + 1))
                plt.legend(loc='upper right')
                # plt.show()
                plt.close()
                mean_fn_pdf.savefig(fig)
            mean_fn_pdf.close()

            mean_fn_dir = '{}/{}_sample_mean_{}.mat'.format(
                trunc_dir, self.sub_folder_name, file_subscript
            )
            sio.savemat(mean_fn_dir,
                        {
                            'target': mu_1,
                            'non_target': mu_0
                        })
        else:
            print('We fail to produce mean plots.')

        if cov_1 is not None and cov_0 is not None:
            sample_cov_t_pdf = bpdf.PdfPages('{}/{}_sample_cov_target_{}.pdf'.format(
                trunc_dir, self.sub_folder_name, file_subscript)
            )
            for i in range(self.num_electrode):
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                X, Y = np.meshgrid(self.time_range, self.time_range)

                fig1 = plt.figure(figsize=(12, 10))
                ax1 = fig1.add_axes([left, bottom, width, height])
                Z1 = cov_1[i, :, :]
                cp1 = plt.contourf(X, Y, Z1)
                fig1.colorbar(cp1)
                ax1.set_title('Target Contour, Channel ' + str(i + 1))
                ax1.set_xlabel('Time (ms)')
                ax1.set_ylabel('Time (ms)')
                ax1.set_ylim(ax1.get_ylim()[::-1])
                sample_cov_t_pdf.savefig(fig1)
                plt.close()
            sample_cov_t_pdf.close()

            sample_cov_nt_pdf = bpdf.PdfPages('{}/{}_sample_cov_non_target_{}.pdf'.format(
                    trunc_dir, self.sub_folder_name, file_subscript)
            )
            for i in range(self.num_electrode):
                # Common configuration
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                X, Y = np.meshgrid(self.time_range, self.time_range)

                fig0 = plt.figure(figsize=(12, 10))
                ax0 = fig0.add_axes([left, bottom, width, height])
                Z0 = cov_0[i, :, :]
                cp0 = plt.contourf(X, Y, Z0)
                fig0.colorbar(cp0)

                ax0.set_title('Non-target Contour, Channel ' + str(i + 1))
                ax0.set_xlabel('Time (ms)')
                ax0.set_ylabel('Time (ms)')
                ax0.set_ylim(ax0.get_ylim()[::-1])
                sample_cov_nt_pdf.savefig(fig0)
                plt.close()
            sample_cov_nt_pdf.close()

        else:
            print('We fail to produce covariance plots.')

    # def import_sample_mean_fn(self, file_subscript):
    #     mean_fn_dir = '{}/{}/{}/{}_sample_mean_{}.mat'.format(
    #         self.parent_path, self.data_type, self.sub_name_short,
    #         self.sub_folder_name, file_subscript
    #     )
    #     # print(mean_fn_dir)
    #     mean_fn = sio.loadmat(mean_fn_dir)
    #     mean_fn_keys, _ = zip(*mean_fn.items())
    #     # print(mean_fn_keys)
    #
    #     mean_tar = mean_fn['target']
    #     mean_ntar = mean_fn['non_target']
    #
    #     return mean_tar, mean_ntar

    def save_mcmc(
            self, s_x_sq_mcmc, rho_mcmc,
            zeta_mcmc, zeta_true,
            beta_tar_mcmc, beta_ntar_mcmc,
            scale_mcmc, var_mcmc, log_lkd_mcmc,
            repet_num, method_name, sim_type,
            scenario_name, job_id=0, sim_dat=True, **kwargs
    ):
        if sim_dat:
            file_dir = '{}/{}/{}/{}_{}_{}_mcmc_trn_{}_{}.mat'.format(
                self.parent_sim_output_path, method_name, scenario_name,
                self.sub_folder_name, sim_type, method_name, repet_num, job_id
            )
            # print('file_dir = {}'.format(file_dir))
            # model-based simulation, mean_fn is the convolution of beta_tar and beta_ntar by eeg_type
            sio.savemat(file_dir,
                        {
                            's_x_sq': s_x_sq_mcmc,
                            'rho': rho_mcmc,
                            'scale': scale_mcmc,
                            'scale_var': var_mcmc,
                            'zeta': zeta_mcmc,
                            'zeta_true': zeta_true,
                            'beta_tar': beta_tar_mcmc,
                            'beta_ntar': beta_ntar_mcmc,
                            'log_lkd': log_lkd_mcmc
                        })
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
                file_dir = '{}/{}_{}_{}_mcmc_trn_{}_{}.mat'.format(
                    file_dir0, self.sub_folder_name, sim_type, method_name, repet_num, job_id
                )
            else:
                file_dir = '{}/{}/{}/{}_{}_{}_mcmc_trn_{}_{}.mat'.format(
                    self.parent_eeg_output_path, method_name, scenario_name,
                    self.sub_folder_name, sim_type, method_name, repet_num, job_id
                )
            # print('file_dir = {}'.format(file_dir))
            # model-based simulation, mean_fn is the convolution of beta_tar and beta_ntar by eeg_type
            sio.savemat(file_dir,
                        {
                            's_x_sq': s_x_sq_mcmc,
                            'rho': rho_mcmc,
                            'scale': scale_mcmc,
                            'scale_var': var_mcmc,
                            'zeta': zeta_mcmc,
                            # 'zeta_true': zeta_true,
                            'beta_tar': beta_tar_mcmc,
                            'beta_ntar': beta_ntar_mcmc,
                            'log_lkd': log_lkd_mcmc
                        })

    def import_mcmc(
            self, sim_type, method_name, repet_num,
            scenario_name, job_id, sim_dat=True, **kwargs
    ):
        if sim_dat:
            file_dir = '{}/{}/{}/{}_{}_{}_mcmc_trn_{}_{}.mat'.format(
                self.parent_sim_output_path, method_name, scenario_name,
                self.sub_folder_name, sim_type, method_name, repet_num, job_id
            )

            bayes_mcmc = sio.loadmat(file_dir)
            bayes_mcmc_keys, _ = zip(*bayes_mcmc.items())
            s_x_sq = bayes_mcmc['s_x_sq']
            rho = bayes_mcmc['rho']
            zeta = bayes_mcmc['zeta']
            zeta_true = bayes_mcmc['zeta_true']
            beta_tar = bayes_mcmc['beta_tar']
            beta_ntar = bayes_mcmc['beta_ntar']
            scale = bayes_mcmc['scale']
            var = bayes_mcmc['scale_var']
            log_lkd = bayes_mcmc['log_lkd']

            return [s_x_sq, rho, zeta, zeta_true,
                    beta_tar, beta_ntar, scale, var, log_lkd]

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
                file_dir = '{}/{}_{}_{}_mcmc_trn_{}_{}.mat'.format(
                    file_dir0, self.sub_folder_name, sim_type, method_name, repet_num, job_id
                )
            else:
                file_dir = '{}/{}/{}/{}_{}_{}_mcmc_trn_{}_{}.mat'.format(
                    self.parent_eeg_output_path, method_name, scenario_name,
                    self.sub_folder_name, sim_type, method_name, repet_num, job_id
                )
            bayes_mcmc = sio.loadmat(file_dir)
            bayes_mcmc_keys, _ = zip(*bayes_mcmc.items())
            s_x_sq = bayes_mcmc['s_x_sq']
            rho = bayes_mcmc['rho']
            zeta = bayes_mcmc['zeta']
            beta_tar = bayes_mcmc['beta_tar']
            beta_ntar = bayes_mcmc['beta_ntar']
            scale = bayes_mcmc['scale']
            var = bayes_mcmc['scale_var']
            log_lkd = bayes_mcmc['log_lkd']

            return [s_x_sq, rho, zeta, beta_tar, beta_ntar,
                    scale, var, log_lkd]

    def save_mcmc_trace_plot(
            self, rho_mcmc, s_x_sq_mcmc,
            scale_mcmc, var_mcmc,
            log_lkd_mcmc, zeta_mean,
            repet_num, sim_type, method_name,
            q, channel_ids, scenario_name, job_id=None
    ):
        r"""
        :param rho_mcmc: array_like, (mcmc_num, channel_dim, q)
        :param s_x_sq_mcmc: array_like, (mcmc_num, channel_dim)
        :param scale_mcmc: array_like, (mcmc_num, channel_dim, 2)
        :param var_mcmc: array_like, (mcmc_num, channel_dim, 2)
        :param log_lkd_mcmc: array_like, (mcmc_num, channel_dim)
        :param zeta_mean: array_like, (num_electrode, n_length)
        :param repet_num: integer
        :param sim_type: string
        :param method_name: string
        :param scenario_name: string
        :param q: integer
        :param channel_ids: array_like
        :param job_id: integer
        :return:
            A systematic plot including traceplot of rho, sigma_sq, and kernel variance lambda,
            log-likelihood change, and mean selection rate across channels.
            For rho, we only display the first component each channel.

            For real data, we don't need sim_type
        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        plot_pdf = bpdf.PdfPages('{}/{}/{}/{}/{}/{}/{}_{}_{}_trace_plot_trn_{}_{}.pdf'.format(
            self.parent_path, self.data_type, self.sub_name_short,
            self.sub_folder_name, method_name, scenario_name,
            self.sub_folder_name, sim_type, method_name, repet_num, job_id)
        )
        s_x_sq_min = np.min(s_x_sq_mcmc, axis=0) - 0.1
        s_x_sq_max = np.max(s_x_sq_mcmc, axis=0) + 0.1

        for i in range(channel_dim):
            fig_1 = plt.figure(figsize=(15, 12))
            ax1 = fig_1.add_subplot(2, 3, 1)
            for p in range(q):
                ax1.plot(rho_mcmc[:, i, p], label='phi'+str(p + 1))
            ax1.legend(loc="upper right")
            ax1.set_ylim(-0.1, 1.1)
            ax1.title.set_text('rho_chan_' + str(i + 1))

            ax2 = fig_1.add_subplot(2, 3, 2)
            ax2.plot(s_x_sq_mcmc[:, i])
            ax2.set_ylim(s_x_sq_min[i], s_x_sq_max[i])
            ax2.title.set_text('s_x_sq_chan_' + str(i + 1))

            ax3 = fig_1.add_subplot(2, 3, 3)
            ax3.plot()
            # ax3.plot(var_mcmc[:, i, 0], label='Target')
            # ax3.plot(var_mcmc[:, i, 1], label='Non-target')
            # # ax3.plot(np.tile(var_mcmc[np.newaxis, i, 0], [10, 1]), label='Target')
            # # ax3.plot(np.tile(var_mcmc[np.newaxis, i, 1], [10, 1]), label='Non-target')
            # ax3.set_ylim(var_min[i], var_max[i])
            # ax3.title.set_text('scale_var channel-' + str(i + 1))
            # ax3.legend(loc='best')

            ax4 = fig_1.add_subplot(2, 3, 4)
            ax4.plot()
            # ax4.plot(scale_mcmc[:, i, 0], label='Target')
            # ax4.plot(scale_mcmc[:, i, 1], label='Non-target')
            # # ax4.plot(np.tile(scale_mcmc[np.newaxis, i, 0], [10, 1]), label='Target')
            # # ax4.plot(np.tile(scale_mcmc[np.newaxis, i, 1], [10, 1]), label='Non-target')
            # ax4.set_ylim(scale_min[i], scale_max[i])
            # ax4.title.set_text('scale_var channel-' + str(i + 1))
            # ax4.legend(loc='best')

            ax5 = fig_1.add_subplot(2, 3, 5)
            ax5.plot(log_lkd_mcmc[:, i])
            ax5.title.set_text('log-lkd channel-' + str(i + 1))

            # mean selection rate
            ax6 = fig_1.add_subplot(2, 3, 6)
            ax6.plot(self.time_range, zeta_mean[i, :], label='Soft-threshold')
            ax6.plot(self.time_range, 1 * (zeta_mean[i, :] > 0.5), label='Hard-threshold')
            ax6.set_ylim(-0.1, 1.1)
            ax6.title.set_text('Selection Indicator, Channel {}'.format(i + 1))
            ax6.set_xlabel('Time (ms)')
            ax6.set_ylabel('Proportion')
            plt.close()
            plot_pdf.savefig(fig_1)

        plot_pdf.close()

    def save_bayes_results(
            self, new_bayes_result, trn_repet_num, test_repet_num,
            method_name, sim_type, target_letters, target_letter_rows, target_letter_cols,
            scenario_name, file_subscript, sim_dat=True, **kwargs
    ):
        r"""
        :param new_bayes_result:
        :param trn_repet_num:
        :param test_repet_num:
        :param method_name:
        :param sim_type:
        :param target_letters:
        :param target_letter_rows:
        :param target_letter_cols:
        :param scenario_name:
        :param file_subscript: string
        :param single_bool: bool
        :param sim_dat: bool
        :return:
        For real data, we don't need sim_type!
        """

        method_folder_name = method_name + 'Pred'
        method_dir0 = "{}/{}/{}".format(
            self.parent_path, method_folder_name, self.sub_name_short
        )
        try:
            os.mkdir(method_dir0)
            print('Directory', method_dir0, ' is created.')
        except FileExistsError:
            print('Directory ', method_dir0, ' already exists.')

        if sim_dat:
            method_dir = "{}/{}".format(
                method_dir0, self.sub_folder_name
            )
            try:
                os.mkdir(method_dir)
                print('Directory', method_dir, ' is created.')
            except FileExistsError:
                print('Directory ', method_dir, ' already exists.')

            method_dir = "{}/{}".format(method_dir, scenario_name)
            try:
                os.mkdir(method_dir)
                print('Directory', method_dir, ' is created.')
            except FileExistsError:
                print('Directory ', method_dir, ' already exists.')
        else:
            method_dir = "{}/{}".format(method_dir0, scenario_name)
            try:
                os.mkdir(method_dir)
                print('Directory', method_dir, ' is created.')
            except FileExistsError:
                print('Directory ', method_dir, ' already exists.')

            if 'scale' in kwargs.keys() and 'gamma' in kwargs.keys():
                scale_val = kwargs['scale']
                gamma_val = kwargs['gamma']
                method_dir = '{}/scale={}, gamma={}'.format(method_dir, scale_val, gamma_val)
                try:
                    os.mkdir(method_dir)
                    print('Directory', method_dir, ' is created.')
                except FileExistsError:
                    print('Directory', method_dir, ' already exists.')

        if test_repet_num is None:
            # assert 1 <= trn_repet_num <= self.num_repetition, print('wrong repetition in training set!')
            pred_repet_num = trn_repet_num
            file_dir = "{}/{}_{}_train_{}_pred_train_{}_{}".format(
                method_dir, self.sub_folder_name, sim_type,
                trn_repet_num, trn_repet_num, file_subscript
            )
        else:
            # assert 1 <= test_repet_num <= self.num_repetition, print('wrong repetition in testing set!')
            pred_repet_num = test_repet_num
            file_dir = "{}/{}_{}_train_{}_pred_test_{}_{}".format(
                method_dir, self.sub_folder_name, sim_type,
                trn_repet_num, test_repet_num, file_subscript
            )

        # Save .mat file by all means
        np.savez('{}.npz'.format(file_dir), **new_bayes_result)
        # Use with np.load(file_name.npz) as data:
        # dict to open the file
        # letter_dim = len(target_letters)

        task = 'w'  # Let task be w for now
        prob_cum = new_bayes_result['prob_cum']
        prob_cum_median = np.median(prob_cum, axis=-2)
        prob_cum_max = np.max(prob_cum_median, axis=-1)

        prob_cum_row = new_bayes_result['prob_cum_row']
        prob_cum_row_median = np.median(prob_cum_row, axis=-2)
        prob_cum_row_max = np.max(prob_cum_row_median, axis=-1)

        prob_cum_col = new_bayes_result['prob_cum_col']
        prob_cum_col_median = np.median(prob_cum_col, axis=-2)
        prob_cum_col_max = np.max(prob_cum_col_median, axis=-1)

        letter_rank = new_bayes_result['letter_cum_rank']
        letter_rank, _ = stats.mode(letter_rank[..., 0], axis=2)
        letter_rank = np.squeeze(letter_rank, axis=-1)
        row_rank = new_bayes_result['row_cum_rank']
        row_rank, _ = stats.mode(row_rank[..., 0], axis=2)
        row_rank = np.squeeze(row_rank, axis=-1)
        col_rank = new_bayes_result['col_cum_rank']
        col_rank, _ = stats.mode(col_rank[..., 0], axis=2)
        col_rank = np.squeeze(col_rank, axis=-1)

        with open('{}.csv'.format(file_dir), task) as f:
            f_writer = csv.writer(f)
            if task == "a":
                f_writer.writerow([' '])
            l0 = ['trn_repetition', trn_repet_num, 'mcmc_sample', new_bayes_result['sample_num'],
                  'scale_hyper_param', new_bayes_result['scale'],
                  'var_hyper_param', new_bayes_result['var']]
            f_writer.writerow(l0)
            letter_row = [' '] * 8
            f_writer.writerow(letter_row)

            f_writer.writerow(['Cumulative sequence prediction:'])
            letter_pred = np.zeros([pred_repet_num])
            row_pred = np.zeros([pred_repet_num])
            col_pred = np.zeros([pred_repet_num])
            letter_row_col_pred = np.zeros([pred_repet_num])

            for i, l_id, l_row_id, l_col_id in zip(
                    np.arange(len(target_letters)), target_letters, target_letter_rows, target_letter_cols
            ):
                l_i = ['Letter ' + l_id.upper(),
                       'Max letter', 'Max prob', 'Target prob',
                       'Max row', 'Max row prob', 'Target row prob',
                       'Max col', 'Max col prob', 'Target col prob'
                       ]
                f_writer.writerow(l_i)
                for j in range(pred_repet_num):
                    l_ij = np.array(self.letter_table)[letter_rank[i, j]]
                    r_ij = np.array(self.row_set)[row_rank[i, j]]
                    c_ij = np.array(self.column_set)[col_rank[i, j]]
                    l_seq_j = ['Sequence ' + str(j+1),
                               l_ij, prob_cum_max[i, j], prob_cum_median[i, j, self.letter_table.index(l_id.upper())],
                               r_ij, prob_cum_row_max[i, j], prob_cum_row_median[i, j, self.row_set.index(l_row_id)],
                               c_ij, prob_cum_col_max[i, j], prob_cum_col_median[i, j, self.column_set.index(l_col_id)]
                               ]
                    # l_seq_j.extend(list(max_letter_cum[i, j]))
                    f_writer.writerow(l_seq_j)
                    if l_ij == l_id:
                        letter_pred[j] = letter_pred[j] + 1

                    if r_ij == l_row_id:
                        row_pred[j] = row_pred[j] + 1

                    if c_ij == l_col_id:
                        col_pred[j] = col_pred[j] + 1

                    if r_ij == l_row_id and c_ij == l_col_id:
                        letter_row_col_pred[j] = letter_row_col_pred[j] + 1

                f_writer.writerow([' '])

            f_writer.writerow(['Letters'])
            f_writer.writerow(list(letter_pred))
            f_writer.writerow(['Rows'])
            f_writer.writerow(list(row_pred))
            f_writer.writerow(['Columns'])
            f_writer.writerow(list(col_pred))
            f_writer.writerow(['Letters based on Rows and Columns'])
            f_writer.writerow(list(letter_row_col_pred))

        return 'Bayes results have been saved!'

    def generate_seq_from_latency_signals(
            self, signals, design_x_mat, eeg_code, eeg_type,
            eta, s_x_sq, rho, s_z_sq, beta_tar, beta_ntar,
            normal_bool, t_df,
            letter_dim, repet_num, q, message,
            sim_folder_name, sim_type_full,
            convol_bool, single_seq_bool, save_plot_bool
    ):
        noise_size = letter_dim * repet_num
        noise_arq_mat = []
        for e in range(self.num_electrode):
            cov_mat_e, noise_arq_e = self.create_arq_noise_close(
                s_x_sq[e], rho[e, :], self.seq_length, noise_size, normal_bool, t_df
            )
            noise_arq_mat.append(noise_arq_e)
        noise_arq_mat = np.reshape(
            np.stack(noise_arq_mat, axis=0),
            [self.num_electrode, letter_dim, repet_num, self.seq_length, 1]
        )
        print('background noise has shape {}'.format(noise_arq_mat.shape))

        signals = np.reshape(
            signals, [self.num_electrode, letter_dim, repet_num, self.num_rep * self.n_length, 1]
        )
        # Since previously the signals have already been permuted, we don't need to permute it again.
        x_seq = design_x_mat @ signals  # add constant and background noise
        x_seq = x_seq + noise_arq_mat + eta[:, np.newaxis, np.newaxis, ...]
        print('x_seq with background noise has shape {}'.format(x_seq.shape))

        self.save_simulation_results(
            sim_folder_name,
            beta_tar, beta_ntar, s_z_sq, s_x_sq, rho,
            x_seq, eeg_code, eeg_type,
            sim_type_full,
            convol_bool, single_seq_bool, save_plot_bool,
            message
        )
        return sim_type_full + ' done!'

    def generate_super_seq_from_latency_signals(
            self, signals, design_x_mat, eeg_code, eeg_type,
            eta, s_x_sq, rho, rho_s, beta_tar, beta_ntar,
            normal_bool, t_df,
            letter_dim, repet_num, message_full,
            sim_folder_name, sim_type_full,
            convol_bool, single_seq_bool, save_plot_bool, noise_arq=None
    ):
        print('signals have shape {}'.format(signals.shape))
        signals = np.reshape(
            signals,
            [self.num_electrode, letter_dim,
             repet_num * self.num_rep * self.n_length, 1]
        )
        super_seq_signals = np.matmul(design_x_mat, signals)
        print('super_seq_signal has shape {}'.format(super_seq_signals.shape))
        super_seq_len = super_seq_signals.shape[2]

        if noise_arq is None:
            if rho_s is None:
                noise_arq = []
                for e in range(self.num_electrode):
                    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html
                    # ar = np.r_[1, -rho[e, :]]
                    # if normal_bool:
                    #     for ll in range(letter_dim):
                    #         noise_arq_e_l = arma_generate_sample(
                    #             ar, ma, super_seq_len, np.sqrt(s_x_sq[e]), burnin=0
                    #         )
                    #         noise_arq.append(noise_arq_e_l)
                    # else:
                    #     for ll in range(letter_dim):
                    #         noise_arq_e_l = arma_generate_sample(
                    #             ar, ma, letter_dim, np.sqrt(s_x_sq[e]),
                    #             distrvs=partial(np.random.standard_t, df=t_df), burnin=0
                    #         )
                    #         noise_arq.append(noise_arq_e_l)
                    _, noise_arq_e = self.create_arq_noise_close(
                        s_x_sq[e], rho[e, :], super_seq_len, letter_dim, normal_bool, t_df
                    )
                    noise_arq.append(noise_arq_e)

                noise_arq = np.reshape(
                    np.stack(noise_arq, axis=0),
                    [self.num_electrode, letter_dim, super_seq_len, 1]
                )
            else:
                print('rho_t = {}, rho_s = {}'.format(rho, rho_s))
                _, noise_arq = self.create_arq_noise_close_multi(
                    s_x_sq, rho, rho_s, super_seq_len, self.num_electrode, letter_dim, normal_bool, t_df
                )
        else:
            print('noise term has already been added to super_seq_signals.')
            # Notice that for super_seq_test, to keep the results comparable,
            # we keep adding noise with additional sequence. So the largest noise must be generated
            # before it enters this function, and no additional noise is required.
        print('noise_arq has shape {}'.format(noise_arq.shape))

        super_seq_signals = super_seq_signals + noise_arq + eta[:, np.newaxis, np.newaxis, np.newaxis]
        print('super seq with background noise has shape {}\n'.format(super_seq_signals.shape))

        self.save_simulation_results(
            sim_folder_name, beta_tar, beta_ntar,
            s_x_sq, rho, rho_s,
            super_seq_signals, eeg_code, eeg_type,
            sim_type_full, letter_dim, repet_num,
            convol_bool, single_seq_bool,
            save_plot_bool, message_full
        )
        return sim_type_full + ' done!'

    def save_partial_gen_eeg_type(
            self, eeg_type_2d, pseudo_type_2d, sim_folder_name, sim_type
    ):
        mat_dir = '{}/SIM_files/{}/sim_dat_partial_gen_{}.mat'.format(
            self.parent_path, sim_folder_name, sim_type
        )
        # print('mat_dir = {}'.format(mat_dir))
        print('{} eeg_type_2d_{} has shape {}'.format(
            sim_folder_name, sim_type, eeg_type_2d.shape)
        )

        plot_pdf = bpdf.PdfPages(
            '{}/SIM_files/{}/sim_dat_partial_gen_{}_compare.pdf'.format(
                self.parent_path, sim_folder_name, sim_type
            ))
        letter_dim = eeg_type_2d.shape[0]
        sseq_flash_len = eeg_type_2d.shape[1]
        for l in range(letter_dim):
            fig = plt.figure(figsize=(7, 6))
            plt.plot(np.arange(sseq_flash_len), pseudo_type_2d[l, :], label='Pseudo')
            plt.plot(np.arange(sseq_flash_len), eeg_type_2d[l, :]*2, label='Partial')
            plt.legend(loc='best')
            plt.xlabel('Super Sequence Flash Number')
            plt.ylabel('Label')
            plt.ylim((-3, 5))
            plt.title('letter_{}'.format(l+1))
            plot_pdf.savefig(fig)
            plt.close()
        plot_pdf.close()

        sio.savemat(mat_dir, {'eeg_type': eeg_type_2d})
        return 'save done!'

    def import_partial_gen_eeg_type(self, sim_folder_name, sim_type):
        mat_dir = '{}/SIM_files/{}/sim_dat_partial_gen_{}.mat'.format(
            self.parent_path, sim_folder_name, sim_type
        )
        partial_gen_mat = sio.loadmat(mat_dir)
        partial_gen_mat_keys, _ = zip(*partial_gen_mat.items())
        eeg_type_2d = partial_gen_mat['eeg_type']
        return eeg_type_2d

    def save_bayes_single_seq_log_lkd(
            self, rep_num_fit, rep_num_pred, train_test_label,
            method_folder_name, scenario_name, eeg_dat_type, file_subscript,
            file_name, log_lkd_pred_36, sim_dat_bool=False, **kwargs
    ):

        if sim_dat_bool:
            common_dir0 = '{}/{}/{}/{}'.format(
                self.parent_path, method_folder_name, self.sub_name_short,
                self.sub_folder_name
            )
            try:
                os.mkdir(common_dir0)
                print('Directory', common_dir0, ' is created.')
            except FileExistsError:
                print('Directory ', common_dir0, ' already exists.')

            common_dir0 = '{}/{}/{}/{}/{}'.format(
                self.parent_path, method_folder_name, self.sub_name_short,
                self.sub_folder_name, scenario_name
            )
            try:
                os.mkdir(common_dir0)
                print('Directory', common_dir0, ' is created.')
            except FileExistsError:
                print('Directory ', common_dir0, ' already exists.')
        else:
            common_dir0 = '{}/{}/{}/{}'.format(
                self.parent_path, method_folder_name, self.sub_name_short, scenario_name
            )
            try:
                os.mkdir(common_dir0)
                print('Directory', common_dir0, ' is created.')
            except FileExistsError:
                print('Directory ', common_dir0, ' already exists.')
            if 'gamma' in kwargs.keys() and 'scale' in kwargs.keys():
                scale_val = kwargs['scale']
                gamma_val = kwargs['gamma']
                common_dir0 = '{}/scale={}, gamma={}'.format(common_dir0, scale_val, gamma_val)
                try:
                    os.mkdir(common_dir0)
                    print('Directory', common_dir0, ' is created.')
                except FileExistsError:
                    print('Directory', common_dir0, ' already exists.')

        print('common_dir0 = \n {}'.format(common_dir0))
        npz_dir = '{}/{}_{}_{}_train_{}_pred_{}_{}_{}_{}.npz'.format(
            common_dir0,
            self.sub_folder_name, eeg_dat_type, train_test_label,
            rep_num_fit, train_test_label, rep_num_pred,
            file_subscript, file_name
        )
        npz_dict = {'log_lkd': log_lkd_pred_36}
        np.savez(npz_dir, **npz_dict)
        return 'single_seq_log_lkd saved!'

    def import_bayes_single_seq_log_lkd(
            self, rep_num_fit, rep_num_pred, train_test_label,
            method_folder_name, scenario_name, eeg_dat_type,
            file_subscript, file_name, sim_dat_bool, **kwargs
    ):
        if sim_dat_bool:
            common_dir0 = '{}/{}/{}/{}/{}'.format(
                self.parent_path, method_folder_name, self.sub_name_short,
                self.sub_folder_name, scenario_name
            )
        else:
            common_dir0 = '{}/{}/{}/{}'.format(
                self.parent_path, method_folder_name,
                self.sub_name_short, scenario_name
            )
            if 'scale' in kwargs.keys() and 'gamma' in kwargs.keys():
                scale_val = kwargs['scale']
                gamma_val = kwargs['gamma']
                common_dir0 = '{}/scale={}, gamma={}'.format(
                    common_dir0, scale_val, gamma_val
                )

        npz_dir = '{}/{}_{}_{}_train_{}_pred_{}_{}_{}_{}.npz'.format(
            common_dir0,
            self.sub_folder_name, eeg_dat_type, train_test_label,
            rep_num_fit, train_test_label, rep_num_pred,
            file_subscript, file_name
        )
        print('npz_dir = {}'.format(npz_dir))
        eeg_bayes_npz = np.load(npz_dir)

        return eeg_bayes_npz


