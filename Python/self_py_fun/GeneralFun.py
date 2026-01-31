from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import scipy.interpolate as ip
from scipy import linalg, stats, signal, interpolate, io as sio
import random
from functools import partial
import time, tqdm, cProfile, pstats
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as skgp_kernels
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA
from scipy.signal import savgol_filter
from scipy.stats import entropy
import itertools as itl
sns.set_context('notebook')
plt.style.use('ggplot')


class EEGGeneralFun:
    # Global constant
    # 6 by 6 grid table:
    letter_table = ['A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'I', 'J', 'K', 'L',
                    'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X',
                    'Y', 'Z', '1', '2', '3', '4',
                    '5', 'SPEAK', '.', 'BS', '!', '_']

    letter_table_sum = 36
    num_rep = 12
    flash_sum = 2
    non_flash_sum = num_rep - flash_sum
    row_set = [1, 2, 3, 4, 5, 6]
    column_set = [7, 8, 9, 10, 11, 12]
    row_column_length = 6
    VALIDATE_ARGS = True
    DAT_TYPE = 'float32'
    MENSAJE = 'When start from the terminal, ' \
              'type python -m file name without \'.py\' suffix!'
    inter_flash_period = 40  # fixed by experimental design prior to analysis
    # inter_flash_period = 256  # Only for SLO_data
    # Declare private variable starting with double underscores
    __parent_path_local = '/Users/niubilitydiu/Dropbox (University of Michigan)/Dissertation/' \
                          'Dataset and Rcode/EEG_MATLAB_data'
    __parent_path_slurm = '/home/mtianwen/EEG_MATLAB_data'

    def __init__(
            self, num_repetition, num_electrode,
            flash_and_pause_length, num_letter=19,
            sampling_rate=32,
            p300_flash_strength=1.0,
            p300_pause_strength=0.0,
            non_p300_strength=0.0,
            n_multiple=5,
            inter_flash_period=5,
            local_bool=True, *args, **kwargs
    ):
        # Specify the data generative parameters
        self.sampling_rate = sampling_rate
        self.num_repetition = num_repetition
        self.num_electrode = num_electrode
        self.flash_and_pause_length = flash_and_pause_length
        self.flash_length = int(0.2 * flash_and_pause_length)
        self.p300_flash_strength = p300_flash_strength
        self.p300_pause_strength = p300_pause_strength
        self.non_p300_strength = non_p300_strength
        self.n_multiple = n_multiple
        self.num_letter = num_letter
        self.local_bool = local_bool
        self.inter_flash_period = inter_flash_period

        # rest period after one complete sequence
        self.rest_period_length = int(n_multiple * flash_and_pause_length)
        # the length of latent Z (p100/p300_eeg_signal)
        self.n_length = int(n_multiple * flash_and_pause_length)
        self.n_iden = np.eye(self.n_length)
        self.seq_length_std = self.num_repetition * self.flash_and_pause_length * self.num_rep
        # self._index_point_1 = np.linspace(0, 0.6, endpoint=False, num=int(0.8 * self.n_length))
        # self._index_point_2 = np.linspace(0.6, 1, endpoint=True, num=self.n_length - int(0.8 * self.n_length))
        # self.index_point = np.concatenate([self._index_point_1, self._index_point_2], axis=0)
        self.total_rep = self.num_repetition * self.num_rep
        self.time_n_length = self.n_multiple * self.inter_flash_period / self.sampling_rate * 1000
        self.time_range = np.linspace(
            start=0.0, stop=self.time_n_length, num=self.n_length, dtype=self.DAT_TYPE
        )
        self.seq_length = (self.num_rep + self.n_multiple - 1) * self.flash_and_pause_length
        self.seq_iden = np.eye(self.seq_length)

        if self.local_bool:
            self.parent_path = self.__parent_path_local
        else:
            self.parent_path = self.__parent_path_slurm

    def _update_num_letter(self, num_letter):
        self.num_letter = num_letter
        print('num_letter is updated to {}'.format(self.num_letter))

    def _update_num_repetition(self, num_repetition):
        self.num_repetition = num_repetition
        print('num_repetition is updated {}'.format(self.num_repetition))

    def _update_num_electrode(self, num_electrode):
        self.num_electrode = num_electrode
        print('num_electrode is updated {}'.format(self.num_electrode))

    def _update_n_length(self, n_length):
        self.n_length = n_length
        print('n_length is updated {}'.format(self.n_length))

    def _update_flash_and_pause_length(self, flash_length, pause_length):
        if flash_length is not None:
            self.flash_length = flash_length
            print('flash_length is updated {}'.format(self.flash_length))
        if pause_length is not None:
            self.pause_length = pause_length
            print('pause_length is updated {}'.format(self.pause_length))
        self.flash_and_pause_length = self.flash_length + self.pause_length
        print('flash_and_pause_length is updated {}'.format(self.flash_and_pause_length))

    def determine_row_column_indices(self, letter):
        r"""
        :param letter: no need to be uppercase
        :return:
        """
        if letter == 'backspace':
            letter = 'BS'
        elif letter == 'SPACE':
            letter = '_'

        letter = letter.upper()
        assert letter in self.letter_table, print('The input doesn\'t belong to the letter table.')
        letter = self.letter_table.index(letter) + 1
        assert 1 <= letter <= self.letter_table_sum
        row_index = int(np.ceil(letter / self.row_column_length))
        column_index = (letter + self.row_column_length - 1) % self.row_column_length + self.row_column_length + 1
        return row_index, column_index

    def determine_letter(self, row_index, column_index):
        r"""
        :param row_index: 1-6
        :param column_index: 7-12
        :return:
        """
        assert 1 <= row_index <= self.row_column_length and self.row_column_length + 1 <= column_index <= self.num_rep
        letter_index = (row_index - 1) * self.row_column_length + (column_index - self.row_column_length)
        return self.letter_table[letter_index - 1]

    # Create a class-free user-defined function
    # to generate prior knowledge
    # Use convolution to generate simulated signal
    # Refer to the tutorial
    # https://practical-neuroimaging.github.io/on_convolution.html
    # https://www.ijser.org/paper/Wavelet-Transform-use-for-P300-Signal-Clustering-by-Self-Organizing-Map.html

    @staticmethod
    def generate_canonical_eeg_signal(
            x_input, y_input, n_length, spline_order
    ):
        assert len(x_input) == len(y_input)
        x_new = np.linspace(np.min(x_input), np.max(x_input), n_length)
        tck = ip.splrep(x_input, y_input, k=spline_order)
        y_smooth = ip.splev(x_new, tck)
        return x_new, y_smooth

    def save_slo_data_mean_fn(self, target_mean, non_target_mean):
        file_dir = '{}/SIM_summary/SLO_mean_fn.mat'.format(
            self.parent_path
        )
        print('target_mean has shape {}'.format(target_mean.shape))
        print('non_target_mean has shape {}'.format(non_target_mean.shape))
        sio.savemat(file_dir,
                    {
                        'target': target_mean,
                        'non_target': non_target_mean
                    })
        return True

    def save_slo_data_cov(self, rho, var):
        file_dir = '{}/SIM_summary/SLO_cov.mat'.format(
            self.parent_path
        )
        print('rho has shape {}'.format(rho.shape))
        print('var has shape {}'.format(var.shape))
        sio.savemat(file_dir,
                    {
                        'rho': rho,
                        'var': var
                    })
        return True

    def save_partial_gen_coefs(self, coefs):
        file_dir = '{}/SIM_summary/partial_gen_coefs.mat'.format(
            self.parent_path
        )
        print('coefs has shape {}'.format(coefs.shape))  # (design_num, N_LENGTH)
        sio.savemat(file_dir, {'coefs': coefs})
        return True

    def import_slo_data_cov(self):
        file_dir = '{}/SIM_summary/SLO_cov.mat'.format(self.parent_path)
        cov_mat = sio.loadmat(file_dir)
        cov_mat_keys, _ = zip(*cov_mat.items())
        # print(cov_mat_keys)
        rho = cov_mat['rho']
        var = cov_mat['var']
        return rho, np.squeeze(var, axis=0)

    def import_partial_gen_coefs(self):
        file_dir = '{}/SIM_summary/partial_gen_coefs.mat'.format(self.parent_path)
        partial_gen_coefs = sio.loadmat(file_dir)
        partial_gen_coefs_keys, _ = zip(*partial_gen_coefs.items())
        # print(partial_gen_coefs_keys)
        coefs = partial_gen_coefs['coefs']
        return coefs

    def import_sim_mean_fn_single(
            self, mean_fn_type, display_bool, scenario_name
    ):
        r"""
        :param mean_fn_type: integer
        :param display_bool: bool
        :param scenario_name: string
        :return:
        """

        if '25' in scenario_name:
            if mean_fn_type == 1:
                y_tar_pred = np.array([
                    0, -0.388, -0.446, -0.020,  0.586,
                    1.428,  2.327,  3.176,  3.877,  4.354,
                    4.550, 4.435, 4.007,  3.300,  2.382,
                    1.367,  0.6,  0.275,  0.150,  0.050,
                    0.000,  0.000, 0.000,  0.000,  0.000
                ])
                y_ntar_pred = np.array([
                    0, -0.05, -0.06, -0.02, 0.12,
                    0.3178, 0.4827, 0.6376, 0.7658, 0.8534,
                    0.8904, 0.8709, 0.7944, 0.6661, 0.4975,
                    0.3076, 0.1233, 0.06, 0.03, 0.01,
                    0, 0, 0, 0, 0
                ])

            else:
                y_tar_pred = np.array([
                    0, -0.7361, -1.5, -1.9183, -1.4918,
                    -0.4558, 0.8, 2.5851, 4.1239, 5.4274,
                    6.3923, 6.9847, 7.2221, 7.1381, 6.7296,
                    5.6, 4.6, 3.4, 2.0, 1.0,
                    0.5, 0.1, 0, 0, 0
                ])
                y_ntar_pred = np.array([
                    0, -0.73, -1.5, -1.9, -1.5,
                    -0.5, 0.8, 1.6, 2.2, 2.4,
                    2.5, 2.4, 2.0, 1.4, 0.9,
                    0.5, 0.1, 0, 0, 0,
                    0, 0, 0, 0, 0
                ])
        elif '35' in scenario_name:
            if mean_fn_type == 1:
                y_tar_pred = np.array([
                    0, -0.2, -0.3952, -0.4680, -0.3558,
                    -0.0831, 0.3274, 0.8443, 1.4281, 2.0370,
                    2.6324, 3.1825, 3.6623, 4.0533, 4.3405,
                    4.5104, 4.5504, 4.4496, 4.2017, 3.8096,
                    3.2899, 2.6762, 2.0202, 1.3869, 0.8455,
                    0.4536, 0.2372, 0.1703, 0.1597, 0.0474,
                    0, 0, 0, 0, 0
                ])
                y_ntar_pred = np.array([
                    0, -0.04, -0.08, -0.09, -0.08,
                    0.0354, 0.1141, 0.2103, 0.3174, 0.4289,
                    0.5380, 0.6390, 0.7272, 0.7990, 0.8514,
                    0.8823, 0.8897, 0.8723, 0.8290, 0.7599,
                    0.6669, 0.5541, 0.4291, 0.3025, 0.1877,
                    0.0989, 0.0468, 0.0326, 0.02, 0.0093,
                    0, 0, 0, 0, 0
                ])
            else:
                y_tar_pred = np.array([
                    0, -0.7544, -1.3544, -1.7876, -1.9299,
                    -1.7358, -1.2238, -0.4538, 0.4941, 1.5370,
                    2.6007, 3.6256, 4.5675, 5.3958, 6.0896,
                    6.6339, 7.0171, 7.2302, 7.2669, 7.1258,
                    6.8122, 6.3391, 5.7274, 5.0034, 4.1960,
                    3.3355, 2.4592, 1.6321, 0.9944, 0.4,
                    0, 0, 0, 0, 0
                ])
                y_ntar_pred = np.array([
                    0, -0.75, -1.35, -1.8, -1.9,
                    -1.73, -1.22, -0.45, 0.5, 1.0, 1.5, 1.9, 2.2,
                    2.4, 2.5, 2.4, 2.2, 1.9, 1.6, 1.2, 0.9,
                    0.6, 0.25, 0.1, 0.05,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ])
        else:
            if mean_fn_type == 1:
                y_tar_pred = np.array([
                    0, -0.2465, -0.5469, -0.4695, -0.1083,
                    0.4509, 1.1296, 1.8580, 2.5746, 3.2267,
                    3.7708, 4.1731, 4.4092, 4.4651, 4.3372,
                    4.0326, 3.5694, 2.9772, 2.2972, 1.5827,
                    0.8992, 0.3251, -0.0486, -0.02, -0.05,
                    0, 0, 0, 0, 0
                ])
                y_ntar_pred = np.array([
                    0, -0.04226, -0.125, -0.1, 0.0315,
                    0.1351, 0.2595, 0.3927, 0.5243, 0.6448,
                    0.7463, 0.8223, 0.8680, 0.8804, 0.8582,
                    0.8021, 0.7151, 0.6023, 0.4711, 0.3314,
                    0.1958, 0.0795, 0.0004, -0.0203, -0.012,
                    0, 0, 0, 0, 0
                ])
            else:
                y_tar_pred = np.array([
                    0., -0.4269, -1.1153, -1.7062, -1.9339,
                    -1.6727, -0.9505, 0.1063, 1.3401, 2.6107,
                    3.8171, 4.8971, 5.8132, 6.5359, 7.0345,
                    7.2775, 7.2434, 6.9336, 6.3816, 5.4,
                    4.0, 2.8, 1.8, 1.0, 0.4,
                    0.2, 0, 0, 0, 0
                ])
                y_ntar_pred = np.array([
                    0, -0.42, -1.1, -1.7, -1.93,
                    -1.67, -1.1, 0, 1.34, 2.0,
                    2.3, 2.5, 2.2, 1.5, 1.0,
                    0.5, 0.25, 0.1, 0.05, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ])

        if display_bool:
            plot_pdf = bpdf.PdfPages(
                '{}/SIM_summary/sim_mean_fn_plots_type_{}_{}.pdf'.format(
                    self.parent_path, mean_fn_type, scenario_name
                )
            )
            fig_i = plt.figure(figsize=(10, 10))
            plt.plot(self.time_range, y_tar_pred, label='Target')
            plt.plot(self.time_range, y_ntar_pred, label='Non-target')
            plt.xlim(0, 1200)  # to make all latent function consistent on x-axis
            plt.xlabel('Latency Time (ms)')
            plt.ylabel('Signal Amplitude (muV)')
            plt.legend(loc="upper right")
            plt.title('Mean Function Type {}'.format(mean_fn_type))
            plot_pdf.savefig(fig_i)
            plot_pdf.close()

        return y_tar_pred[np.newaxis, :, np.newaxis], y_ntar_pred[np.newaxis, :, np.newaxis]

    def import_sim_mean_fn_multi(
            self, mean_fn_type, display_bool, scenario_name
    ):
        r"""
        :param mean_fn_type: string
        :param display_bool: bool_variable
        :param scenario_name: string
        :return: scenario_name='TrueGen' by default
        """

        if mean_fn_type == 'multi_channel':

            y_tar_pred = np.array([
            [0.000000e+00,  8.229730e-01,  1.623497e+00,  2.379737e+00,  3.071064e+00,
             3.678620e+00,  4.185832e+00,  4.578867e+00,  4.847001e+00,  4.982922e+00,
             4.982922e+00,  4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
             3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,  6.123234e-16,
             -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01, -2.449294e-16,
             0, 0, 0, 0, 0],
            [-2.449294e-16, -5.877853e-01, -9.510565e-01, -9.510565e-01,
             -5.877853e-01,  6.123234e-16,  8.229730e-01,  1.623497e+00,
             2.379737e+00,  3.071064e+00,  3.678620e+00,  4.185832e+00,
             4.578867e+00,  4.847001e+00,  4.982922e+00,  4.982922e+00,
             4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
             3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,
             0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
             0.000000e+00,  0.000000e+00],
            [0, 0, 0, 0.000000e+00,  8.229730e-01,  1.623497e+00,  2.379737e+00,  3.071064e+00,
             3.678620e+00,  4.185832e+00,  4.578867e+00,  4.847001e+00,  4.982922e+00,
             4.982922e+00,  4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
             3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,  6.123234e-16,
             -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01, -2.449294e-16,
             0, 0]
        ])

            y_ntar_pred = np.array([
            [0.000000e+00,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
             7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
             9.965845e-01,  9.694003e-01,  9.157733e-01, 8.371665e-01,  7.357239e-01,
             6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  1.224647e-16,
            -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01, -4.898587e-17,
             0, 0, 0, 0, 0],
            [-4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01,
             -1.175571e-01,  1.224647e-16,  1.645946e-01,  3.246995e-01,
             4.759474e-01,  6.142127e-01,  7.357239e-01,  8.371665e-01,
             9.157733e-01,  9.694003e-01,  9.965845e-01,  9.965845e-01,
             9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
             6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,
             0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
             0.000000e+00,  0.000000e+00],
            [0, 0, 0, 0.000000e+00,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
             7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
             9.965845e-01,  9.694003e-01,  9.157733e-01, 8.371665e-01,  7.357239e-01,
             6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  1.224647e-16,
            -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01, -4.898587e-17,
             0, 0]
        ])

        else:
            # multi_channel_2
            y_tar_pred = np.array([
                [-2.449294e-16, -5.877853e-01, -9.510565e-01, -9.510565e-01, -5.877853e-01,
                 6.123234e-16,  8.229730e-01, 1.623497e+00,  2.379737e+00,  3.071064e+00,
                 3.678620e+00,  4.185832e+00, 4.578867e+00,  4.847001e+00,  4.982922e+00,
                 4.982922e+00,  4.847001e+00,  4.578867e+00,  4.185832e+00,  3.678620e+00,
                 3.071064e+00,  2.379737e+00,  1.623497e+00,  8.229730e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
                [1.469576e-16, -3.526712e-01, -5.706339e-01, -5.706339e-01, -3.526712e-01,
                 3.673940e-16,  4.937838e-01, 9.740982e-01,  1.427842e+00,  1.842638e+00,
                 2.207172e+00,  2.511499e+00,  2.747320e+00,  2.908201e+00,  2.989753e+00,
                 2.989753e+00,  2.908201e+00,  2.747320e+00,  2.511499e+00,  2.207172e+00,
                 1.842638e+00,  1.427842e+00,  9.740982e-01,  4.937838e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
                [-7.347882e-17, -1.763356e-01, -2.853169e-01, -2.853169e-01, -1.763356e-01,
                 1.836970e-16,  2.468919e-01,  4.870491e-01,  7.139211e-01,  9.213192e-01,
                 1.103586e+00, 1.255750e+00,  1.373660e+00,  1.454100e+00,  1.494877e+00,
                 1.494877e+00,  1.454100e+00,  1.373660e+00,  1.255750e+00,  1.103586e+00,
                 9.213192e-01,  7.139211e-01,  4.870491e-01,  2.468919e-01,  0.000000e+00,
                 0.000000e+00, 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00]
            ])

            y_ntar_pred = np.array([
                [-4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01,
                 1.224647e-16,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                 7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                 9.965845e-01,  9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                 6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
                [-4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01,
                 1.224647e-16,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                 7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                 9.965845e-01,  9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                 6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
                [-4.898587e-17, -1.175571e-01, -1.902113e-01, -1.902113e-01, -1.175571e-01,
                 1.224647e-16,  1.645946e-01,  3.246995e-01,  4.759474e-01,  6.142127e-01,
                 7.357239e-01,  8.371665e-01,  9.157733e-01,  9.694003e-01,  9.965845e-01,
                 9.965845e-01,  9.694003e-01,  9.157733e-01,  8.371665e-01,  7.357239e-01,
                 6.142127e-01,  4.759474e-01,  3.246995e-01,  1.645946e-01,  0.000000e+00,
                 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00]
            ])

        channel_dim, n_length = y_tar_pred.shape

        if display_bool:
            plot_pdf = bpdf.PdfPages(
                '{}/SIM_summary/sim_mean_fn_plots_type_{}_{}.pdf'.format(
                    self.parent_path, mean_fn_type, scenario_name)
            )
            for e in range(channel_dim):
                fig_i = plt.figure(figsize=(10, 10))
                plt.plot(self.time_range, y_tar_pred[e, :], label='Target')
                plt.plot(self.time_range, y_ntar_pred[e, :], label='Non-target')
                plt.xlim(0, 1000)  # to make all latent function consistent on x-axis
                # plt.ylim(-2, 6)
                plt.xlabel('Latency Time (ms)')
                plt.ylabel('Signal Amplitude (muV)')
                plt.legend(loc="upper right")
                plt.title('Mean Function Channel {}'.format(e+1))
                plot_pdf.savefig(fig_i)
            plot_pdf.close()

        return y_tar_pred[..., np.newaxis], y_ntar_pred[..., np.newaxis]

    def generate_single_code_specific(
            self, row_index, column_index, row_location, column_location
    ):
        # Note that the four indices start from 1.
        assert 0 < row_index <= int(self.num_rep / 2) < column_index <= self.num_rep
        assert 0 < self.flash_length <= self.flash_and_pause_length
        assert 0 < row_location <= self.num_rep and \
               0 < column_location <= self.num_rep \
               and row_location != column_location

        non_target_index = np.setdiff1d(np.arange(self.num_rep) + 1, np.array([row_index, column_index]))
        non_target_single_code = np.random.permutation(non_target_index)
        # insert row_index, column_index by row_location and column_location
        non_target_single_code = np.insert(non_target_single_code, row_location - 1, row_index, axis=0)
        non_target_single_code = np.insert(non_target_single_code, column_location - 1, column_index, axis=0)

        return non_target_single_code

    def generate_single_code_and_type(
            self, row_index, column_index,
            simple_array=True, specific=False, row_location=1, col_location=12
    ):

        assert 0 < row_index <= int(self.num_rep / 2) < column_index <= self.num_rep
        assert 0 < self.flash_length <= self.flash_and_pause_length

        if specific:
            single_code = self.generate_single_code_specific(
                row_index, column_index, row_location, col_location
            )
        else:
            single_code = np.random.permutation(self.num_rep) + 1

        row_index_permute = np.where(single_code == row_index)[0][0]
        column_index_permute = np.where(single_code == column_index)[0][0]
        single_type = np.ones(self.num_rep) * self.non_p300_strength

        if simple_array:
            single_type[row_index_permute] = self.p300_flash_strength
            single_type[column_index_permute] = self.p300_flash_strength
        else:
            single_type[row_index_permute] = self.p300_pause_strength
            single_type[column_index_permute] = self.p300_pause_strength
            single_code = np.repeat(single_code, self.flash_and_pause_length)
            single_type = np.repeat(single_type, self.flash_and_pause_length)

            low_index_row_flash = row_index_permute * self.flash_and_pause_length
            upp_index_row_flash = low_index_row_flash + self.flash_length
            low_index_col_flash = column_index_permute * self.flash_and_pause_length
            upp_index_col_flash = low_index_col_flash + self.flash_length

            single_type[low_index_row_flash:upp_index_row_flash] = self.p300_flash_strength
            single_type[low_index_col_flash:upp_index_col_flash] = self.p300_flash_strength

            for code_index in range(self.num_rep):
                low_index_pause = code_index * self.flash_and_pause_length + self.flash_length
                upp_index_pause = (code_index + 1) * self.flash_and_pause_length
                single_code[low_index_pause:upp_index_pause] = 0.0

        return single_code, single_type

    def generate_multiple_code_and_type(
            self, letter=None, simple_array=True, specific=True, row_loc=1, col_loc=12, repetition_num=None
    ):

        multiple_code = []
        multiple_type = []
        if letter is None:
            letter = np.random.choice(self.letter_table)
        row_i, column_i = self.determine_row_column_indices(letter)
        print('letter {}, row index is {}, column index is {}'.format(letter, row_i, column_i))
        if repetition_num is None:
            repetition_num = self.num_repetition

        for j in range(repetition_num):
            single_code, single_type = self.generate_single_code_and_type(
                row_i, column_i, simple_array, specific, row_loc, col_loc
            )
            multiple_code.append(single_code)
            multiple_type.append(single_type)
        multiple_code = np.stack(multiple_code, axis=0)
        multiple_type = np.stack(multiple_type, axis=0)

        if simple_array:
            multiple_code = np.reshape(multiple_code, [self.num_rep * self.num_repetition])
            multiple_type = np.reshape(multiple_type, [self.num_rep * self.num_repetition])
        else:
            multiple_code = np.reshape(multiple_code,
                                       [self.num_rep * self.num_repetition * self.flash_and_pause_length])
            multiple_type = np.reshape(multiple_type,
                                       [self.num_rep * self.num_repetition * self.flash_and_pause_length])
            multiple_code = np.concatenate([multiple_code, np.zeros([self.rest_period_length])], axis=0)
            multiple_type = np.concatenate([multiple_type, np.zeros([self.rest_period_length])], axis=0)
        return multiple_code, multiple_type, row_i, column_i, letter

    def generate_eeg_type_from_letter_eeg_code(self, eeg_code, target_letter):
        assert len(eeg_code.shape) == 1, print('Convert the input eeg_code to 1d array!')
        assert target_letter in self.letter_table, print('Wrong input!')
        row_id, col_id = self.determine_row_column_indices(target_letter)
        eeg_type = np.zeros_like(a=eeg_code, dtype=self.DAT_TYPE)
        eeg_type[eeg_code == row_id] = 1
        eeg_type[eeg_code == col_id] = 1
        return eeg_type

    def generate_multiple_letter_code_and_type(
            self, letters, simple_array=True, specific=False, row_loc=1, col_loc=12
    ):

        r"""
        :param letters: A list of letters that belongs to the grid defined in the self.letter_table
        :param simple_array: bool, whether we export the arrays with unnecessary zeros.
        :param specific: bool
        :param row_loc: integer
        :param col_loc: integer
        :return: two 1d-array including eeg_type and eeg_code

        note: the repetition is hidden in the self.num_repetition if we don't specify it.
        """

        if specific:
            print('target flashes are fixed at {} and {}.'.format(row_loc, col_loc))
        else:
            print('target flashes are located at random.')
        eeg_code = []
        eeg_type = []
        for _, letter_i in enumerate(letters):
            eeg_code_i, eeg_type_i, _, _, _ = \
                self.generate_multiple_code_and_type(letter_i, simple_array, specific, row_loc, col_loc)
            eeg_code.append(eeg_code_i)
            eeg_type.append(eeg_type_i)
        if simple_array:
            eeg_code = np.reshape(np.stack(eeg_code, axis=0),
                                  [self.num_letter * self.num_repetition * self.num_rep])
            eeg_type = np.reshape(np.stack(eeg_type, axis=0),
                                  [self.num_letter * self.num_repetition * self.num_rep])
        else:
            dim_temp = self.num_repetition * self.num_rep * self.flash_and_pause_length
            eeg_code = np.reshape(np.stack(eeg_code, axis=0),
                                  [self.num_letter * (dim_temp + self.rest_period_length)])
            eeg_type = np.reshape(np.stack(eeg_type, axis=0),
                                  [self.num_letter * (dim_temp + self.rest_period_length)])
        return eeg_code, eeg_type

    @staticmethod
    def generate_multiple_mis_specified_type(
            eeg_type_1d, prop=0.1
    ):
        # Find existing target flashes from original eeg_type_1d
        print('We randomly select {}% target stimuli to be missed.'.format(prop * 100))
        target_ids = np.where(eeg_type_1d == 1)[0].tolist()
        tar_total_len = len(target_ids)
        mis_target_ids = random.sample(target_ids, int(prop*tar_total_len))
        eeg_type_1d_mis = np.copy(eeg_type_1d)
        eeg_type_1d_mis[mis_target_ids] = 0
        return eeg_type_1d_mis

    # Generate pseudo eeg signals using canonical eeg signal
    def generate_pseudo_eeg_signals(
            self, eeg_code_subset,
            target_row_index, target_col_index,
            target_y, non_target_y,
            target_strength, non_target_strength,
            target_sigma_sq, non_target_sigma_sq, x_sigma_sq
    ):
        nn = len(eeg_code_subset)
        times = np.arange(0, nn, 1)
        mm = self.n_length
        extended_eeg_signal = np.zeros([nn + mm - 1])
        target_set = np.array([target_row_index, target_col_index])

        for i in range(nn):
            input_value = eeg_code_subset[i]
            if input_value > 0:
                if np.isin(input_value, target_set):
                    target_y_error = np.random.multivariate_normal(
                        np.zeros(mm), np.diag(target_sigma_sq * np.ones(mm))
                    )
                    extended_eeg_signal[i:(i + mm)] += target_strength * target_y + target_y_error
                else:
                    non_target_y_error = np.random.multivariate_normal(
                        np.zeros(mm), np.diag(non_target_sigma_sq * np.ones(mm))
                    )
                    extended_eeg_signal[i:(i + mm)] += non_target_strength * non_target_y + non_target_y_error

        extra_times = np.arange(mm - 1) * 1 + nn
        times_and_tails = np.concatenate((times, extra_times))

        error = np.random.multivariate_normal(np.zeros(nn + mm - 1),
                                              np.diag(x_sigma_sq * np.ones(nn + mm - 1)))
        extended_eeg_signal += error

        return times_and_tails, extended_eeg_signal

    @staticmethod
    def create_mexican_hat_curve(a, b, c, d, time_range):
        phi_time = a * (1 - b * (time_range - c) ** 2) * np.exp(-d * (time_range - c) ** 2)
        return phi_time

    # Assume evenly distributed time
    def create_group_mexican_hat_curve(self, aa, bb, cc, dd, grp_num, time_range):
        phi_times = [0]
        for i in range(grp_num):
            phi_time = self.create_mexican_hat_curve(aa[i], bb[i], cc[i], dd[i], time_range)
            phi_times = np.concatenate([phi_times, phi_time], axis=0)
        return phi_times[1:]

    @staticmethod
    def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):

        r"""
        sigma_sq: scalar_like,
        first_column_except_1: array_like, 1d-array, except diagonal 1.
        return:
            2d-array with dimension (len(first_column)+1, len(first_column)+1)
        """
        first_column = np.r_[1, first_column_except_1]
        del first_column_except_1
        assert first_column[0] == 1, print('the first entry should be 1!')
        cov_mat = sigma_sq * linalg.toeplitz(first_column)
        return cov_mat

    def create_hetero_toeplitz_cov_mat(self, sigma_sq_vec, first_column):
        r"""
        sigma_sq_vec: 1d-array of length n, must be positive
        first_column: 1d-array, have the same dimension as sigma_vec

        return:
            2d-matrix of (n, n)
        """
        assert len(sigma_sq_vec) == len(first_column), print('Different input vector shapes!')
        sigma_vec = np.sqrt(sigma_sq_vec)
        cov_mat = sigma_vec * self.create_toeplitz_cov_mat(1, first_column) * sigma_vec
        return cov_mat

    def create_compound_symmetry_cov_mat(self, sigma_sq, rho, n):

        r"""
        sigma_sq: scalar
        rho: scalar
        n: integer
        return:
            2d matrix with dimension (n, n)
        """
        # first_column = np.concatenate([[1], rho * np.ones([n - 1])], axis=0)
        first_column = rho * np.ones([n - 1])
        cov_mat = self.create_toeplitz_cov_mat(sigma_sq, first_column)
        return cov_mat

    def create_hetero_compound_symmetry_cov_mat(self, sigma_vec, rho):

        r"""
        sigma_sq_vec: 1d-array of size n, must be positive
        rho: scalar
        return:
            2d-matrix of (n, n)
        """
        n = len(sigma_vec)
        cov_mat = np.matmul(np.matmul(np.diag(sigma_vec), self.create_compound_symmetry_cov_mat(1, rho, n)),
                            np.diag(sigma_vec))
        return cov_mat

    def create_ar1_cov_mat(self, sigma_sq, rho, n):

        r"""
        sigma_sq: scalar
        rho: scalar, should be within -1 and 1.
        n: integer
        return: 2d-matrix of (n,n)
        """
        rho_tile = rho * np.ones([n - 1])
        first_column_except_1 = np.cumprod(rho_tile)
        cov_mat = self.create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
        return cov_mat

    def create_hetero_ar1_cov_mat(self, sigma_vec, rho):

        r"""

        sigma_sq_vec: 1d-array of size n, must be positive
        rho: scalar, between -1 and 1

        return:
            2d-matrix of size (len(sigma_sq_vec), len(sigma_sq_vec))
        """
        n = len(sigma_vec)
        cov_mat = np.diag(sigma_vec) @ self.create_ar1_cov_mat(1, rho, n) @ np.diag(sigma_vec)
        return cov_mat

    @staticmethod
    def create_ar1_pres_mat_close(
            sigma_sq, rho, n
    ):
        r"""
        :param sigma_sq: scalar_like,
        :param rho: scalar_like,
        :param n: integer
        :return: square matrix of size (n, n)
        For AR(1) with sigma_sq and rho,
            Kac-Murdock-Szego matrix provides an analytical solution to the inverse:
            Given Corr = AR(1), rho = q, sigma_sq
            Corr_inv = 1/sigma_sq/(1-rho**2) * tri-diagonal matrix, where
            main diagonal = (1, 1+rho**2, ..., 1+rho**2, 1)
            +1/-1 off-diagonal = -rho
            The final precision matrix, P = U**(-1/2) @ Corr_inv @ U**(-1/2) if we have heterogeneous sigma_sq.
            https://mathoverflow.net/questions/65795/inverse-of-an-ar1-or-laplacian-or-kac-murdock-szegö-matrix/65819
        """
        pres_mat = -rho * (np.eye(n, k=1) + np.eye(n, k=-1)) + (1 + rho ** 2) * np.eye(n)
        pres_mat[0, 0] = 1
        pres_mat[-1, -1] = 1
        pres_mat = pres_mat / (sigma_sq * (1 - rho**2))

        return pres_mat

    def create_std_ar1_pres_candidate(
            self, rho_set, n
    ):
        r"""
        :param rho_set: list of number
        :param n: integer, matrix size
        :return: list of pres_chky, pres_chky_t, log_det arrays
        """
        assert 0 <= np.min(rho_set) <= np.max(rho_set) < 1
        # rho_level = len(rho_set)
        pres_chky_init = []
        pres_chky_t_set_init = []
        for _, rho_id in enumerate(rho_set):
            pres_chky_id = self.create_ar1_pres_mat_close(1, rho_id, n)
            pres_chky_id_t = np.linalg.cholesky(pres_chky_id).T
            pres_chky_init.append(pres_chky_id)
            pres_chky_t_set_init.append(pres_chky_id_t)
        pres_chky_init = np.stack(pres_chky_init, axis=0)
        pres_chky_t_set_init = np.stack(pres_chky_t_set_init, axis=0)
        _, logdet_pres_chky_set = np.linalg.slogdet(pres_chky_t_set_init)

        return pres_chky_init, pres_chky_t_set_init, logdet_pres_chky_set

    @staticmethod
    def create_ar2_pres_mat(sigma_sq, rho, n):
        assert n >= 3, print('not enough dimension to support AR(2) structure.')
        cov_mat = np.eye(n)
        for i in range(n-1):
            if i > 0:
                cov_mat[i, i+1] = rho[0] + rho[1] * cov_mat[i-1, i]
            else:
                cov_mat[i, i+1] = rho[0]
            for j in range(min(i+2, n), n):
                cov_mat[i, j] = cov_mat[i, j-1] * rho[0] + cov_mat[i, j-2] * rho[1]
        cov_mat = cov_mat + cov_mat.T - np.eye(n)
        cov_chky_inv = np.linalg.inv(np.linalg.cholesky(cov_mat))

        pres_mat = np.matmul(cov_chky_inv.T, cov_chky_inv)
        # Strictly enforce zeros to those outside the banded
        pres_mat_clear = np.zeros_like(pres_mat)
        for i in range(n):
            pres_mat_clear[i, i] = pres_mat[i, i]
        for i in range(n-1):
            pres_mat_clear[i, i + 1] = pres_mat[i, i + 1]
            pres_mat_clear[i + 1, i] = pres_mat[i + 1, i]
        for i in range(n-2):
            pres_mat_clear[i, i + 2] = pres_mat[i, i + 2]
            pres_mat_clear[i + 2, i] = pres_mat[i + 2, i]
        return sigma_sq * cov_mat, pres_mat_clear / sigma_sq

    def create_std_ar2_pres_candidate(
            self, rho_set, n
    ):
        r"""
        :param rho_set: list of 1-d array
        :param n: integer, matrix size
        :return: list of pres_chky, pres_chky_t and log_det arrays
        """

        pres_chky_set = []
        pres_chky_t_set = []
        for _, rho_id in enumerate(rho_set):
            _, pres_chky_id = self.create_ar2_pres_mat(1, rho_id, n)
            pres_chky_id_t = np.linalg.cholesky(pres_chky_id).T
            pres_chky_set.append(pres_chky_id)
            pres_chky_t_set.append(pres_chky_id_t)
        pres_chky_set = np.stack(pres_chky_set, axis=0)
        pres_chky_t_set = np.stack(pres_chky_t_set, axis=0)
        _, logdet_pres_chky_set = np.linalg.slogdet(pres_chky_t_set)

        return pres_chky_set, pres_chky_t_set, logdet_pres_chky_set

    def produce_pre_compute_rhos(
            self, q, n, level_const=10
    ):
        r"""
        :param q: integer
        :param n: integer
        :param level_const: integer
        :return:
        for q=2, the sufficient and necessary condition for invertible matrix is
        rho_1 + rho_2 < 1
        """
        assert q <= 2, print('We only support up to AR(2) now.')
        if q == 1:
            rho_set = list(np.arange(level_const) / 10)
            rho_level = len(rho_set)
            pres_chky_init, pres_chky_t_set_init, logdet_pres_chky_set = self.create_std_ar1_pres_candidate(
                rho_set, n
            )
        else:
            rho_set_init = list(np.arange(level_const) / 10)
            rho_set = []
            for i in range(level_const):
                for j in range(level_const):
                    if rho_set_init[i] == rho_set_init[j] == 0:
                        rho_set.append(np.array([0, 0]))
                    elif rho_set_init[i] + rho_set_init[j] < 1 and rho_set_init[i] > rho_set_init[j]:
                        rho_set.append(np.array([rho_set_init[i], rho_set_init[j]]))
                    else:
                        pass
            rho_level = len(rho_set)
            pres_chky_init, pres_chky_t_set_init, logdet_pres_chky_set = self.create_std_ar2_pres_candidate(
                rho_set, n
            )
        # print('rho_set = {}'.format(rho_set))
        # else:
        #     rho_set = [np.array([0])]
        #     rho_level = 1
        #     # pres_chky_init = np.eye(n)
        #     pres_chky_init, pres_chky_t_set_init, logdet_pres_chky_set = self.generate_std_ar1_pres_candidate(
        #         rho_set, n
        #     )

        logdet_pres_chky_set = np.tile(logdet_pres_chky_set[:, np.newaxis], [1, self.num_electrode])
        pres_chky_t_set = pres_chky_t_set_init[:, np.newaxis, np.newaxis, :, :]
        pres_chky_t_set = np.tile(pres_chky_t_set, [1, self.num_electrode, 1, 1, 1])
        return rho_set, rho_level, pres_chky_t_set_init, pres_chky_t_set, logdet_pres_chky_set

    @staticmethod
    def create_rbf_kernel_fn(
            scale, x_input
    ):
        rbf = skgp_kernels.RBF(scale)
        cov = rbf.__call__(x_input)
        cov = cov + np.eye(cov.shape[0]) * 1e-10
        return cov

    @staticmethod
    def create_exp_sine_sq_kernel_fn(
            scale, p, x_input
    ):
        exp_sine_sq = skgp_kernels.ExpSineSquared(scale, p)
        cov = exp_sine_sq.__call__(x_input)
        cov = cov + np.eye(cov.shape[0]) * 1e-10
        return cov

    @staticmethod
    def create_matern_kernel_fn(
            scale, nu, x_input
    ):
        matern = skgp_kernels.Matern(scale, nu)
        cov = matern.__call__(x_input)
        cov = cov + np.eye(cov.shape[0]) * 1e-10
        return cov

    @staticmethod
    def create_gamma_exp_kernel_fn(
            scale, gamma, x_input
    ):

        assert 0 <= gamma <= 2, print('Invalid gamma input.')
        base_kernel = skgp_kernels.RBF(scale)
        cov = -np.log(base_kernel.__call__(x_input))*2
        cov = np.exp(-cov ** (gamma / 2)) + np.eye(cov.shape[0]) * 1e-10
        return cov

    @staticmethod
    def create_rational_quadratic_kernel_fn(
            scale, alpha, x_input
    ):
        rq = skgp_kernels.RationalQuadratic(scale, alpha)
        cov = rq.__call__(x_input)
        cov = cov + np.eye(cov.shape[0]) * 1e-10
        return cov

    def create_permute_beta_id(self, letter_dim, repet_dim, eeg_type):

        r"""
        letter_dim: integer
        repet_dim: integer
        eeg_type: 1d-array
            the binary value of the stimuli
        """

        dim_temp = letter_dim * repet_dim * self.num_rep
        assert eeg_type.shape == (dim_temp,), \
            print('eeg_type has wrong input shape {}, should have {}.'.format(eeg_type.shape, dim_temp))
        id_beta = np.zeros([dim_temp]) - 99
        # Since we have misspecified cases, we use general way to compute target stimuli
        tar_stm_total = int(np.sum(eeg_type))
        id_beta[eeg_type == 1] = np.arange(tar_stm_total)
        id_beta[eeg_type != 1] = np.arange(tar_stm_total, dim_temp)
        return id_beta.astype('int')

    @staticmethod
    def block_diagonal_mat(gamma_mat, channel_dim=1):
        gamma_mat = np.stack([np.diag(gamma_mat[i, :])
                              for i in range(channel_dim)], axis=0)

        return gamma_mat

    def generate_latency_signals(
            self, s_x_sq, rho, rho_s,
            sample_tar, sample_ntar,
            mean_fn_tar, mean_fn_ntar, permute_id,
            eeg_code_rs, eeg_type_rs,
            letter_dim, repet_num,
            message, sim_name,
            sim_type, save_plots
    ):
        r"""
        :param s_x_sq: array_like, (channel_dim,)
        :param rho:
        :param rho_s:
        :param sample_tar: array_like, (channel_dim, noise_size_tar, n_length, 1)
        :param sample_ntar: array_like, (channel_dim, noise_size_ntar, n_length, 1)
        :param mean_fn_tar: array_like, (channel_dim, n_length, 1)
        :param mean_fn_ntar: array_like, (channel_dim, n_length, 1)
        :param permute_id: 1d array_like, (noise_size_total,)
        :param eeg_code_rs: 2d array_like, (letter_dim, num_repetition*12)
        :param eeg_type_rs: 2d array_like, (letter_dim, num_repetition*12)
        :param letter_dim:
        :param repet_num:
        :param message: str_like
        :param sim_name: str_like
        :param sim_type: str_like
        :param save_plots: bool_like
        :return: array_like, (channel_dim, letter_dim, num_repetition*12, n_length, 1)

         If num_electrode = 1, the input dimension for num_electrode is collapsed.
         For latency signals, we will add s_z_sq * kappa_z to increase the complexity.
         The model-based simulation study has no latency level noise.
        """

        sample_tar = sample_tar + mean_fn_tar[:, np.newaxis, ...]
        sample_ntar = sample_ntar + mean_fn_ntar[:, np.newaxis, ...]
        sample = np.concatenate([sample_tar, sample_ntar], axis=1)
        sample = sample[:, permute_id, ...]
        sample = np.reshape(
            sample, [self.num_electrode,
                     self.num_letter,
                     self.num_repetition * self.num_rep,
                     self.n_length, 1]
        )
        print('pseudo sample has shape {}\n'.format(sample.shape))
        self.save_simulation_results(
            sim_name, mean_fn_tar, mean_fn_ntar, s_x_sq, rho, rho_s,
            sample, eeg_code_rs, eeg_type_rs, sim_type,
            letter_dim, repet_num,
            convolution_bool=False, single_seq_bool=False,
            save_plots_bool=save_plots, message=message
        )

    def save_simulation_results(
            self, sim_folder_short,
            any_tar, any_ntar, s_x_sq, rho, rho_s,
            pseudo_signals, eeg_code, eeg_type, sim_type,
            letter_dim, repet_num,
            convolution_bool=True, single_seq_bool=True,
            save_plots_bool=True, message=None
    ):
        r"""
        :param sim_folder_short: string for folder name
        :param any_tar:
        :param any_ntar:
        :param s_x_sq: AR(q), (num_electrode, seq_length, seq_length)
        :param rho:
        :param rho_s:
        :param pseudo_signals:
        :param eeg_code:
        :param eeg_type:
        :param sim_type: string, simulation type, serve suffix
        :param letter_dim:
        :param repet_num:
        :param convolution_bool:
        :param single_seq_bool:
        :param save_plots_bool:
        :param message:
        :return:
        """
        dir_name0 = '{}/SIM_files/{}'.format(self.parent_path, sim_folder_short)
        try:
            os.mkdir(dir_name0)
            print('Directory', dir_name0, 'is created.\n')
        except FileExistsError:
            print('Directory', dir_name0, 'already exists.\n')

        dir_name = '{}/{}'.format(dir_name0, message)
        try:
            os.mkdir(dir_name)
            print('Directory', dir_name, 'is created.\n')
        except FileExistsError:
            print('Directory', dir_name, 'already exists.\n')

        if save_plots_bool:
            plot_pdf = bpdf.PdfPages(
                '{}/signal_plots_{}.pdf'.format(dir_name, sim_type)
            )
            if convolution_bool:
                if single_seq_bool:
                    seq_length = pseudo_signals.shape[3]
                    time_seq_length = seq_length / self.sampling_rate * 1000  # (ms)
                    time_range_seq = np.linspace(
                        start=0.0, stop=time_seq_length, num=seq_length, dtype=self.DAT_TYPE
                    )
                    for i in range(letter_dim):
                        for j in range(self.num_electrode):
                            for k in range(repet_num):
                                eeg_type_expand_iter = np.zeros([seq_length])
                                target_ids = np.where(eeg_type[i, k * self.num_rep:(k + 1) * self.num_rep] == 1)[0] * \
                                             self.flash_and_pause_length
                                eeg_type_expand_iter[target_ids] = 1
                                fig = plt.figure(figsize=(7, 6))
                                plt.plot(time_range_seq, pseudo_signals[j, i, k, :, 0], label="Convoluted signals")
                                plt.plot(time_range_seq, eeg_type_expand_iter, label="Target")
                                plt.legend(loc="upper right")
                                plt.xlabel('Time (ms)')
                                plt.ylabel('Signal Amplitude (muV)')
                                plt.title('letter_{}_chan_{}_seq_{}'
                                          .format(i + 1, j + 1, k + 1))
                                plot_pdf.savefig(fig)
                                plt.close()
                else:
                    # pseudo_signals have shape, (letter_dim, channel_dim, super_seq_length, 1)
                    super_seq_length = pseudo_signals.shape[2]  # total number of repetitions
                    time_super_seq_length = super_seq_length / self.sampling_rate * 1000  # (ms)
                    time_range_super_seq = np.linspace(
                        start=0.0, stop=time_super_seq_length, num=super_seq_length, dtype=self.DAT_TYPE
                    )
                    for i in range(letter_dim):
                        for j in range(self.num_electrode):
                            eeg_type_expand_iter = np.zeros([super_seq_length])
                            target_ids = np.where(eeg_type[i, :] == 1)[0] * self.flash_and_pause_length
                            eeg_type_expand_iter[target_ids] = 1
                            fig = plt.figure(figsize=(14, 12))
                            plt.plot(time_range_super_seq, pseudo_signals[j, i, :, 0], label="Signals")
                            plt.plot(time_range_super_seq, eeg_type_expand_iter, label="Target")
                            plt.legend(loc="upper right")
                            plt.xlabel('Time (ms)')
                            plt.ylabel('Signal Amplitude (muV)')
                            plt.title('letter_{}_chan_{}'.format(i + 1, j + 1))
                            plot_pdf.savefig(fig)
                            plt.close()

            else:
                max_val = np.max(pseudo_signals) + 0.5
                min_val = np.min(pseudo_signals) - 0.5
                # Since we don't distinguish between training and testing for sim-gen-direct,
                # We keep the range of j as num_rep * num_repetition.
                for i in range(letter_dim):
                    print('letter-index={}\n'.format(i + 1))
                    for j in range(self.num_rep * repet_num):
                        if j % 12 == 0:
                            print('flash-index={}'.format(j))
                        for k in range(self.num_electrode):
                            fig = plt.figure(figsize=(7, 6))
                            plt.plot(self.time_range, pseudo_signals[k, i, j, :, 0], label="observed")
                            if eeg_type[i, j] == 1:
                                plt.plot(self.time_range, any_tar[k, :, 0], label="noise-free")
                            else:
                                plt.plot(self.time_range, any_ntar[k, :, 0], label="noise-free")
                            plt.title('letter_{}_type_{}_code_{}_chan_{}'.format(
                                i + 1, int(eeg_type[i, j]), int(eeg_code[i, j]), k + 1)
                            )
                            plt.legend(loc="upper right")
                            plt.xlabel('Time (ms)')
                            plt.ylabel('Signal Amplitude (muV)')
                            plt.ylim(min_val, max_val)
                            plot_pdf.savefig(fig)
                            plt.close()
            plot_pdf.close()

        mat_dir = '{}/sim_dat_{}.mat'.format(
            dir_name, sim_type
        )
        print('mat_dir = {}'.format(mat_dir))
        if rho_s is None:
            rho_s = 0
        sio.savemat(mat_dir,
                    {
                        'tar': any_tar,
                        'ntar': any_ntar,
                        'rho_s': rho_s,
                        's_x_sq': s_x_sq,  # ar(q) matrix
                        'rho': rho,
                        'signals': pseudo_signals,
                        'eeg_code': eeg_code,
                        'eeg_type': eeg_type,
                        'message': message
                    })

        return 'save done!'

    @staticmethod
    def create_banded_psd_matrix(input_matrix, q):
        r"""
        :param input_matrix: usually covariance matrix or precision matrix,
            (..., n_length, n_length)
        :param q: positive integer, the auto-regressive integer q
        :return: the banded matrix with the same shape as input_matrix
            when |x - y| <= q, the values are the same, otherwise, the values are zero.
        """
        output_matrix = np.copy(input_matrix)
        _, x, y = np.indices(input_matrix.shape)
        output_matrix[np.abs(x - y) > q] = 0

        return output_matrix

    @staticmethod
    def is_pos_def(input_matrix):
        if np.allclose(input_matrix, input_matrix.T):
            try:
                np.linalg.cholesky(input_matrix)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def compute_ising_log_prior(
            self, gamma_mat_pre, tau, beta_ising, gamma_neighbor
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
            Notice that here the input is 0,1 coding scheme, while the standard definition is -1,1 coding scheme.
        """
        assert 0 <= tau < self.n_length, print('tau has incorrect index!')
        gamma_neighbor_mat = gamma_mat_pre[:, max(tau - gamma_neighbor, 0):(tau + gamma_neighbor + 1)]
        z_neighbor = -np.ones_like(gamma_neighbor_mat)
        z_neighbor[gamma_neighbor_mat == gamma_mat_pre[:, tau, np.newaxis]] = 1
        z_neighbor = np.sum(z_neighbor, axis=-1) - 1  # exclude itself (which is always 1)

        return beta_ising * z_neighbor

    @staticmethod
    def generate_proposal_tgp_zeta_state(
            old_tpg_zeta, tpg_step_size
    ):
        r"""
        :param old_tpg_zeta: (channel_dim, u)
        :param tpg_step_size: (channel_dim,)
        :return: generate new tpg zeta without restriction
        """
        u = old_tpg_zeta.shape[1]
        new_tpg_zeta = np.random.multivariate_normal(
            mean=np.zeros_like(tpg_step_size),
            cov=np.diag(tpg_step_size),
            size=u
        )
        return np.transpose(new_tpg_zeta) + old_tpg_zeta

    @staticmethod
    def compute_log_prior_ratio_tpg_zeta(
            tpg_zeta_old, tpg_zeta_new, eigen_val
    ):
        r"""
        :param tpg_zeta_old: (channel_dim, u)
        :param tpg_zeta_new: (channel_dim, u)
        :param eigen_val: (u,)
        :return:
        """

        tpg_zeta_rv = stats.multivariate_normal(
            mean=np.zeros_like(eigen_val),
            cov=np.diag(eigen_val)
        )
        tpg_zeta_new_log_pdf = tpg_zeta_rv.logpdf(tpg_zeta_new)
        tpg_zeta_old_log_pdf = tpg_zeta_rv.logpdf(tpg_zeta_old)

        return tpg_zeta_new_log_pdf - tpg_zeta_old_log_pdf

    def generate_proposal_sigma_sq_state(
            self, old_state, s_sq_stpsize, channel_ids
    ):
        r"""
        args:
        -----
            old_state: array_like, (num_electrode,)
            zeta: array_like, (num_electrode,)
            channel_ids: array_like, optional

        return:
        -----
            proposed state of var_sigma_sq by symmetric random walk
        note:
        -----
            proposal distribution to generate variance parameter
        """
        if channel_ids is None:
            channel_ids = np.arange(self.num_electrode)
        channel_dim = len(channel_ids)
        new_state = stats.multivariate_normal(
            mean=old_state, cov=s_sq_stpsize * np.eye(channel_dim)
        ).rvs(1)
        if channel_dim == 1:
            new_state = np.array([new_state])
        accept_init = np.ones_like(new_state)
        for e in range(channel_dim):
            if new_state[e] <= 0.05:
                new_state[e] = np.copy(old_state[e])
                accept_init[e] = 0

        return new_state, accept_init

    @staticmethod
    def generate_proposal_proportion_state(
            old_state, **kwargs
    ):
        accept_init = 1
        option = kwargs['option']
        if option == 'rw':
            if 'step_size' in kwargs.keys():
                step_size = kwargs['step_size']
                new_state = np.random.normal(
                    loc=old_state, scale=np.sqrt(step_size), size=1
                )[0]
            else:
                new_state = np.copy(old_state)
        else:
            # Independent MH
            if 'scale' in kwargs.keys():
                scale = kwargs['scale']
                new_state = np.random.exponential(scale=scale, size=1)[0]
            else:
                new_state = np.copy(old_state)
        # print('new_state = {}'.format(new_state))

        if new_state <= 0.01 or new_state >= 0.99:
            new_state = np.copy(old_state)
            accept_init = 0

        return new_state, accept_init

    @staticmethod
    def compute_log_prior_ratio_s_sq(s_sq_old, s_sq_new, alpha_s, beta_s, channel_ids):
        r"""
        args:
        -----
            s_sq_old: array_like, previous state of s_sq, (num_electrode,)
            s_sq_new: array_like, proposed state of s_sq, (num_electrode,)
            tau: integer, 0 to n_length
            alpha_s: scalar value > 0
            beta_s: rate value > 0 (rate = 1/scale)
        return:
        -----
            array_like value, (num_electrode,)
        note:
        -----
            we assume s_sq ~ InvGamma(alpha_s, beta_s), rho ~ Uniform(0, 1)
            Not to be confused, I always use alpha, beta parametrization
            where beta_s is the inverse of scale.
        """
        assert s_sq_new.shape == s_sq_old.shape == (len(channel_ids),)
        s_sq_old_log_pdf = stats.gamma.logpdf(s_sq_old, a=alpha_s, loc=0, scale=1/beta_s)
        s_sq_new_log_pdf = stats.gamma.logpdf(s_sq_new, a=alpha_s, loc=0, scale=1/beta_s)

        return s_sq_new_log_pdf - s_sq_old_log_pdf

    @staticmethod
    def compute_log_prior_ratio_proportion(
            rho_old, rho_new, a=1, b=2
    ):
        r"""
        :param rho_old:
        :param rho_new:
        :param a: positive value
        :param b: positive value
        :return: Assume rho[e, p] ~ Unif(0, 1)
        """
        # if channel_ids is None:
        #     channel_ids = np.arange(self.num_electrode)
        # channel_dim = len(channel_ids)
        # If we assume uniform (0, 1) prior, then they share the same pdf
        new_log_pdf = stats.beta.logpdf(rho_new, a, b)
        old_log_pdf = stats.beta.logpdf(rho_old, a, b)

        return new_log_pdf - old_log_pdf

    @staticmethod
    def compute_beta_credible_bound(beta_mcmc):
        beta_mean = np.mean(beta_mcmc, axis=0)
        beta_low = np.quantile(beta_mcmc, q=0.025, axis=0)
        beta_upp = np.quantile(beta_mcmc, q=0.975, axis=0)
        return [beta_mean, beta_low, beta_upp]

    def create_arq_noise_close(
            self, sigma_sq, rho, seq_length, size, normal_bool=True, df=5
    ):
        r"""
        :param sigma_sq: scalar_like, diagonal variance
        :param rho: scalar_like,
        :param seq_length: integer, the dimension of the covariance matrix
        :param size: integer, batch size of noise variable
        :param normal_bool: bool_like,
        :param df: integer
        :return:
        """
        if len(rho) == 1:
            # Evaluate whether rho is of dimension 1
            cov_mat = self.create_ar1_cov_mat(sigma_sq, rho, seq_length)
        else:
            cov_mat, _ = self.create_ar2_pres_mat(sigma_sq, rho, seq_length)
        cov_half_mat = np.linalg.cholesky(cov_mat)
        # del cov_mat
        if normal_bool:
            print('The background noise follows normal distribution.')
            std_noise = np.random.normal(loc=0, scale=1, size=(size, seq_length, 1))
        else:
            print('The background noise follows student-t distribution with df {}'.format(df))
            std_noise = np.random.standard_t(df, size=(size, seq_length, 1))

        ar1_noise = np.matmul(cov_half_mat, std_noise)
        del std_noise
        return cov_mat, ar1_noise

    def create_arq_noise_close_multi(
            self, sigma_s_sq, rho_t, rho_s, seq_length, channel_dim, size,
            normal_bool=True, df=5
    ):
        r"""
        :param sigma_s_sq: positive scalar
        :param rho_t: either 1d-vector or 2d-vector
        :param rho_s: scalar between 0 and 1
        :param seq_length: integer
        :param channel_dim: integer
        :param size:
        :param normal_bool:
        :param df:
        :return: cov_mat based on C_s kronecker C_t
        """
        if len(rho_t) == 1:
            corr_t_mat = self.create_ar1_cov_mat(1, rho_t, seq_length)
        else:
            corr_t_mat, _ = self.create_ar2_pres_mat(1, rho_t, seq_length)
        cov_s_mat = self.create_compound_symmetry_cov_mat(sigma_s_sq, rho_s, channel_dim)
        cov_s_t_mat = np.kron(cov_s_mat, corr_t_mat)
        cov_s_t_half_mat = np.linalg.cholesky(cov_s_t_mat)
        # print('cov_s_t_mat has shape {}'.format(cov_s_t_mat.shape))
        if normal_bool:
            print('The background noise follows multivariate normal distribution.')
            std_noise = np.random.normal(loc=0, scale=1, size=(size, seq_length*channel_dim, 1))
        else:
            print('The background noise follows student-t distribution with df {}'.format(df))
            std_noise = np.random.standard_t(df, size=(size, seq_length*channel_dim, 1))

        noise_transform = np.transpose(np.reshape(
            np.matmul(cov_s_t_half_mat, std_noise),
            [size, channel_dim, seq_length, 1]), axes=(1, 0, 2, 3)
        )
        del std_noise
        return cov_s_t_mat, noise_transform

    def generate_trunc_norm(
            self, mean, cov, c, d, r_underscore, x0,
            n, burn_in_samples=0, thinning=1
    ):
        r"""
        :param mean: (p, 1)
        :param cov: (p, p)
        :param c: lower-bound (p,)
        :param d: upper-bound (p,)
        :param r_underscore: (m, p), restriction matrix c < r_tilta x < d
        :param x0: (length, 1)
        :param n:
        :param burn_in_samples:
        :param thinning:
        :return:
            We follow the thesis entitled 'Efficient Sampling Methods for Truncated Multivariate Normal and
        Student-t Distributions Subject to Linear Inequality Constraints' by Yanfang Li from NCSU.

        1. standardize the TMVN from
            w ~ TMVN(mu, Cov, R_tilta, c, d) to
            x ~ TN(0, I, R, a, b), where
            x = Cov^-1/2 (w - mu), a = c - R_tilta mu, b = d - R_tilta mu,
            R = R_tilta Cov^1/2.
        """
        m, p = r_underscore.shape
        pos_def = self.is_pos_def(cov)
        if not pos_def:
            cov = np.copy(cov + np.eye(p))

        cov_chky = np.linalg.cholesky(cov)  # lower-triangular term
        cov_chky_inv = np.linalg.inv(cov_chky)

        r = np.matmul(r_underscore, cov_chky)
        a = c[:, np.newaxis] - np.matmul(r_underscore, mean)
        b = d[:, np.newaxis] - np.matmul(r_underscore, mean)
        x0 = np.matmul(cov_chky_inv, x0 - mean)
        X = np.zeros([n, p, 1])
        x = np.copy(x0)

        for j in range(-burn_in_samples, n * thinning):
            for i in range(p):
                r_i = r[:, i]
                r_not_i = np.delete(r, [i], axis=-1)
                x_not_i = np.delete(x, [i], axis=0)
                a_minus_rx = np.squeeze(a - np.matmul(r_not_i, x_not_i), axis=-1)
                b_minus_rx = np.squeeze(b - np.matmul(r_not_i, x_not_i), axis=-1)
                j_pos_ind = np.where(r_i > 0)[0]
                j_neg_ind = np.where(r_i < 0)[0]
                n_neg = len(j_neg_ind)
                l_pos = np.max(a_minus_rx[j_pos_ind] / r_i[j_pos_ind])
                u_pos = np.min(b_minus_rx[j_pos_ind] / r_i[j_pos_ind])

                if n_neg > 0:
                    l = np.max(np.array([l_pos, np.max(b_minus_rx[j_neg_ind] / r_i[j_neg_ind])]))
                    u = np.min(np.array([u_pos, np.min(a_minus_rx[j_neg_ind] / r_i[j_neg_ind])]))
                else:
                    l = l_pos; u = u_pos
                if l >= u:
                    x_ji = x0[i, 0]
                    print('cannot find solution x.')
                elif u <= -35:
                    x_ji = u
                elif l >= 35:
                    x_ji = l
                else:
                    rv = stats.truncnorm(l, u, loc=0, scale=1)
                    x_ji = rv.rvs(1)[0]
                x[i, 0] = np.copy(x_ji)
            if j >= 0:
                if thinning == 1:
                    X[j, ...] = np.copy(x)
                elif thinning == 0:
                    X[int(np.floor(j / thinning)), ...] = np.copy(x)
        # return to the original scale
        x = np.matmul(cov_chky, x) + mean
        for i in range(p):
            if x[i] < c[i]:
                x[i] = c[i]
            if x[i] > d[i]:
                x[i] = d[i]
        return x

    @staticmethod
    def adjust_step_size(old_step_size, accept_rate):
        assert 0 <= accept_rate <= 1, print('accept_rate should range between 0 and 1.')
        if 0.3 <= accept_rate <= 0.4:
            new_step_size = old_step_size
        elif accept_rate < 0.25:
            new_step_size = old_step_size * 0.9
        else:
            new_step_size = old_step_size * 1.1
        return new_step_size

    def b_spline_naive(self, x, k, i, t):
        if k == 0:
            return 1.0 if t[i] <= x < t[i + 1] else 0.0
        if t[i + k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i]) / (t[i + k] - t[i]) * self.b_spline_naive(x, k - 1, i, t)
        if t[i + k + 1] == t[i + 1]:
            c2 = 0.0
        else:
            c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * self.b_spline_naive(x, k - 1, i + 1, t)
        return c1 + c2

    def create_design_mat_gen_bayes_seq(self, repetition_dim):
        r"""
        :param repetition_dim: integer
        :return: design_x
        """
        # Create a zero matrix
        dm_row = (repetition_dim*self.num_rep + self.n_multiple - 1) * self.flash_and_pause_length
        dm_col = repetition_dim*self.num_rep*self.n_length
        dm = np.zeros([dm_row, dm_col])
        for trial_id in range(repetition_dim*self.num_rep):
            row_id_low = trial_id * self.flash_and_pause_length
            row_id_upp = row_id_low + self.n_length
            col_id_low = trial_id * self.n_length
            col_id_upp = col_id_low + self.n_length
            dm[row_id_low:row_id_upp, col_id_low:col_id_upp] = self.n_iden
        return dm

    # https://math.unm.edu/~ghuerta/tseries/week8-1.pdf
    @staticmethod
    def arma_generate_sample_fix_randn(
            ar, ma, n_sample, rand_num, scale=1, axis=0, burnin=0
    ):

        if np.ndim(n_sample) == 0:
            n_sample = [n_sample]
        if burnin:
            # handle burin time for nd arrays, maybe there is a better trick in scipy.fft code
            newsize = list(n_sample)
            newsize[axis] += burnin
            newsize = tuple(newsize)
            fslice = [slice(None)] * len(newsize)
            fslice[axis] = slice(burnin, None, None)
            fslice = tuple(fslice)
        else:
            newsize = tuple(n_sample)
            fslice = tuple([slice(None)] * np.ndim(newsize))
        eta = scale * rand_num
        return signal.lfilter(ma, ar, eta, axis=axis)[fslice]

    def generate_arq_noise_minus(
            self, ar, ma, n_sample, rand_num, scale, burnin=0
    ):
        r"""
        :param ar: np.array([1, - coefs])
        :param ma: np.array([1, coefs])
        :param n_sample: positive integer
        :param rand_num: 1d-array_like,
        :param scale: positive float number
        :param burnin: integer
        :return:
        """
        # First generate arq noise with error(t) included
        arq_noise = self.arma_generate_sample_fix_randn(
            ar, ma, n_sample, rand_num, scale, axis=0, burnin=burnin
        )
        arq_noise_minus = arq_noise - scale * rand_num[burnin:]
        return arq_noise, arq_noise_minus

    def create_ratio_mat_inv_half(
            self, r_iter, dt_mat
    ):
        if isinstance(r_iter, float):
            r_iter = np.array([r_iter])
        r_iter = np.tile(r_iter[:, np.newaxis, np.newaxis], [1, self.n_length, 1])
        r_iter = np.concatenate([r_iter, np.ones_like(r_iter)], axis=1)
        r_iter = r_iter[:, np.newaxis, ...]
        ratio_mat = np.matmul(dt_mat, r_iter) ** (-1/2)
        return ratio_mat

    def create_kernel_mat_complex(
            self, scale_param, index_points, kernel_option, **kwargs
    ):
        if kernel_option == 'rbf':
            kappa_theta = self.create_rbf_kernel_fn(scale_param, index_points)

        elif kernel_option == 'gamma_exp':
            gamma = kwargs['gamma']
            kappa_theta = self.create_gamma_exp_kernel_fn(scale_param, gamma, index_points)

        elif kernel_option == 'sine':
            periodicity = kwargs['periodicity']
            kappa_theta = self.create_exp_sine_sq_kernel_fn(scale_param, periodicity, index_points)

        elif kernel_option == 'rbf_+_sine':
            kappa_theta_rbf = self.create_rbf_kernel_fn(scale_param, index_points)
            scale_param_sine = kwargs['scale_sine']
            periodicity = kwargs['periodicity']
            kappa_theta_sine = self.create_exp_sine_sq_kernel_fn(scale_param_sine, periodicity, index_points)
            kappa_theta = kappa_theta_rbf + kappa_theta_sine

        elif kernel_option == 'gamma_exp_+_sine':
            gamma = kwargs['gamma']
            scale_param_sine = kwargs['scale_sine']
            periodicity = kwargs['periodicity']
            kappa_theta_gamma_exp = self.create_gamma_exp_kernel_fn(scale_param, gamma, index_points)
            kappa_theta_sine = self.create_exp_sine_sq_kernel_fn(scale_param_sine, periodicity, index_points)
            kappa_theta = kappa_theta_gamma_exp + kappa_theta_sine

        else:
            kappa_theta = np.eye(len(index_points))
            print('kappa_theta has shape {}'.format(kappa_theta.shape))
            print('Independent Gaussian Kernel.')

        kappa_theta_chol_inv = np.linalg.inv(np.linalg.cholesky(kappa_theta))
        kappa_theta_inv = np.matmul(kappa_theta_chol_inv.T, kappa_theta_chol_inv)

        return kappa_theta, kappa_theta_inv

    def represent_letter_prob(self, letter_prob_vector):
        assert len(letter_prob_vector) == self.letter_table_sum, \
            print('incorrect prob vector input.')

        letter_prob_comb = []
        for i in range(self.letter_table_sum):
            letter_prob_comb.append(
                self.letter_table[i] + '\'s p=' + str(letter_prob_vector[i])
            )
        letter_prob_comb = np.reshape(np.stack(letter_prob_comb), [6, 6])
        return letter_prob_comb

    @staticmethod
    def exp_normalize(x):
        # if x is a multi-dimensional array, we exp-normalize along the rightmost dimension.
        if isinstance(x, np.ndarray):
            b = np.max(x, axis=-1, keepdims=True)
            y = np.exp(x - b)
            return y / np.sum(y, axis=-1, keepdims=True)
        else:
            return '{} is not a numpy array.'.format(x)

    @staticmethod
    def bhattacharyya_gaussian_distance(sample1, sample2):
        r"""
        :param sample1: (batch_num, feature_length)
        :param sample2: (batch_num, feature_length)
        :return:  https://www.wikiwand.com/en/Bhattacharyya_distance

        """
        assert sample1.shape[1] == sample2.shape[1], print('Different feature vector lengths.')
        mean1 = np.mean(sample1, axis=0, keepdims=True)
        cov1 = np.cov(sample1, rowvar=False)
        mean2 = np.mean(sample2, axis=0, keepdims=True)
        cov2 = np.cov(sample2, rowvar=False)

        cov = (1 / 2) * (cov1 + cov2)
        cov_chky_inv = np.linalg.inv(np.linalg.cholesky(cov))
        cov_inv = np.matmul(cov_chky_inv.T, cov_chky_inv)
        stat_1 = np.squeeze((1 / 8) * np.sqrt(np.matmul(np.matmul(mean1 - mean2, cov_inv), (mean1 - mean2).T)))
        stat_2 = (1 / 2) * np.log(
            np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
        )
        return stat_1 + stat_2

    @staticmethod
    def compute_matrix_inv_cholesky(input_matrix):
        input_mat_chky_inv = np.linalg.inv(np.linalg.cholesky(input_matrix))
        if len(input_matrix.shape) == 2:
            input_mat_inv = np.matmul(
                np.transpose(input_mat_chky_inv), input_mat_chky_inv
            )
        else:
            input_mat_inv = np.matmul(
                np.transpose(input_mat_chky_inv, axes=(0, 2, 1)), input_mat_chky_inv
            )
        return input_mat_inv

    def estimate_compound_symmetry_matrix(self, input_matrix):

        n, _ = input_matrix.shape
        diag_val = np.mean(np.diagonal(input_matrix))
        off_diag_val = (np.sum(input_matrix) - n * diag_val) / (n**2 - n)
        print('diag_val = {}, off_diag_val = {}'.format(diag_val, off_diag_val))
        rho = off_diag_val / diag_val
        cs_matrix = self.create_compound_symmetry_cov_mat(diag_val, rho, n)
        return cs_matrix

