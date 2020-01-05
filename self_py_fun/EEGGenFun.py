from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import scipy.interpolate as ip
from scipy import linalg
import os
import scipy.io as sio
import time
# import seaborn as sns
# from datetime import datetime


class EEGGeneralFun:
    # Global constant
    # 6 by 6 grid table:
    letter_table = ['A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'I', 'J', 'K', 'L',
                    'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X',
                    'Y', 'Z', '1', '2', '3', '4',
                    '5', 'SPEAK', '.', 'BS', '!', '_']

    # 5 by 5 grid table:
    # letter_table = ['A', 'B', 'C', 'D', 'E',
    #                 'F', 'G', 'H', 'I', 'J',
    #                 'K', 'L', 'M', 'N', 'O',
    #                 'P', 'Q', 'R', 'S', 'T',
    #                 'U', 'V', 'W', 'X', '_']

    letter_table_sum = 36
    # letter_table_sum = 25
    num_rep = 12
    # num_rep = 10
    flash_sum = 2
    non_flash_sum = num_rep - flash_sum
    row_set = np.array([1, 2, 3, 4, 5, 6])
    # row_set = np.array([1, 2, 3, 4, 5])
    column_set = np.array([7, 8, 9, 10, 11, 12])
    # column_set = np.array([6, 7, 8, 9, 10])
    row_column_length = 6
    # row_column_length = 5
    VALIDATE_ARGS = True
    DAT_TYPE = 'float32'
    MENSAJE = 'When start from the terminal, ' \
              'type python -m file name without \'.py\' suffix!'
    inter_flash_period = 40  # fixed by experimental design prior to analysis

    # Declare private variable starting with double underscores
    __parent_path_local = '/Users/niubilitydiu/Box Sync/Dissertation/' \
                          'Dataset and Rcode/EEG_MATLAB_data'
    __parent_path_slurm = '/home/mtianwen/EEG_MATLAB_data'

    def __init__(self, num_repetition, num_electrode,
                 flash_and_pause_length, num_letter,
                 sampling_rate=256,
                 p300_flash_strength=1.0,
                 p300_pause_strength=0.0, non_p300_strength=0.0,
                 n_multiple=5,
                 local_bool=True, *args, **kwargs):
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

        # rest period after one complete sequence
        self.rest_period_length = int(n_multiple * flash_and_pause_length)
        # the length of latent Z (p100/p300_eeg_signal)
        self.n_length = int(n_multiple * flash_and_pause_length)
        self.seq_length_std = self.num_repetition * self.flash_and_pause_length * self.num_rep
        self.index_points = np.linspace(start=0., stop=1., num=self.n_length,
                                        endpoint=True, dtype=self.DAT_TYPE)[:, np.newaxis]
        # self.side_zero_row = int(self.n_length / self.flash_and_pause_length) - 1
        self.total_rep = self.num_repetition * self.num_rep
        self.time_n_length = self.n_multiple * self.inter_flash_period / self.sampling_rate * 1000
        self.time_range = np.linspace(
            start=0.0, stop=self.time_n_length, num=self.n_length, dtype=self.DAT_TYPE
        )

        if self.local_bool:
            self.parent_path = self.__parent_path_local
        else:
            self.parent_path = self.__parent_path_slurm

    # Simulation-related Functions:
    def determine_row_column_indices(self, letter):

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

    # Reverse the above process
    def determine_letter(self, row_index, column_index):
        assert 1 <= row_index <= self.row_column_length and self.row_column_length + 1 <= column_index <= self.num_rep
        letter_index = (row_index-1) * self.row_column_length + (column_index - self.row_column_length)
        return self.letter_table[letter_index-1]

    # Create a class-free user-defined function
    # to generate prior knowledge
    # Use convolution to generate simulated signal
    # Refer to the tutorial
    # https://practical-neuroimaging.github.io/on_convolution.html
    # https://www.ijser.org/paper/Wavelet-Transform-use-for-P300-Signal-Clustering-by-Self-Organizing-Map.html

    @staticmethod
    def generate_canonical_eeg_signal(
            x_input, y_input, n_length,
            spline_order, display=False):
        assert len(x_input) == len(y_input)
        x_new = np.linspace(np.min(x_input), np.max(x_input), n_length)
        tck = ip.splrep(x_input, y_input, k=spline_order)
        y_smooth = ip.splev(x_new, tck)

        if display:
            # plt.figure()
            plt.plot(x_new, y_smooth)
            # plt.show()

        return x_new, y_smooth

    # @staticmethod
    # def reduce_precision(tensor, precision_bits=4):
    #     n = 2 ** precision_bits
    #     return tf.round(n * tensor) / n

    def generate_single_code_and_type(self, row_index, column_index, simple_array=True):

        assert 0 < row_index <= int(self.num_rep/2) < column_index <= self.num_rep
        assert 0 < self.flash_length <= self.flash_and_pause_length

        single_code = np.random.permutation(self.num_rep) + 1
        # print('single_permutation is', single_permutation)
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

    def generate_multiple_code_and_type(self, letter=None, simple_array=True):

        multiple_code = []
        multiple_type = []
        if letter is None:
            letter = np.random.choice(self.letter_table)
        row_i, column_i = self.determine_row_column_indices(letter)
        print('letter {}, row index is {}, column index is {}'.format(letter, row_i, column_i))

        for j in range(self.num_repetition):
            single_code, single_type = self.generate_single_code_and_type(
                row_i, column_i, simple_array=simple_array)

            multiple_code.append(single_code)
            multiple_type.append(single_type)
        multiple_code = np.stack(multiple_code, axis=0)
        multiple_type = np.stack(multiple_type, axis=0)

        if simple_array:
            multiple_code = np.reshape(multiple_code, [self.num_rep*self.num_repetition])
            multiple_type = np.reshape(multiple_type, [self.num_rep*self.num_repetition])
        else:
            multiple_code = np.reshape(multiple_code,
                                       [self.num_rep*self.num_repetition*self.flash_and_pause_length])
            multiple_type = np.reshape(multiple_type,
                                       [self.num_rep*self.num_repetition*self.flash_and_pause_length])
            # Add zeros to the end
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

    def generate_multiple_letter_code_and_letter(self, letters, simple_array=True):

        r"""

        :param letters: A list of letters that belongs to the grid defined in the self.letter_table
        :param simple_array: bool, whether we export the arrays with unnecessary zeros.
        :return: two 1d-array including eeg_type and eeg_code
        """

        eeg_code = []
        eeg_type = []
        for _, letter_i in enumerate(letters):
            eeg_code_i, eeg_type_i, _, _, _ = \
                self.generate_multiple_code_and_type(letter_i, simple_array)
            eeg_code.append(eeg_code_i)
            eeg_type.append(eeg_type_i)
        if simple_array:
            eeg_code = np.reshape(np.stack(eeg_code, axis=0),
                                  [self.num_letter*self.num_repetition*self.num_rep])
            eeg_type = np.reshape(np.stack(eeg_type, axis=0),
                                  [self.num_letter*self.num_repetition*self.num_rep])
        else:
            dim_temp = self.num_repetition * self.num_rep * self.flash_and_pause_length
            eeg_code = np.reshape(np.stack(eeg_code, axis=0),
                                  [self.num_letter * (dim_temp + self.rest_period_length)])
            eeg_type = np.reshape(np.stack(eeg_type, axis=0),
                                  [self.num_letter * (dim_temp + self.rest_period_length)])
        # print('eeg_code has shape {}'.format(eeg_code.shape))
        return eeg_code, eeg_type

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

    def write_and_save_letter_prob(
            self, total_log_prob, truncate_index,
            sub_folder_name, subset_index,
            experiment_id, latent_z_type, mcmc_id,
            mcmc_training_index, target_letter
    ):
        if subset_index > 1:
            a_or_w = 'a'
            # file-output.py
            f = open('./Simulation/{}/latent_z_type_{}/experiment_id_{}/mcmc_id_{}/'
                     'letter_{}_prediction.txt'
                     .format(sub_folder_name, latent_z_type,
                             experiment_id, mcmc_id, target_letter), a_or_w)
            if subset_index == mcmc_training_index+1:
                f.write('\n The testing probability is as follows:\n')
            f.write('Use first {} sequences\n'.format(subset_index))

        else:
            a_or_w = 'w'
            # file-output.py
            f = open('./Simulation/{}/latent_z_type_{}/experiment_id_{}/mcmc_id_{}/'
                     'letter_{}_prediction.txt'
                     .format(sub_folder_name, latent_z_type,
                             experiment_id, mcmc_id, target_letter), a_or_w)
            f.write('This is study {} using MCMC samples truncated from {}:\n'
                    .format(experiment_id, truncate_index))
            f.write('Use first {} sequence\n'.format(subset_index))

        [ordered_total_log_prob_, ordered_letter_table] = \
            list(zip(*sorted(zip(total_log_prob, self.letter_table), reverse=True)))

        print('ordered_letter_{}=\n{}'.format(subset_index, ordered_letter_table))
        # print('ordered_log_prob_{}=\n{}'.format(subset_index, ordered_total_log_prob_))

        f.write('ordered_letter=\n{}\n'.format(ordered_letter_table))
        f.write('ordered_log_prob=\n{}\n'.format(ordered_total_log_prob_))
        f.close()

    def create_prior_effect_curve(self, n_length, initial_x, initial_y,
                                  spline_order, display_indicator):
        assert len(initial_x) == len(initial_y)
        # initial_seq_length = len(initial_y)
        final_x, final_y = self.generate_canonical_eeg_signal(
            x_input=initial_x, y_input=initial_y, n_length=n_length,
            spline_order=spline_order, display=display_indicator)
        return final_x, final_y

    @staticmethod
    def create_mexican_hat_curve(a, b, c, d, time_range):
        phi_time = a * (1 - b * (time_range - c)**2) * np.exp(-d*(time_range - c)**2)
        return phi_time

    # Assume evenly distributed time
    def create_group_mexican_hat_curve(self, aa, bb, cc, dd, grp_num, time_range):
        phi_times = [0]
        for i in range(grp_num):
            phi_time = self.create_mexican_hat_curve(aa[i], bb[i], cc[i], dd[i], time_range)
            phi_times = np.concatenate([phi_times, phi_time], axis=0)
        return phi_times[1:]

    # @staticmethod
    # def create_tridiagonal_mat(diag, sub, sup):
    #     n = tf.shape(diag)[0]
    #     r = tf.range(n)
    #     ii = tf.concat([r, r[1:], r[:-1]], axis=0)
    #     jj = tf.concat([r, r[:-1], r[1:]], axis=0)
    #     idx = tf.stack([ii, jj], axis=1)
    #     values = tf.concat([diag, sub, sup], axis=0)
    #     return tf.scatter_nd(idx, values, [n, n])

    @staticmethod
    def create_toeplitz_cov_mat(sigma_sq, first_column):

        r"""

        sigma_sq: scalar value
        first_column: array_like
            1d-array, starting with 1, values within -1 and 1.
        return:
            2d-array with dimension (len(first_column), len(first_column))
        """
        # assert first_column[0] == 1, print('the first entry should be 1!')
        cov_mat = sigma_sq * linalg.toeplitz(first_column)
        return cov_mat

    def create_hetero_toeplitz_cov_mat(self, sigma_vec, first_column):
        r"""

        sigma_vec: 1d-array of length n, must be positive
        first_column: 1d-array, have the same dimension as sigma_vec

        return:
            2d-matrix of (n, n)
        """
        assert len(sigma_vec) == len(first_column), print('Different input vector shapes!')
        cov_mat = np.diag(sigma_vec) @ self.create_toeplitz_cov_mat(1, first_column) @ np.diag(sigma_vec)
        return cov_mat

    def create_compound_symmetry_cov_mat(self, sigma_sq, rho, n):

        r"""

        sigma_sq: scalar
        rho: scalar
        n: integer
        return:
            2d matrix with dimension (n, n)
        """
        first_column = np.concatenate([[1], rho * np.ones([n-1])], axis=0)
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
        cov_mat = np.diag(sigma_vec) @ self.create_compound_symmetry_cov_mat(1, rho, n) @ np.diag(sigma_vec)
        return cov_mat

    def create_ar1_cov_mat(self, sigma_sq, rho, n):
        r"""

        sigma_sq: scalar
        rho: scalar, should be within -1 and 1.
        n: integer

        return:
            2d-matrix of (n,n)
        """
        rho_tile = rho * np.ones([n-1])
        first_column = np.concatenate([[1], np.cumprod(rho_tile)], axis=0)
        cov_mat = self.create_toeplitz_cov_mat(sigma_sq, first_column)
        return cov_mat

    def create_hetero_ar1_cov_mat(self, sigma_vec, rho):

        r"""

        sigma_sq_vec: 1d-array of size n, must be positive
        rho: scalar, between -1 and 1

        return:
            2d-matrix of size (len(sigma_sq_vec), len(sigma_sq_vec))
        """
        n = len(sigma_vec)
        # rho_tile = rho * np.ones([n-1])
        # first_column = np.concatenate([[1], np.cumprod(rho_tile)], axis=0)
        cov_mat = np.diag(sigma_vec) @ self.create_ar1_cov_mat(1, rho, n) @ np.diag(sigma_vec)
        return cov_mat

    def create_gaussian_kernel_fn(self, scale_1, u, ki=None, scale_2=0, display_plot=False):
        r"""
        args:
        -----
            scale_1: positive scalar value to control the smoothness of the kernel
            u: positive integer, the number of basis functions extracted
            ki: positive floating number within the index points,
            scale_2: scalar value, only valid if ki is a floating number
        return:
        -----
            An array of (n, u) where the columns are basis functions

        note:
        -----
            create gaussian kernel with formula:
            exp(-(x-x')**2/(2*scale_1**2)-(x-ki)**2/(2*scale_2**2)-(x'-ki)**2/(2*scale_2**2),

            The larger scale_1 value is, the smoother the GP is (which may result in boundary effect),

            The smaller value, the kernel dense concentrates on ki.
        """
        assert 0 < u <= self.n_length, \
            print('input u is not a positive integer smaller than {}'.format(self.n_length))
        xx, yy = np.meshgrid(self.index_points, self.index_points, sparse=False)
        log_z = -(xx - yy)**2 / (2 * scale_1**2)
        if ki is not None and scale_2 > 0:
            assert 0 <= ki <= 1, print('input ki should be between 0 and 1.')
            log_z += - (xx - ki)**2 / (2 * scale_2**2) - (yy - ki) ** 2 / (2 * scale_2**2)
        z = np.exp(log_z)

        # print('xx has shape {}'.format(xx.shape))
        # print('yy has shape {}'.format(yy.shape))
        # print(xx[0, :])
        # print(yy[0, :])
        # print('z has shape {}'.format(z.shape))

        z_eigen_val, z_eigen_fn = np.linalg.eigh(z)
        # Need to reverse the order of eigenvalue and eigenfunctions
        z_eigen_val_descend = z_eigen_val[::-1]
        z_eigen_fn_descend = z_eigen_fn[:, ::-1]

        if display_plot:
            plt.figure()
            plt.contourf(xx, yy, z, 20, cmap='RdGy')
            # Not sure what color to choose, type a wrong word and python will give me all available selection
            plt.colorbar()
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])

            plt.figure()
            for i in range(self.n_length):
                plt.plot(z[i, :])

            plt.figure()
            plt.plot(z_eigen_val_descend)
            plt.figure()
            for u_id in range(u):
                plt.plot(z_eigen_fn_descend[:, u_id])
            plt.show()

        return z_eigen_val_descend[:u], z_eigen_fn_descend[:, :u]

    @staticmethod
    def compute_hermittan_matrix_inv(input_matrix):
        r"""
        input_matrix: array_like,
            can have batch dimension, but the last two dimensions must be the same and symmetric.

        return:
            array_like, the inverse of input_matrix using Cholesky decomposition to guarantee the stability.
        """
        # Later work to check positive definitiveness
        assert input_matrix.shape[-2] == input_matrix.shape[-1], print('The input matrix is not symmetric')
        input_chky_mat = np.linalg.cholesky(input_matrix)
        input_inv_half_mat = np.linalg.inv(input_chky_mat)

        shape_length = len(input_matrix.shape)
        axes = list(np.arange(shape_length-2))
        axes.extend([shape_length-1, shape_length-2])

        input_inv_half_mat_t = np.transpose(input_inv_half_mat, axes)
        input_inv_mat = input_inv_half_mat_t @ input_inv_half_mat

        return input_inv_mat

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
        id_beta = np.zeros([dim_temp]) - 1
        id_beta[eeg_type == 1] = np.arange(self.flash_sum*letter_dim*repet_dim)
        id_beta[eeg_type != 1] = np.arange(self.flash_sum*letter_dim*repet_dim, dim_temp)

        return id_beta.astype('int32')

    def generate_pseudo_signals(
            self, cov_mat, std_sample_tar, std_sample_ntar,
            mean_fn_tar, mean_fn_ntar, permute_id,
            eeg_code_rs, eeg_type_rs, message, sim_name,
            save_plots
    ):
        r"""
        args:
        -----
            cov_mat:
                array_like, with input shape (num_electrode, n_length, n_length)
            std_sample_tar:
                array_like, with input shape (letter_dim * num_repetition * 2, num_electrode, n_length, 1)
            std_sample_ntar:
                array_like, with input shape (letter_dim * num_repetition * 2, num_electrode, n_length, 1)
            mean_fn_tar:
                array_like, with input shape (num_electrode, n_length, 1)
            mean_fn_ntar:
                array_like, with input shape (num_electrode, n_length, 1)
            permute_id:
                1d-array, with length (letter_dim * num_repetition * num_rep,)
            eeg_code_rs:
                2d-array, with input shape (letter_dim, num_repetition * num_rep)
            eeg_type_rs:
                2d_array, with input shape (letter_dim, num_repetition * num_rep)
            message:
                string
            sim_name:
                string
            save_plots:
                bool

        return:
        -----
            pseudo signals in the direct manner

        note:
        -----
            If num_electrode = 1, the input dimension for num_electrode is collapsed.
        """
        cov_chky = np.linalg.cholesky(cov_mat)
        sample_tar = cov_chky @ std_sample_tar + mean_fn_tar
        sample_ntar = cov_chky @ std_sample_ntar + mean_fn_ntar
        sample = np.concatenate([sample_tar, sample_ntar], axis=0)
        sample = sample[permute_id, ...]
        sample = np.reshape(sample, [self.num_letter,
                                     self.num_repetition*self.num_rep,
                                     self.num_electrode,
                                     self.n_length,
                                     1])
        print('pseudo sample has shape {}'.format(sample.shape))
        self.save_simulation_results(
            sim_name, mean_fn_tar, mean_fn_ntar,
            cov_mat, cov_chky,
            sample, eeg_code_rs, eeg_type_rs,
            convol=False, as_pres=False, save_plots=save_plots,
            message=message
        )

    def save_simulation_results(
            self, sim_folder_name,
            true_delta_tar, true_delta_ntar,
            true_pres_chky_tar, true_pres_chky_ntar,
            pseudo_signals,
            eeg_code=None, eeg_type=None,
            convol=True,
            as_pres=True, save_plots=True,
            message=None
    ):

        r"""
        pseudo_signals: array_like
            should have input shape (letter_dim, rep_dim*num_rep, channel_dim, 25, 1)
        eeg_code: None or array_like
            if array_like, it should have the input shape (letter_dim, rep_dim * num_rep)
        eeg_type: array_like
            should have the same input shape as eeg_code

        """
        if not os.path.exists('{}/SIM_files/{}'.format(self.parent_path, sim_folder_name)):
            os.mkdir('{}/SIM_files/{}'.format(self.parent_path, sim_folder_name))

        if save_plots:
            plot_pdf = bpdf.PdfPages('{}/SIM_files/{}/pseudo_signal_plots.pdf'
                                     .format(self.parent_path, sim_folder_name))
            if convol:
                for i in range(self.num_letter):
                    # eeg_type_expand = np.zeros_like(np.squeeze(pseudo_signals, axis=(1, 3)))
                    eeg_type_expand = np.zeros_like(pseudo_signals[:, 0, :, 0])
                    target_ids = np.where(eeg_type[i, :] == 1)[0] * self.flash_and_pause_length
                    # print('target_ids have shape {}'.format(target_ids.shape))
                    eeg_type_expand[i, target_ids] = 1
                    # print('eeg_type_expand has shape {}'.format(eeg_type_expand.shape))
                    for j in range(self.num_electrode):
                        fig = plt.figure(figsize=(14, 12))
                        plt.plot(pseudo_signals[i, j, :, 0], label="Convol signals")
                        plt.plot(eeg_type_expand[i, :], label="Target")
                        plt.legend(loc="upper right")
                        plt.title('{}_convol_letter_{}_chan_{}'.format(message, i+1, j+1))
                        plot_pdf.savefig(fig)
                        plt.close()
                plot_pdf.close()

            else:
                time_index = np.arange(self.n_length)
                max_val = np.max(pseudo_signals)
                min_val = np.min(pseudo_signals)
                for i in range(self.num_letter):
                    for j in range(self.num_rep * self.num_repetition):
                        for k in range(self.num_electrode):
                            fig = plt.figure(figsize=(7, 6))
                            plt.plot(time_index, pseudo_signals[i, j, k, :, 0])
                            plt.title('{}_direct_letter_{}_type_{}_code_{}_chan_{}'
                                      .format(message, i+1, eeg_type[i, j], eeg_code[i, j], k+1))
                            plt.xlabel('Time (# obs)')
                            plt.ylabel('Signal Amplitude')
                            plt.ylim(min_val, max_val)
                            # plt.show()
                            plot_pdf.savefig(fig)
                            plt.close()
                plot_pdf.close()

        if as_pres:
            sio.savemat('{}/SIM_files/{}/sim_dat.mat'.
                        format(self.parent_path, sim_folder_name),
                        {
                            'delta_tar': true_delta_tar,
                            'delta_ntar': true_delta_ntar,
                            'pres_chky_tar': true_pres_chky_tar,
                            'pres_chky_ntar': true_pres_chky_ntar,
                            'signals': pseudo_signals,
                            'eeg_code': eeg_code,
                            'eeg_type': eeg_type
                        })
        else:
            sio.savemat('{}/SIM_files/{}/sim_dat.mat'.
                        format(self.parent_path, sim_folder_name),
                        {
                            'delta_tar': true_delta_tar,
                            'delta_ntar': true_delta_ntar,
                            'cov_tar': true_pres_chky_tar,
                            'cov_ntar': true_pres_chky_ntar,
                            'signals': pseudo_signals,
                            'eeg_code': eeg_code,
                            'eeg_type': eeg_type,
                            'message': message
                        })

    def import_simulation_results(self, sim_folder_name, reshape_to_3d=False, as_pres=True):

        sim_dat = sio.loadmat('{}/SIM_files/{}/sim_dat.mat'.format(self.parent_path, sim_folder_name))
        sim_keys, _ = zip(*sim_dat.items())

        delta_tar = sim_dat['delta_tar']
        delta_ntar = sim_dat['delta_ntar']
        signals = sim_dat['signals']
        eeg_code = sim_dat['eeg_code']
        eeg_type = sim_dat['eeg_type']
        # print('delta_tar has shape {}'.format(delta_tar.shape))
        # print('delta_ntar has shape {}'.format(delta_ntar.shape))
        # print('signals have shape {}'.format(signals.shape))
        # print('eeg_code has shape {}'.format(eeg_code.shape))
        # print('eeg_type has shape {}'.format(eeg_type.shape))

        if as_pres:
            cov_tar = sim_dat['pres_chky_tar']
            cov_ntar = sim_dat['pres_chky_ntar']
            message = ''
        else:
            cov_tar = sim_dat['cov_tar']
            cov_ntar = sim_dat['cov_ntar']
            message = sim_dat['message'][0]

        if reshape_to_3d:
            eeg_code = np.reshape(eeg_code, [self.num_letter, self.num_repetition, self.num_rep])
            # eeg_code = eeg_code[:letter_dim, :trn_repetition, :]
            eeg_type = np.reshape(eeg_type, [self.num_letter, self.num_repetition, self.num_rep])
            # eeg_type = eeg_type[:letter_dim, :trn_repetition, :]

        # print('signals have shape {}'.format(signals.shape))
        # signals = np.transpose(signals, [0, 2, 1])

        return delta_tar, delta_ntar, \
               cov_tar, cov_ntar, \
               signals, eeg_code, eeg_type,\
               message

