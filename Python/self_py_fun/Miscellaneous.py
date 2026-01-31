import numpy as np
from scipy import linalg


def return_cumulative_array_in_list(input_array):
    input_len = len(input_array)
    output_ls = []
    for e in range(input_len):
        output_ls.append(input_array[:e+1])

    return output_ls


def misc_create_toeplitz_cov_mat(sigma_sq, first_column_except_1):

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


def misc_create_compound_symmetry_cov_mat(diagonal_val, off_diagonal_val, n):

    r"""
    diagonal_val: scalar
    off_diagonal_val: scalar
    n: integer
    return:
        2d matrix with dimension (n, n)
    """
    rho = off_diagonal_val / diagonal_val
    first_column = rho * np.ones([n - 1])
    cov_mat = misc_create_toeplitz_cov_mat(diagonal_val, first_column)
    return cov_mat
