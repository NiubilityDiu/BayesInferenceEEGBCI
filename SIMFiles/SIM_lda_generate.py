import sys
sys.path.insert(0, './self_py_fun')
# from self_py_fun.swLDAFun import *
from self_py_fun.xDAFun import *
import self_py_fun.SIMConvolGlobal as sg
# import self_py_fun.EEGConvolGlobal as gc
print(os.getcwd())
np.random.seed(612)

sim_name = 'sim_'
name_ids = 2020010400 + np.array([1, 2, 3, 4, 5]) + 25
sigma_val = np.array([1, 10, 20, 30, 40])
rho = 0.8
display_plot = True  # for pseudo mean fn
save_plot = False  # for final pseudo signals
mean_fn_scale = 1.0

PData = EEGGeneralFun(
    num_repetition=sg.num_repetition, num_electrode=sg.num_electrode_generate,
    flash_and_pause_length=sg.flash_and_pause_length,
    num_letter=sg.letter_dim,
    n_multiple=sg.n_multiple,
    local_bool=sg.local_use
)

x_input = np.array([0, 5, 10, 15, 24])
y_input_similar_1 = np.array([0, 0, 4, 0, 0]) / mean_fn_scale
y_input_similar_0 = np.array([0, 0, 1, 0, 0]) / mean_fn_scale
plt.figure()
_, similar_mean_fn_tar = PData.generate_canonical_eeg_signal(
    x_input, y_input_similar_1, sg.n_length, 1, display_plot)
_, similar_mean_fn_ntar = PData.generate_canonical_eeg_signal(
    x_input, y_input_similar_0, sg.n_length, 1, display_plot)
similar_mean_fn_tar = similar_mean_fn_tar[:, np.newaxis]
similar_mean_fn_ntar = similar_mean_fn_ntar[:, np.newaxis]
plt.title('similar_mean_fn')

'''
# Generate pseudo letter, pseudo code, and pseudo type
# Use the eeg_code, eeg_type from K151
# eeg_code, eeg_type = PData.generate_multiple_letter_code_and_letter(sg.letters, True)
# Import the training set without subsetting yet
EEGObj = EEGPreFun(
    data_type=gc.data_type,
    sub_folder_name=gc.sub_file_name,
    # EEGGeneralFun class
    num_repetition=gc.num_repetition,
    num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    num_letter=gc.num_letter,
    n_multiple=gc.n_multiple
)

[eeg_signals,
 eeg_code_real_data,
 eeg_type_real_data] = EEGObj.import_eeg_processed_dat(
    gc.file_subscript, reshape_to_1d=False)
# Produce truncated eeg signals subset and all descriptive statistics by the entire sequence
eeg_signals_trun_sub, eeg_type_real_data_sub = EEGObj.create_truncate_segment_batch(
    np.squeeze(eeg_signals, axis=-1), eeg_type_real_data, letter_dim=gc.letter_dim,
    trn_repetition=gc.num_repetition)

print('eeg_signals has shape {}'.format(eeg_signals.shape))
print('eeg_signals_trun_sub has shape {}'.format(eeg_signals_trun_sub.shape))
print('eeg_type_real_data_sub has shape {}'.format(eeg_type_real_data_sub.shape))

# start_time = time.time()
[eeg_signals_trun_t_mean,
 eeg_signals_trun_nt_mean,
 _, _] = EEGObj.produce_trun_mean_cov_subset(
    eeg_signals_trun_sub, eeg_type_real_data_sub)
channel_id = [5, 11]  # Channel 12
eeg_type_real_data_1d = np.reshape(eeg_type_real_data, [sg.letter_dim * sg.num_repetition * sg.num_rep])
permute_id = PData.create_permute_beta_id(sg.letter_dim, sg.num_repetition, eeg_type_real_data_1d)

eeg_code_2d = np.reshape(eeg_code_real_data, [sg.letter_dim, sg.total_stm_num])
eeg_type_2d = np.reshape(eeg_type_real_data, [sg.letter_dim, sg.total_stm_num])
'''

[eeg_code, eeg_type] = PData.generate_multiple_letter_code_and_letter(sg.letters)
eeg_code_2d = np.reshape(eeg_code, [sg.letter_dim, sg.num_repetition * sg.num_rep])
eeg_type_2d = np.reshape(eeg_type, [sg.letter_dim, sg.num_repetition * sg.num_rep])
permute_id = PData.create_permute_beta_id(sg.letter_dim, sg.num_repetition, eeg_type)

print('eeg_code_2d has shape {}'.format(eeg_code_2d.shape))
print('eeg_type_2d has shape {}'.format(eeg_type_2d.shape))
print('permute_id has shape {}'.format(permute_id.shape))


'''
plt.figure()
# x_input = np.array([0, 2, 5, 7, 11, 15, 20, 24])
y_input_sharp_neg_1 = np.array([0, 0, -1, 0, 0, 5, 0, 0]) / mean_fn_scale
y_input_sharp_neg_0 = np.array([0, 0, -1, 0, 0, 1, 0, 0]) / mean_fn_scale
_, n1_similar_mean_fn_tar = PData.generate_canonical_eeg_signal(
    x_input_neg, y_input_sharp_neg_1, sg.n_length, 1, display_plot)
_, n1_similar_mean_fn_ntar = PData.generate_canonical_eeg_signal(
    x_input_neg, y_input_sharp_neg_0, sg.n_length, 1, display_plot)
n1_similar_mean_fn_tar = n1_similar_mean_fn_tar[:, np.newaxis]
n1_similar_mean_fn_ntar = n1_similar_mean_fn_ntar[:, np.newaxis]
plt.title('n1_similar_mean_fn')

# Start from standardized normal
std_sample_tar = np.random.multivariate_normal(
    mean=np.zeros([sg.n_length]),
    cov=np.eye(sg.n_length),
    size=2 * sg.letter_dim * sg.num_repetition
)[..., np.newaxis]
std_sample_ntar = np.random.multivariate_normal(
    mean=np.zeros([sg.n_length]),
    cov=np.eye(sg.n_length),
    size=10 * sg.letter_dim * sg.num_repetition
)[..., np.newaxis]

print('std_sample_tar has shape {}'.format(std_sample_tar.shape))
print('std_sample_ntar has shape {}'.format(std_sample_ntar.shape))
'''

# The following are based on multiple channels.
# Start from standardized normal
std_sample_tar = np.stack(
    [np.random.multivariate_normal(
        mean=np.zeros([sg.n_length]),
        cov=np.eye(sg.n_length),
        size=2 * sg.letter_dim * sg.num_repetition)[..., np.newaxis]
     for i in range(sg.num_electrode_generate)], axis=1)

std_sample_ntar = np.stack(
    [np.random.multivariate_normal(
        mean=np.zeros([sg.n_length]),
        cov=np.eye(sg.n_length),
        size=10 * sg.letter_dim * sg.num_repetition)[..., np.newaxis]
     for j in range(sg.num_electrode_generate)], axis=1)

print('std_sample_tar has shape {}'.format(std_sample_tar.shape))
print('std_sample_ntar has shape {}'.format(std_sample_ntar.shape))

similar_mean_fn_tar = np.tile(similar_mean_fn_tar[np.newaxis, ...], [sg.num_electrode_generate, 1, 1])
similar_mean_fn_ntar = np.tile(similar_mean_fn_ntar[np.newaxis, ...], [sg.num_electrode_generate, 1, 1])

print('similar_mean_fn_tar has shape {}'.format(similar_mean_fn_tar.shape))
print('similar_mean_fn_ntar has shape {}'.format(similar_mean_fn_ntar.shape))

'''
# 1. independent covariance matrix and sharp mean functions
cov_mat_indep = sigma_val * np.eye(sg.n_length)
cov_chky_indep = np.linalg.cholesky(cov_mat_indep)
message_1 = 'indep_cov_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_indep, std_sample_tar, std_sample_ntar,
    sharp_mean_fn_tar, sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_1, sim_name + '1', save_plot
)

# 2. compound symmetric covariance matrix and sharp mean functions
cov_mat_cs = PData.create_compound_symmetry_cov_mat(sigma_val, rho, sg.n_length)
cov_chky_cs = np.linalg.cholesky(cov_mat_cs)
message_2 = 'cs_cov_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_cs, std_sample_tar, std_sample_ntar,
    sharp_mean_fn_tar, sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_2, sim_name + '2', save_plot
)
'''

for i in range(5):
    cov_mat_ar1 = PData.create_ar1_cov_mat(sigma_val[i], rho, sg.n_length)
    message = 'ar1_cov_s_sq_{}_similar_mean_fn'.format(sigma_val[i])
    PData.generate_pseudo_signals(
        cov_mat_ar1, std_sample_tar, std_sample_ntar,
        similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
        eeg_code_2d, eeg_type_2d, message, sim_name + str(name_ids[i]), save_plot
    )

'''
# 1. ar(1) covariance matrix and sharp mean functions
cov_mat_ar1_1 = PData.create_ar1_cov_mat(sigma_val[0], rho, sg.n_length)
message_1 = 'ar1_cov_s_sq_1_similar_mean_fn_1'
PData.generate_pseudo_signals(
    cov_mat_ar1_1, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_1, sim_name + str(name_ids[0]), save_plot
)

# 10. ar(1) covariance matrix and sharp mean functions
cov_mat_ar1_10 = PData.create_ar1_cov_mat(sigma_val[1], rho, sg.n_length)
message_10 = 'ar1_cov_s_sq_10_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_10, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_10, sim_name + str(name_ids[1]), save_plot
)

# 20. ar(1) covariance matrix with sharp mean functions
cov_mat_ar1_20 = PData.create_ar1_cov_mat(sigma_val[2], rho, sg.n_length)
message_20 = 'ar1_s_sq_20_cov_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_20, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_20, sim_name + str(name_ids[2]), save_plot
)

# 30. ar(1) covariance matrix with sharp mean functions
cov_mat_ar1_30 = PData.create_ar1_cov_mat(sigma_val[3], rho, sg.n_length)
message_30 = 'ar1_s_sq_30_cov_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_30, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_30, sim_name + str(name_ids[3]), save_plot
)

# 40. ar(1) covariance matrix with sharp mean functions
cov_mat_ar1_40 = PData.create_ar1_cov_mat(sigma_val[4], rho, sg.n_length)
message_40 = 'ar1_s_sq_40_cov_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_40, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_40, sim_name + str(name_ids[4]), save_plot
)

# 4. heterogeneous ar(1) covariance matrix and sharp mean functions
sigma_vec_short = np.array([1.0, 1.2, 2, 1.2, 1.0])
_, sigma_vec = PData.generate_canonical_eeg_signal(x_input, sigma_vec_short, sg.n_length, 1, False)
cov_mat_heter_ar1 = PData.create_hetero_ar1_cov_mat(sigma_vec, rho)
cov_chky_heter_ar1 = np.linalg.cholesky(cov_mat_heter_ar1)
message_4 = 'heter_ar1_cov_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_heter_ar1, std_sample_tar, std_sample_ntar,
    sharp_mean_fn_tar, sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_4, sim_name + '4', save_plot
)

# 5. toeplitz covariance matrix and sharp mean functions
# first column, correlation matrix, start with 1
sigma_vec_short_2 = np.array([1.0, 0.4, 0.7, 0.4, 0.2])
_, sigma_vec_2 = PData.generate_canonical_eeg_signal(x_input, sigma_vec_short_2, sg.n_length, 1, False)
cov_mat_toeplitz = PData.create_toeplitz_cov_mat(sigma_val, sigma_vec_2)
cov_chky_toeplitz = np.linalg.cholesky(cov_mat_toeplitz)
message_5 = 'toeplitz_cov_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_toeplitz, std_sample_tar, std_sample_ntar,
    sharp_mean_fn_tar, sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_5, sim_name + '5', save_plot
)

# 6. use it from real subject K151, channel 12
cov_mat_real_data = np.cov(eeg_signals_trun_sub[:, channel_id[0], :], rowvar=False)
cov_chky_real_data = np.linalg.cholesky(cov_mat_real_data)
message_6 = 'k151_chan_12_cov_and_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_real_data, std_sample_tar, std_sample_ntar,
    sharp_mean_fn_tar, sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_6, sim_name + '6', save_plot
)

# 26. use it from real subject k151, channel 6 and 12 combined
cov_mat_real_data_2 = np.stack([np.cov(eeg_signals_trun_sub[:, channel_id[i], :], rowvar=False)
                                for i in range(sg.num_electrode_generate)], axis=0)
cov_chky_real_data_2 = np.linalg.cholesky(cov_mat_real_data_2)
message_26 = 'K151_chan_6_12_cov_and_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_real_data_2, std_sample_tar_2, std_sample_ntar_2,
    sharp_mean_fn_tar_2, sharp_mean_fn_ntar_2, permute_id,
    eeg_code_2d, eeg_type_2d, message_26, sim_name + '26', save_plot
)

# 230. ar(1) covariance matrix and sharp mean functions
cov_mat_ar1_230 = PData.create_ar1_cov_mat(sigma_val[1], rho, sg.n_length)
message_230 = 'ar1_cov_s_sq_10_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_230, std_sample_tar_2, std_sample_ntar_2,
    sharp_mean_fn_tar_2, sharp_mean_fn_ntar_2, permute_id,
    eeg_code_2d, eeg_type_2d, message_230, sim_name + '230', save_plot
)

# 2300. ar(1) covariance matrix with sharp mean functions
cov_mat_ar1_2300 = PData.create_ar1_cov_mat(sigma_val[2], rho, sg.n_length)
message_2300 = 'ar1_s_sq_100_cov_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_2300, std_sample_tar_2, std_sample_ntar_2,
    sharp_mean_fn_tar_2, sharp_mean_fn_ntar_2, permute_id,
    eeg_code_2d, eeg_type_2d, message_2300, sim_name + '2300', save_plot
)

# 2600. ar(1) covariance matrix with sharp mean functions
cov_mat_ar1_2600 = PData.create_ar1_cov_mat(sigma_val[3], rho, sg.n_length)
message_2600 = 'ar1_s_sq_200_cov_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1_2600, std_sample_tar_2, std_sample_ntar_2,
    sharp_mean_fn_tar_2, sharp_mean_fn_ntar_2, permute_id,
    eeg_code_2d, eeg_type_2d, message_2600, sim_name + '2600', save_plot
)

# 7. independent covariance matrix and similar mean fn
# sample_tar_7 = cov_chky_real_data @ std_sample_tar + sharp_mean_fn_tar
# sample_ntar_7 = cov_chky_real_data @ std_sample_ntar + sharp_mean_fn_ntar
# sample_7 = np.concatenate([sample_tar_7, sample_ntar_7], axis=0)
# sample_7 = sample_7[permute_id, ...]
# print('sample_7 has shape {}'.format(sample_7.shape))
message_7 = 'indep_cov_and_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_indep, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_7, sim_name + '7', save_plot
)

# 8. cs cov matrix and similar mean fn
message_8 = 'cs_cov_mat_and_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_cs, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_8, sim_name + '8', save_plot
)

# 9. ar(1) cov matrix and similar mean fn
message_9 = 'ar1_cov_and_similar_data_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_9, sim_name + '9', save_plot
)

# 10. hetero ar(1) cov matrix and similar mean fn
message_10 = 'heter_ar1_cov_and_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_heter_ar1, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_10, sim_name + '10', save_plot
)

# 11. toeplitz cov mat and similar mean fn
message_11 = 'toeplitz_cov_and_similar_data_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_toeplitz, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_11, sim_name + '11', save_plot
)

# 12. K151 channel 12 cov mat and similar mean fn
message_12 = 'real_data_cov_and_similar_data_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_real_data, std_sample_tar, std_sample_ntar,
    similar_mean_fn_tar, similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_12, sim_name + '12', save_plot
)

# We need to change heter-AR(1) and Toeplitz here
sigma_vec_short_neg = np.array([1.0, 1.2, 2, 1.2, 1.0, 4, 2.0, 1.0])
_, sigma_vec_neg = PData.generate_canonical_eeg_signal(x_input_neg, sigma_vec_short_neg, sg.n_length, 1, False)
cov_mat_heter_ar1_neg = PData.create_hetero_ar1_cov_mat(sigma_vec_neg, rho)
cov_chky_heter_ar1_neg = np.linalg.cholesky(cov_mat_heter_ar1_neg)

sigma_vec_short_2_neg = np.array([1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.2, 0])
_, sigma_vec_2_neg = PData.generate_canonical_eeg_signal(x_input_neg, sigma_vec_short_2_neg, sg.n_length, 1, False)
cov_mat_toeplitz_neg = PData.create_toeplitz_cov_mat(sigma_val, sigma_vec_2_neg)
cov_chky_toeplitz_neg = np.linalg.cholesky(cov_mat_toeplitz_neg)


# 13. indep cov mat with n1_sharp_mean_fn
message_13 = 'indep_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_indep, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_13, sim_name + '13', save_plot
)

# 14. cs cov mat with n1_sharp_mean_fn
message_14 = 'cs_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_cs, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_14, sim_name + '14', save_plot
)


# 15. ar(1) cov mat with n1_sharp_mean_fn
cov_mat_ar1 = PData.create_ar1_cov_mat(sigma_val[0], rho, sg.n_length)
message_15 = 'ar1_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_15, sim_name + '15', save_plot
)

# 15. ar(1) cov mat with n1_sharp_mean_fn
cov_mat_ar1 = PData.create_ar1_cov_mat(sigma_val[1], rho, sg.n_length)
message_150 = 'ar1_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_150, sim_name + '150', save_plot
)


# 16. hetero ar(1) cov mat with n1_sharp_mean_fn
message_16 = 'heter_ar1_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_heter_ar1_neg, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_16, sim_name + '16', save_plot
)

# 17. toeplitz cov mat with n1_sharp_mean_fn
message_17 = 'toeplitz_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_toeplitz_neg, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_17, sim_name + '17', save_plot
)

# 18. k151 channel 12 cov mat with n1_sharp_mean_fn
message_18 = 'real_data_cov_and_n1_sharp_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_real_data, std_sample_tar, std_sample_ntar,
    n1_sharp_mean_fn_tar, n1_sharp_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_18, sim_name + '18', save_plot
)

# 19. indep cov mat with n1_similar_mean_fn
message_19 = 'indep_cov_and_n1_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_indep, std_sample_tar, std_sample_ntar,
    n1_similar_mean_fn_tar, n1_similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_19, sim_name + '19', save_plot
)

# 20. cs cov mat with n1_similar_mean_fn
message_20 = 'cs_cov_and_n1_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_cs, std_sample_tar, std_sample_ntar,
    n1_similar_mean_fn_tar, n1_similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_20, sim_name + '20', save_plot
)

# 21. ar(1) cov mat with n1_similar_mean_fn
message_21 = 'ar1_cov_and_n1_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_ar1, std_sample_tar, std_sample_ntar,
    n1_similar_mean_fn_tar, n1_similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_21, sim_name + '21', save_plot
)

# 22. hetero ar(1) cov mat with n1_similar_mean_fn
message_22 = 'heter_ar1_cov_and_n1_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_heter_ar1_neg, std_sample_tar, std_sample_ntar,
    n1_similar_mean_fn_tar, n1_similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_22, sim_name + '22', save_plot
)

# 23. toeplitz cov mat with n1_similar_mean_fn
message_23 = 'toeplitz_cov_and_n1_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_toeplitz_neg, std_sample_tar, std_sample_ntar,
    n1_similar_mean_fn_tar, n1_similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_23, sim_name + '23', save_plot
)

# 24. K151 channel 12 cov mat with n1_similar_mean_fn
message_24 = 'real_data_cov_and_n1_similar_mean_fn'
PData.generate_pseudo_signals(
    cov_mat_real_data, std_sample_tar, std_sample_ntar,
    n1_similar_mean_fn_tar, n1_similar_mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, message_24, sim_name + '24', save_plot
)
'''