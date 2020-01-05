import sys
sys.path.insert(0, './self_py_fun')
import self_py_fun.EEGConvolGlobal as gc
from self_py_fun.swLDAFun import *
# print(os.getcwd())
# tf.compat.v1.random.set_random_seed(612)
np.random.seed(612)

EEGswLDAObj = SWLDAPred(
    data_type=gc.data_type,
    sub_folder_name=gc.sub_file_name,
    # EEGGeneralFun class
    sampling_rate=gc.sampling_rate,
    num_repetition=gc.num_repetition,
    num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    p300_flash_strength=gc.p300_flash_strength,
    p300_pause_strength=gc.p300_pause_strength,
    non_p300_strength=gc.non_p300_strength,
    num_letter=gc.num_letter,
    n_multiple=gc.n_multiple)
EEGswLDAObj.print_sub_trn_info(gc.num_repetition)

# Import the training set without subsetting yet
[eeg_signals, eeg_code,
 eeg_type] = EEGswLDAObj.import_eeg_processed_dat(gc.file_subscript, reshape_to_1d=False)

# Produce truncated eeg signals subset
eeg_signals_trun, eeg_type_sub = EEGswLDAObj.create_truncate_segment_batch(
    np.squeeze(eeg_signals, axis=-1), eeg_type, letter_dim=gc.num_letter,
    trn_repetition=gc.num_repetition)

print('eeg_signal has shape {}'.format(eeg_signals.shape))
print('eeg_signals_trun has shape {}'.format(eeg_signals_trun.shape))
print('eeg_type_sub has shape {}'.format(eeg_type_sub.shape))
print('eeg_signals_trun has sigma_sq {}'.format(np.var(eeg_signals_trun, axis=(0, 2))))

[eeg_signals_trun_t_mean,
 eeg_signals_trun_nt_mean,
 eeg_signals_trun_t_cov,
 eeg_signals_trun_nt_cov] = EEGswLDAObj.produce_trun_mean_cov_subset(
    eeg_signals_trun, eeg_type_sub)

eeg_signals_trun_mean = np.mean(eeg_signals_trun, axis=0)
'''
# for i in range(gc.num_electrode):
#     plt.figure()
#     plt.plot(eeg_signals_trun_t_mean[i, :], label="target-mean")
#     plt.plot(eeg_signals_trun_nt_mean[i, :], label="non-target-mean")
#     plt.plot(eeg_signals_trun_mean[i, :], label="all-mean")
#     plt.legend(loc="upper right")
#     plt.title('chan-'+str(i+1))

t_mean_std = eeg_signals_trun_t_mean - eeg_signals_trun_mean
nt_mean_std = eeg_signals_trun_nt_mean - eeg_signals_trun_mean

EEGswLDAObj.produce_mean_covariance_plots(
    t_mean_std, nt_mean_std, None, None, 'sample', gc.file_subscript
)
'''
# Save the entire training sequence and
# extended eeg_type/label for matlab usage.
'''
EEGswLDAObj.save_trun_signal_1d_label(
    eeg_signals_trun, eeg_type_sub, gc.file_subscript)
'''

