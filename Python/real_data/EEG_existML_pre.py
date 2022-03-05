import self_py_fun.GlobalEEG as gc
from self_py_fun.ExistMLFun import *
sns.set_context('notebook')
reshape_to_1d = False
show_dim_bool = True

bp_low = 0.5
# bp_upp = 6
bp_upp = float(sys.argv[4])
if float(bp_upp) == int(bp_upp):
    bp_upp = int(bp_upp)
else:
    bp_upp = float(bp_upp)

if bp_upp < 0:
    eeg_file_suffix = 'raw'
else:
    eeg_file_suffix = 'raw_bp_{}_{}'.format(bp_low, bp_upp)
eeg_file_suffix_2 = '{}_{}_from_{}'.format(
    gc.file_subscript, gc.DEC_FACTOR, eeg_file_suffix
)

EEGswLDAObj = ExistMLPred(
    data_type=gc.DATA_TYPE,
    sub_folder_name=gc.sub_file_name,
    sub_name_short=gc.sub_file_name[:4],
    # EEGGeneralFun class
    # sampling_rate=gc.sampling_rate,
    num_repetition=gc.NUM_REPETITION,
    num_electrode=gc.NUM_ELECTRODE,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)


# Import the training set without subsetting yet
[eeg_signals, eeg_code, eeg_type] = EEGswLDAObj.import_eeg_processed_dat(eeg_file_suffix_2, reshape_to_1d)
eeg_signals = np.transpose(eeg_signals, [1, 0, 2, 3])

# Produce truncated eeg signals subset
eeg_signals_trun, eeg_type_sub = EEGswLDAObj.create_truncate_segment_batch(
    eeg_signals, eeg_type, gc.LETTER_DIM,
    gc.NUM_REPETITION, show_dim_bool
)

[eeg_signals_trun_t_mean,
 eeg_signals_trun_nt_mean,
 eeg_signals_trun_t_cov,
 eeg_signals_trun_nt_cov] = EEGswLDAObj.produce_trun_mean_cov_subset(
    eeg_signals_trun, eeg_type_sub
)

EEGswLDAObj.produce_mean_covariance_plots(
    eeg_signals_trun_t_mean, eeg_signals_trun_nt_mean,
    None, None, eeg_file_suffix_2, sim_dat=False
)

# For odd/even inference purpose
[eeg_signals_trun_odd, eeg_type_odd, eeg_code_odd,
 eeg_signals_trun_even, eeg_type_even, eeg_code_even] = EEGswLDAObj.split_trunc_train_set_odd_even(
    eeg_signals_trun, eeg_type, eeg_code, gc.rep_odd_id, gc.rep_even_id
)

# Save the entire training sequence and
# extended eeg_type/label for matlab usage.

eeg_signals_trun = np.transpose(eeg_signals_trun, [1, 0, 2])
EEGswLDAObj.save_truncate_signal_1d_real(
    eeg_signals_trun, eeg_type, eeg_code,
    eeg_file_suffix_2,
    gc.NUM_REPETITION, array_3d_bool=False
)
# Save odd/even sequence for matlab use.
EEGswLDAObj.save_truncate_signal_1d_real(
    eeg_signals_trun_odd, eeg_type_odd, eeg_code_odd,
    eeg_file_suffix_2 + '_odd',
    len(gc.rep_odd_id), array_3d_bool=True
)
EEGswLDAObj.save_truncate_signal_1d_real(
    eeg_signals_trun_even, eeg_type_even, eeg_code_even,
    eeg_file_suffix_2 + '_even',
    len(gc.rep_even_id), array_3d_bool=True
)


# Produce the truncated mean curves for training set here
eeg_type_odd_1d = np.reshape(
    eeg_type_odd, [gc.LETTER_DIM * len(gc.rep_odd_id) * gc.NUM_REP]
)
eeg_signals_trun_odd = np.transpose(eeg_signals_trun_odd, [1, 0, 2])
[eeg_signals_odd_trun_t_mean, eeg_signals_odd_trun_nt_mean,
 eeg_signals_odd_trun_t_cov, eeg_signals_odd_trun_nt_cov] = EEGswLDAObj.produce_trun_mean_cov_subset(
    eeg_signals_trun_odd, eeg_type_odd_1d
)

# print(eeg_signals_odd_trun_t_cov[:, :5, :5])

EEGswLDAObj.produce_mean_covariance_plots(
    eeg_signals_odd_trun_t_mean, eeg_signals_odd_trun_nt_mean,
    None, None, eeg_file_suffix_2 + '_odd', sim_dat=False
)

# Produce the truncated mean curves for testing set here
eeg_type_even_1d = np.reshape(
    eeg_type_even, [gc.LETTER_DIM * len(gc.rep_even_id) * gc.NUM_REP]
)
eeg_signals_trun_even = np.transpose(eeg_signals_trun_even, [1, 0, 2])
[eeg_signals_even_trun_t_mean,
 eeg_signals_even_trun_nt_mean,
 eeg_signals_even_trun_t_cov,
 eeg_signals_even_trun_nt_cov] = EEGswLDAObj.produce_trun_mean_cov_subset(
    eeg_signals_trun_even, eeg_type_even_1d
)

EEGswLDAObj.produce_mean_covariance_plots(
    eeg_signals_even_trun_t_mean, eeg_signals_even_trun_nt_mean,
    None, None, eeg_file_suffix_2 + '_even', sim_dat=False
)