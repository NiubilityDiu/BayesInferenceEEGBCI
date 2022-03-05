from self_py_fun.PreFun import *
import self_py_fun.GlobalEEG as gc

show_dim_bool = True
flash_and_pause_length = 40
# bp_low = 0.5
bp_low = float(sys.argv[5])
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

EEGObj = EEGPreFun(
    data_type=gc.DATA_TYPE,
    sub_folder_name=gc.sub_file_name,
    sub_name_short=gc.sub_file_name[:4],
    # EEGGeneralFun class
    num_repetition=gc.NUM_REPETITION,
    num_electrode=gc.NUM_ELECTRODE,
    flash_and_pause_length=flash_and_pause_length,
    n_multiple=gc.N_MULTIPLE,
    num_letter=gc.LETTER_DIM,
    local_bool=gc.local_use
)


# Raw data
[eeg_signals, eeg_code, eeg_type, eeg_pis] = EEGObj.import_eeg_dat(
    eeg_file_suffix, show_dim_bool
)

'''
# Compute spatial dependency structure from raw data
# Down-sampling should not affect spatial dependency relationship
eeg_spatial_corr_est = np.corrcoef(eeg_signals[:1200, :], rowvar=False)
# Save spatial correlation matrix
EEGObj.save_eeg_spatial_corr(eeg_spatial_corr_est, eeg_file_suffix)
'''

# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# X, Y = np.meshgrid(np.arange(16), np.arange(16))
# fig1 = plt.figure(figsize=(12, 10))
# ax1 = fig1.add_axes([left, bottom, width, height])
# Z1 = np.copy(eeg_spatial_corr_est)
# cp1 = plt.contourf(X, Y, Z1)
# fig1.colorbar(cp1)
# ax1.set_title('EEG Signals Spatial Correlation Estimation')
# ax1.set_xlabel('Channel Index')
# ax1.set_ylabel('Channel Index')
# ax1.set_ylim(ax1.get_ylim()[::-1])
# # sample_cov_t_pdf.savefig(fig1)
# plt.show()
# plt.close()


# Truncated data
[eeg_signals_subset, eeg_code_subset, eeg_type_subset] = EEGObj.truncate_raw_sequence(
    eeg_pis, eeg_signals, eeg_code, eeg_type, show_dim_bool
)

# # Save raw truncated signals
# EEGObj.save_truncated_signal(
#     eeg_signals_subset, eeg_code_subset, eeg_type_subset, eeg_file_suffix
# )

# Try down-sampling and see the result
eeg_signals_down_sample = EEGObj.moving_average_decimate(
    eeg_signals_subset,
    gc.DEC_FACTOR,
    show_dim_bool
)

# Save down-sampled signals
EEGObj.save_sample_signal_down(
    eeg_signals_down_sample, eeg_code_subset, eeg_type_subset,
    gc.DEC_FACTOR, 'from_' + eeg_file_suffix
)
