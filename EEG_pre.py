import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.EEGPreFun import *
import self_py_fun.EEGConvolGlobal as gc
print(os.getcwd())
dec_factor = 8
print('The pre-work is for Subject {}!\n'.format(gc.sub_file_name))
print('Set gc.file_subscript to \'raw_trun\' for data pre-processing!')

EEGObj = EEGPreFun(
    data_type=gc.data_type,
    sub_folder_name=gc.sub_file_name,
    # EEGGeneralFun class
    num_repetition=gc.num_repetition,
    num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    num_letter=gc.num_letter)

# Raw data
[eeg_signals, eeg_code, eeg_type, eeg_pis] = EEGObj.import_eeg_dat()
# print('eeg_signals have shape {}'.format(eeg_signals.shape))

# Truncated data
[eeg_signals_subset, eeg_code_subset,
 eeg_type_subset] = EEGObj.truncate_raw_sequence(
    eeg_pis=eeg_pis, eeg_signal=eeg_signals, eeg_code=eeg_code, eeg_type=eeg_type
)
# print('eeg_signals_subset has shape {}'.format(eeg_signals_subset.shape))

# Save raw truncated signals
EEGObj.save_truncated_signal(
    eeg_signals_subset, eeg_code_subset, eeg_type_subset
)
# Try down-sampling and see the result
eeg_signals_down_sample = EEGObj.moving_average_decimate(
    eeg_signals_subset=eeg_signals_subset,
    eeg_code_subset=eeg_code_subset,
    eeg_type_subset=eeg_type_subset,
    dec_factor=dec_factor
)
print('eeg_signals_down_sample has shape {}'.format(eeg_signals_down_sample.shape))
# Save down-sampled signals
EEGObj.save_down_sample_signal(
    eeg_signals_down_sample, eeg_code_subset, eeg_type_subset
)
