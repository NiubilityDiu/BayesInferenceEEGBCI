import sys
sys.path.insert(0, './self_py_fun')
import self_py_fun.GlobalEEG as gc
from self_py_fun.ExistMLFun import *
# from scipy.special import comb
np.random.seed(gc.seed_index)

file_subscript = '{}_{}'.format(gc.file_subscript, gc.DEC_FACTOR)
sim_dat_bool = False
bp_low = 0.5
# bp_upp = 6
bp_upp = float(sys.argv[4])
if float(bp_upp) == int(bp_upp):
    bp_upp = int(bp_upp)
else:
    bp_upp = float(bp_upp)
# zeta_binary = 0.7
zeta_binary = float(sys.argv[5])
letter_dim = gc.LETTER_DIM
repet_num_fit = len(gc.rep_odd_id)
repet_num_pred = len(gc.rep_even_id)
method_name = 'EEGswLDA'

if bp_upp < 0:
    eeg_file_suffix = 'raw'
else:
    eeg_file_suffix = 'raw_bp_{}_{}'.format(bp_low, bp_upp)
eeg_file_suffix_2 = 'ML_{}_{}_from_{}'.format(
    gc.file_subscript, gc.DEC_FACTOR, eeg_file_suffix
)
if bp_upp < 0:
    eeg_file_suffix_3 = 'odd_raw_zeta_{}'.format(zeta_binary)
else:
    eeg_file_suffix_3 = 'odd_raw_bp_0.5_{}_zeta_{}'.format(bp_upp, zeta_binary)

print('We are using channel {} for prediction.'.format(gc.channel_id + 1))
channel_id_str = [str(e+1) for e in gc.channel_id]

if len(gc.channel_id) == 16:
    channel_name = 'all_channels'
    channel_dim = 16
else:
    channel_name = 'channel_' + '_'.join(channel_id_str)
    channel_dim = len(gc.channel_id)

EEGswLDAObj = ExistMLPred(
    data_type=gc.DATA_TYPE,
    sub_folder_name=gc.sub_file_name,
    sub_name_short=gc.sub_file_name[:4],
    # EEGGeneralFun class
    sampling_rate=gc.SAMPLING_RATE,
    num_repetition=gc.NUM_REPETITION,
    num_electrode=channel_dim,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    p300_flash_strength=gc.P300_FLASH_STRENGTH,
    p300_pause_strength=gc.P300_PAUSE_STRENGTH,
    non_p300_strength=gc.NON_P300_STRENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)

# Training set (odd sequences) prediction
[eeg_signals_odd, eeg_code_odd, _] = EEGswLDAObj.import_eeg_odd_even_dat(
    eeg_file_suffix_2 + '_odd'
)

print('eeg_signals_odd have shape {}'.format(eeg_signals_odd.shape))
print('eeg_code_odd has shape {}'.format(eeg_code_odd.shape))

eeg_signals_odd_3d = np.reshape(
    eeg_signals_odd, [letter_dim, repet_num_fit, gc.NUM_REP, gc.NUM_ELECTRODE, gc.N_LENGTH]
)
eeg_signals_odd_exclude = np.reshape(
    eeg_signals_odd_3d[:, 1:, ...],
    [letter_dim*(repet_num_fit-1)*gc.NUM_REP, gc.NUM_ELECTRODE, gc.N_LENGTH]
)
eeg_code_odd_3d = np.reshape(
    eeg_code_odd, [letter_dim, repet_num_fit, gc.NUM_REP]
)
eeg_code_odd_exclude = np.reshape(
    eeg_code_odd_3d[:, 1:, :], [1, letter_dim*(repet_num_fit-1)*gc.NUM_REP]
)

# Testing set (even sequences) prediction
[eeg_signals_even, eeg_code_even, _] = EEGswLDAObj.import_eeg_odd_even_dat(
    eeg_file_suffix_2 + '_even'
)
print('eeg_signals_even have shape {}'.format(eeg_signals_even.shape))
print('eeg_code_even has shape {}'.format(eeg_code_even.shape))

eeg_signals_even_3d = np.reshape(
    eeg_signals_even, [letter_dim, repet_num_pred, gc.NUM_REP, gc.NUM_ELECTRODE, gc.N_LENGTH]
)
eeg_signals_even_exclude = np.reshape(
    eeg_signals_even_3d[:, 1:, ...],
    [letter_dim*(repet_num_pred-1)*gc.NUM_REP, gc.NUM_ELECTRODE, gc.N_LENGTH]
)
eeg_code_even_3d = np.reshape(
    eeg_code_even, [letter_dim, repet_num_pred, gc.NUM_REP]
)
eeg_code_even_exclude = np.reshape(
    eeg_code_even_3d[:, 1:, :], [1, letter_dim*(repet_num_pred-1)*gc.NUM_REP]
)

gc_file_subscript = '{}_{}_from_{}'.format(
    gc.file_subscript, gc.DEC_FACTOR, eeg_file_suffix
)

swlda_wts_i = EEGswLDAObj.import_eeg_matlab_swlda_wts_train(
    file_subscript, channel_name, eeg_file_suffix_3,
    channel_dim=channel_dim, channel_id=None
)

b = swlda_wts_i['b']
inmodel = swlda_wts_i['inmodel']
pval = swlda_wts_i['pval']
# se = swlda_wts_i['se']
# stats = swlda_wts_i['stats']

# Use inmodel as channel selection criterion
inmodel_mat = np.sum(np.reshape(inmodel, [channel_dim, gc.N_LENGTH]), axis=-1)
print(inmodel_mat)
print(np.argsort(inmodel_mat)[::-1] + 1)

# # Use pval as channel selection criterion
# p_val_mat = np.sum(np.reshape(1 * (pval < 0.1), [channel_dim, gc.N_LENGTH]), axis=-1)
# print(p_val_mat)

# Use absolute of weight vector as channe selection criterion
b_abs = np.abs(b)
b_abs_q = np.quantile(b_abs, q=0.9)
# print(b_abs, b_abs_q)
b_abs_binary = np.sum(np.reshape(b_abs >= b_abs_q, [channel_dim, gc.N_LENGTH]), axis=-1)
# print(b_abs_binary)
# print(np.argsort(b_abs_binary)[::-1] + 1)

EEGswLDAObj.plot_swlda_select_feature(
    inmodel,
    'down_{}_{}_{}'.format(gc.DEC_FACTOR, channel_name, eeg_file_suffix_3),
    channel_name, sim_dat_bool, channel_ids=gc.channel_id
)


# predict on odd sequences
eeg_signals_odd_2 = np.reshape(
    eeg_signals_odd[:, gc.channel_id, :],
    [gc.LETTER_DIM * repet_num_fit * gc.NUM_REP, channel_dim * gc.N_LENGTH]
)

swlda_pred_prob_odd = EEGswLDAObj.swlda_predict_y_prob(
    b, inmodel, eeg_signals_odd_2
)

swlda_pred_dict_odd = EEGswLDAObj.ml_predict(
    swlda_pred_prob_odd, eeg_code_odd,
    gc.LETTER_DIM, repet_num_fit
)

EEGswLDAObj.save_exist_ml_pred_results(
    swlda_pred_dict_odd, repet_num_fit, repet_num_fit,  # int(comb(rep_num_fit, 5)),
    gc.TARGET_LETTERS, method_name,
    channel_name, eeg_file_suffix_2 + '_zeta_' + str(zeta_binary),
    train_bool=True, sim_dat_bool=sim_dat_bool
)

# Exclude the first sequence
eeg_signals_odd_2_exclude = np.reshape(
    eeg_signals_odd_exclude[:, gc.channel_id, :],
    [gc.LETTER_DIM * (repet_num_fit-1) * gc.NUM_REP, channel_dim * gc.N_LENGTH]
)

swlda_pred_prob_odd_exclude = EEGswLDAObj.swlda_predict_y_prob(
    b, inmodel, eeg_signals_odd_2_exclude
)

swlda_pred_dict_odd_exclude = EEGswLDAObj.ml_predict(
    swlda_pred_prob_odd_exclude, eeg_code_odd_exclude,
    gc.LETTER_DIM, repet_num_fit-1
)

EEGswLDAObj.save_exist_ml_pred_results(
    swlda_pred_dict_odd_exclude, repet_num_fit, repet_num_fit-1,  # int(comb(rep_num_fit, 5)),
    gc.TARGET_LETTERS, method_name,
    channel_name, eeg_file_suffix_2 + '_zeta_' + str(zeta_binary),
    train_bool=True, sim_dat_bool=sim_dat_bool
)


# predict on even sequences
eeg_signals_even_2 = np.reshape(
    eeg_signals_even[:, gc.channel_id, :],
    [gc.LETTER_DIM * repet_num_pred * gc.NUM_REP, channel_dim * gc.N_LENGTH]
)

swlda_pred_prob_even = EEGswLDAObj.swlda_predict_y_prob(
    b, inmodel, eeg_signals_even_2
)

swlda_pred_dict_even = EEGswLDAObj.ml_predict(
    swlda_pred_prob_even, eeg_code_even,
    gc.LETTER_DIM, repet_num_pred
)

EEGswLDAObj.save_exist_ml_pred_results(
    swlda_pred_dict_even, repet_num_fit, repet_num_pred,  # int(comb(rep_num_pred, 5)),
    gc.TARGET_LETTERS, method_name,
    channel_name, eeg_file_suffix_2 + '_zeta_' + str(zeta_binary),
    train_bool=False, sim_dat_bool=sim_dat_bool
)


# Exclude the first sequence
eeg_signals_even_2_exclude = np.reshape(
    eeg_signals_even_exclude[:, gc.channel_id, :],
    [gc.LETTER_DIM * (repet_num_pred-1) * gc.NUM_REP, channel_dim * gc.N_LENGTH]
)

swlda_pred_prob_even_exclude = EEGswLDAObj.swlda_predict_y_prob(
    b, inmodel, eeg_signals_even_2_exclude
)

swlda_pred_dict_even_exclude = EEGswLDAObj.ml_predict(
    swlda_pred_prob_even_exclude, eeg_code_even_exclude,
    gc.LETTER_DIM, repet_num_pred-1
)

EEGswLDAObj.save_exist_ml_pred_results(
    swlda_pred_dict_even_exclude, repet_num_fit, repet_num_pred-1,  # int(comb(rep_num_pred, 5)),
    gc.TARGET_LETTERS, method_name,
    channel_name, eeg_file_suffix_2 + '_zeta_' + str(zeta_binary),
    train_bool=False, sim_dat_bool=sim_dat_bool
)
