import self_py_fun.GlobalEEG as gc
from self_py_fun.ExistMLFun import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
sns.set_context('notebook')

# Global constants
# reshape_1d_bool = True
rep_num_fit = len(gc.rep_odd_id)
rep_num_pred = len(gc.rep_even_id)
# file_subscript = 'ML_down'
sim_dat_bool = False
bp_low = 0.5
# bp_upp = 6
bp_upp = float(sys.argv[4])
if float(bp_upp) == int(bp_upp):
    bp_upp = int(bp_upp)
else:
    bp_upp = float(bp_upp)
# method_name = 'BAG'
method_name = sys.argv[5]
# logistic regression, support vector machine, random forest, bagging, and ada boost

if bp_upp < 0:
    eeg_file_suffix = 'raw'
else:
    eeg_file_suffix = 'raw_bp_{}_{}'.format(bp_low, bp_upp)
file_subscript = 'ML_down_{}'.format(gc.DEC_FACTOR)

ExistMLObj = ExistMLPred(
    data_type=gc.DATA_TYPE,
    sub_folder_name=gc.sub_file_name,
    sub_name_short=gc.sub_file_name[:4],
    num_repetition=gc.NUM_REPETITION,
    num_electrode=gc.NUM_ELECTRODE,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)

file_subscript = file_subscript + '_from_' + eeg_file_suffix
[eeg_signals_odd, eeg_code_odd,
 eeg_type_odd] = ExistMLObj.import_eeg_odd_even_dat(
    file_subscript + '_odd'
)
print('eeg_signals_odd has shape {}'.format(eeg_signals_odd.shape))
print('eeg_type_odd_1d has shape {}'.format(eeg_type_odd.shape))
print('eeg_code_odd_1d has shape {}'.format(eeg_code_odd.shape))

[eeg_signals_even, eeg_code_even,
 eeg_type_even] = ExistMLObj.import_eeg_odd_even_dat(
    file_subscript + '_even'
)
print('eeg_signals_even has shape {}'.format(eeg_signals_even.shape))
print('eeg_type_even_1d has shape {}'.format(eeg_type_even.shape))
print('eeg_code_even_1d has shape {}'.format(eeg_code_even.shape))

file_subscript_2 = 'ML_down_{}_{}'.format(gc.DEC_FACTOR, eeg_file_suffix)
channel_name = 'channel_{}'.format(gc.channel_id + 1)
print('channel names are '.format(channel_name))

eeg_signals_odd_e_id = eeg_signals_odd[:, gc.channel_id, :]
eeg_signals_even_e_id = eeg_signals_even[:, gc.channel_id, :]

if len(gc.channel_id) > 1:
    channel_dim = len(gc.channel_id)
    sample_size_odd = rep_num_fit * gc.LETTER_DIM * gc.NUM_REP
    eeg_signals_odd_e_id = np.reshape(
        eeg_signals_odd_e_id, [sample_size_odd, channel_dim * gc.N_LENGTH]
    )

    sample_size_even = rep_num_pred * gc.LETTER_DIM * gc.NUM_REP
    eeg_signals_even_e_id = np.reshape(
        eeg_signals_even_e_id, [sample_size_even, channel_dim * gc.N_LENGTH]
    )

# Perform support vector classifier methods
ml_obj = None

if method_name == 'LR':
    ml_obj = LogisticRegression(max_iter=1000)

elif method_name == 'SVC':
    ml_obj = SVC(max_iter=1000, probability=True)

elif method_name == 'RF':
    ml_obj = RandomForestClassifier(random_state=0)

elif method_name == 'BAG':
    ml_obj = BaggingClassifier()

elif method_name == 'ADA':
    ml_obj = AdaBoostClassifier()

elif method_name == 'XGBoost':
    ml_obj = XGBClassifier(n_estimators=200, learning_rate=0.02)
method_name = 'EEG' + method_name

# Train with odd sequence first
ml_obj.fit(eeg_signals_odd_e_id, eeg_type_odd[0, :])


# Rename channel names
if len(gc.channel_id) == gc.NUM_ELECTRODE:
    channel_name = 'all_channels'
else:
    channel_id_str = [str(e+1) for e in gc.channel_id]
    channel_name = 'channel_' + '_'.join(channel_id_str)

# Currently, we apply the ML method to each channel separately
dict_odd_e_id = ExistMLObj.exist_ml_fit_predict(
    ml_obj, eeg_signals_odd_e_id, eeg_code_odd,
    gc.TARGET_LETTERS, rep_num_fit, rep_num_fit,
    method_name, channel_name, file_subscript_2,
    train_bool=True, sim_dat_bool=sim_dat_bool
)
print(dict_odd_e_id['cum'])

dict_even_e_id = ExistMLObj.exist_ml_fit_predict(
    ml_obj, eeg_signals_even_e_id, eeg_code_even,
    gc.TARGET_LETTERS, rep_num_fit, rep_num_pred,
    method_name, channel_name, file_subscript_2,
    train_bool=False, sim_dat_bool=sim_dat_bool
)
print(dict_even_e_id['cum'])
