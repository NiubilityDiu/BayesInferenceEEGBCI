import self_py_fun.GlobalSIM as sg
from self_py_fun.ExistMLFun import *
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
sns.set_context('notebook')


# Global constants
reshape_option = 1  # convert it to 1d-array
sim_dat_bool = True
# method_name = 'XGBoost'
method_name = sys.argv[6]

ExistMLObj = ExistMLPred(
    data_type=sg.data_type,
    sub_folder_name=sg.sim_common,
    sub_name_short='sim_' + str(sg.design_num + 1),
    num_repetition=sg.REPETITION_TRN,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE,
    local_bool=sg.local_use
)

trn_name = sg.sim_type + '_down_' + sg.scenario_name + '_train'
[signals_trun, eeg_type_1d, eeg_code_1d] = ExistMLObj.import_sim_ml_trunc_dataset(
    trn_name, sg.LETTER_DIM, sg.REPETITION_TRN, reshape_option
)
print('signals_trun has shape {}'.format(signals_trun.shape))
print('eeg_type_1d has shape {}'.format(eeg_type_1d.shape))
print('eeg_code_1d has shape {}'.format(eeg_code_1d.shape))

test_name = sg.sim_type + '_down_' + sg.scenario_name + '_test'
[signals_trun_test, eeg_type_test_1d, eeg_code_test_1d] = ExistMLObj.import_sim_ml_trunc_dataset(
    test_name, sg.LETTER_DIM, sg.REPETITION_TEST, reshape_option
)

print('signals_trun_test has shape {}'.format(signals_trun_test.shape))
print('eeg_type_test_1d has shape {}'.format(eeg_type_test_1d.shape))
print('eeg_code_test_1d has shape {}'.format(eeg_code_test_1d.shape))

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
    ml_obj = XGBClassifier(n_estimators=100, learning_rate=0.02)

ml_obj.fit(signals_trun, eeg_type_1d[0, :])
method_name = 'EEG' + method_name


# Currently, we apply the ML method to each channel separately
dict_odd_e_id = ExistMLObj.exist_ml_fit_predict(
    ml_obj, signals_trun, eeg_code_1d,
    sg.LETTERS, sg.REPETITION_TRN, sg.REPETITION_TRN,
    method_name, sg.scenario_name, 'down',
    train_bool=True, sim_dat_bool=sim_dat_bool
)
print(dict_odd_e_id)

dict_even_e_id = ExistMLObj.exist_ml_fit_predict(
    ml_obj, signals_trun_test, eeg_code_test_1d,
    sg.LETTERS, sg.REPETITION_TRN, sg.REPETITION_TEST,
    method_name, sg.scenario_name, 'down',
    train_bool=False, sim_dat_bool=sim_dat_bool
)
print(dict_even_e_id)
