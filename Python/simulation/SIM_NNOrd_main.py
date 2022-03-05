from self_py_fun.NNOrdinaryFun import *
from self_py_fun.ExistMLFun import *
import self_py_fun.GlobalSIM as sg
sns.set_context('notebook')

if sg.local_use:
    DATA_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data'
    SAVE_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data/SIM_files/'
else:
    DATA_DIC = '/home/mtianwen/EEG_MATLAB_data'
    SAVE_DIC = '/home/mtianwen/EEG_MATLAB_data/SIM_files/'

sim_dat_bool = True
sim_name_short = 'sim_' + str(sg.design_num + 1)
method_name = 'NNOrd'
sim_type_2 = sg.sim_type + '_down_' + sg.scenario_name
results = get_nn_ordinary_results(
    sim_dat_bool,
    sg.design_num+1, sg.subset_num+1, sg.scenario_name,
    k_num=None, dec_factor=1, eeg_file_suffix=None, channel_id=None,
    datadic=DATA_DIC, savedic=SAVE_DIC
)

# print(results.keys())
train_loss = results['train_loss']
train_accuracy = results['train_accuracy']
test_loss = results['test_loss']
test_accuracy = results['test_accuracy']
train_prob = results['train_probability']
test_prob = results['test_probability']


# Convert binary probs to letter-based probs
NNTrain = ExistMLPred(
    data_type='SIM_files',
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    num_repetition=sg.repet_num_fit,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE_FIT,
    local_bool=sg.local_use
)

[_, _, eeg_code_1d] = NNTrain.import_sim_ml_trunc_dataset(
        sim_type_2 + '_train',
        sg.LETTER_DIM, sg.REPETITION_TRN, 3
)

eeg_code_1d = np.reshape(
        eeg_code_1d[:, :sg.repet_num_fit, :],
        [1, sg.LETTER_DIM * sg.repet_num_fit * sg.NUM_REP]
)
nn_pred_dict = NNTrain.ml_predict(
        train_prob[:, np.newaxis], eeg_code_1d, sg.LETTER_DIM, sg.repet_num_fit
)
NNTrain.save_exist_ml_pred_results(
    nn_pred_dict, sg.repet_num_fit, sg.repet_num_fit, sg.LETTERS,
    method_name, sg.scenario_name, 'down', True, sim_dat_bool
)

NNTest = ExistMLPred(
    data_type='SIM_files',
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    num_repetition=sg.REPETITION_TEST,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE_FIT,
    local_bool=sg.local_use
)

# Test combined
[_, _, eeg_code_test_1d] = NNTest.import_sim_ml_trunc_dataset(
    sim_type_2 + '_test',
    sg.LETTER_DIM, sg.REPETITION_TEST, 3
)
nn_pred_dict_test = NNTest.ml_predict(
    test_prob[:, np.newaxis], eeg_code_test_1d, sg.LETTER_DIM, sg.REPETITION_TEST
)
NNTest.save_exist_ml_pred_results(
    nn_pred_dict_test, sg.repet_num_fit, sg.REPETITION_TEST,
    sg.LETTERS, method_name, sg.scenario_name, 'down', False, sim_dat_bool
)
