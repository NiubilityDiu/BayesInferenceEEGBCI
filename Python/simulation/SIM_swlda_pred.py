from self_py_fun.ExistMLFun import *
sns.set_context('notebook')

# Global constants
LETTERS = []
LETTERS.extend('The_quick_brown_fox')
# train_bool = True
file_subscript = 'down'
NUM_ELECTRODE = 1
FLASH_PAUSE_LENGTH = 5
LETTER_DIM = len(LETTERS)
# N_MULTIPLE = 6
REPETITION_TRN = 5  # total number of training set
REPETITION_TEST = 5  # total number of testing set (fixed for prediction)
NUM_REP = 12
# N_LENGTH = N_MULTIPLE * FLASH_PAUSE_LENGTH

# local_use = True
local_use = (sys.argv[1] == 'T')
# sim_type = 'ML'
sim_type = sys.argv[2]
# design_num = 1
design_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
subset_num = 100
# repet_num_fit = 5
repet_num_fit = int(sys.argv[3])
# fit value can be smaller than REPETITION_TRN, although we usually use complete training set

scenario_names = ['TrueGen', 'MisNoiseDist', 'MisLatencyLen25',
                  'MisLatencyLen35', 'MisSignalDist', 'MisTypeDist']
# scenario_name = scenario_names[1 - 1]
scenario_name = scenario_names[int(sys.argv[4]) - 1]
N_MULTIPLE_FIT = 6
zeta_0 = 0.7
# zeta_0 = float(sys.argv[5])
N_LENGTH_FIT = N_MULTIPLE_FIT * FLASH_PAUSE_LENGTH
sim_name_short = 'sim_' + str(design_num + 1)
method_name = 'EEGswLDA'


for subset_id in range(subset_num):
    SwLDATrain = ExistMLPred(
        data_type='SIM_files',
        sub_folder_name=sim_name_short + '_dataset_' + str(subset_id+1),
        sub_name_short=sim_name_short,
        num_repetition=repet_num_fit,
        num_electrode=NUM_ELECTRODE,
        flash_and_pause_length=FLASH_PAUSE_LENGTH,
        num_letter=LETTER_DIM,
        n_multiple=N_MULTIPLE_FIT,
        local_bool=local_use
    )

    SwLDATest = ExistMLPred(
        data_type='SIM_files',
        sub_folder_name=sim_name_short + '_dataset_' + str(subset_id+1),
        sub_name_short=sim_name_short,
        num_repetition=REPETITION_TEST,
        num_electrode=NUM_ELECTRODE,
        flash_and_pause_length=FLASH_PAUSE_LENGTH,
        num_letter=LETTER_DIM,
        n_multiple=N_MULTIPLE_FIT,
        local_bool=local_use
    )
    sim_type_2 = sim_type + '_down_' + scenario_name

    # training combined
    sim_common = 'sim_' + str(design_num + 1) + '_dataset_' + str(subset_id + 1)
    # SwLDATrain.sub_folder_name = sim_common
    [signals_trun, _, eeg_code_1d] = SwLDATrain.import_sim_ml_trunc_dataset(
        sim_type_2 + '_train',
        LETTER_DIM, REPETITION_TRN, 3
    )
    signals_trun = np.reshape(
        signals_trun, [LETTER_DIM, REPETITION_TRN, NUM_REP, N_LENGTH_FIT]
    )
    signals_trun = np.reshape(
        signals_trun[:, :repet_num_fit, :, :],
        [LETTER_DIM * repet_num_fit * NUM_REP, N_LENGTH_FIT]
    )
    eeg_code_1d = np.reshape(
        eeg_code_1d[:, :repet_num_fit, :],
        [1, LETTER_DIM * repet_num_fit * NUM_REP]
    )

    # Import swlda wts
    file_subscript_2 = file_subscript + '_zeta_' + str(zeta_0)
    swlda_wts_i = SwLDATrain.import_sim_matlab_swlda_wts_train(
        repet_num_fit, file_subscript_2, scenario_name
    )
    b = swlda_wts_i['b']
    inmodel = swlda_wts_i['inmodel']

    # Plot selected features
    SwLDATrain.plot_swlda_select_feature(
        inmodel, str(repet_num_fit) + '_down_zeta_' + str(zeta_0),
        scenario_name, sim_dat=True
    )

    # Predict on training set
    swlda_predict_prob = SwLDATrain.swlda_predict_y_prob(b, inmodel, signals_trun)
    # Collapse the electrode dimension
    swlda_pred_dict = SwLDATrain.ml_predict(
        swlda_predict_prob, eeg_code_1d, LETTER_DIM, repet_num_fit,
        test=False, num_repetition_test=None
    )
    SwLDATrain.save_exist_ml_pred_results(
        swlda_pred_dict, repet_num_fit, repet_num_fit,
        LETTERS, method_name, scenario_name,
        'down_zeta_' + str(zeta_0),
        train_bool=True, sim_dat_bool=True
    )

    # Test combined
    # SwLDATest.sub_folder_name = sim_common
    [signals_trun_test, _, eeg_code_test_1d] = SwLDATest.import_sim_ml_trunc_dataset(
        sim_type_2 + '_test',
        LETTER_DIM, REPETITION_TEST, 3
    )
    swlda_predict_prob_test = SwLDATest.swlda_predict_y_prob(
        b, inmodel, signals_trun_test
    )
    swlda_pred_test_dict = SwLDATest.ml_predict(
        swlda_predict_prob_test, eeg_code_test_1d,
        LETTER_DIM, REPETITION_TEST,
        test=True, num_repetition_test=REPETITION_TEST
    )
    SwLDATest.save_exist_ml_pred_results(
        swlda_pred_test_dict, repet_num_fit, REPETITION_TEST,
        LETTERS, method_name, scenario_name,
        'down_zeta_' + str(zeta_0),
        train_bool=False, sim_dat_bool=True
    )
