from self_py_fun.BayesGenFun import *
from self_py_fun.ExistMLFun import *
import self_py_fun.GlobalSIM as sg
from sklearn.linear_model import LogisticRegression
np.random.seed(sg.seed_index)
sns.set_context('notebook')

# Generate latency files
display_plot = True  # for pseudo mean fn
save_plot = False  # for final pseudo signals
simple_array_bool = True  # whether we save redundant 1/0
specific_bool = False  # whether we fix the location of target flashes
row_loc = 3  # if specific_bool = True, the location of target flash 1
col_loc = 10  # if specific_bool = True, the location of target flash 2
sim_name_short = 'sim_' + str(sg.design_num + 1)
rho_s = 0  # Not used in the single channel setting

SimData = EEGGeneralFun(
    num_repetition=sg.NUM_REPETITION,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE,
    local_bool=sg.local_use
)

# single channel code
[eeg_code_1d, eeg_type_1d] = SimData.generate_multiple_letter_code_and_type(
    sg.LETTERS, simple_array_bool, specific_bool, row_loc, col_loc
)
eeg_code_2d = np.reshape(eeg_code_1d, [sg.LETTER_DIM, sg.NUM_REPETITION * sg.NUM_REP])
eeg_type_2d = np.reshape(eeg_type_1d, [sg.LETTER_DIM, sg.NUM_REPETITION * sg.NUM_REP])
print('eeg_code_2d has shape {}'.format(eeg_code_2d.shape))
print('eeg_type_2d has shape {}'.format(eeg_type_2d.shape))

# Here mean_fun are different for latency signal length
mean_fn_tar, mean_fn_ntar = SimData.import_sim_mean_fn_single(
    sg.mean_fn_type, display_plot, sg.scenario_name
)
# mean_fn_tar = np.tile(mean_fn_tar, [sg.NUM_ELECTRODE, 1, 1])
# mean_fn_ntar = np.tile(mean_fn_ntar, [sg.NUM_ELECTRODE, 1, 1])

if 'Type' in sg.scenario_name:
    # Separate procedure for MisTypeDist specification:
    # Step 1: Use existing logistic regression from TrueGen dataset to generate (Z, Y) pool
    # Notice that this dataset includes (Z_indicator, Z_value, Y_label) columns.
    ExistMLObj = ExistMLPred(
        data_type=sg.data_type,
        sub_folder_name=sg.sim_common,
        sub_name_short=sim_name_short,
        num_repetition=sg.REPETITION_TRN,
        num_electrode=sg.NUM_ELECTRODE,
        flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
        num_letter=sg.LETTER_DIM,
        n_multiple=sg.N_MULTIPLE_FIT,
        local_bool=sg.local_use
    )

    # Step 1: Fit the logistic regression to form the pair (Z, Y)
    z_tar_len = 1000
    z_ntar_len = int(5 * z_tar_len)
    z_temp_tar = []
    z_temp_ntar = []
    for e in range(sg.NUM_ELECTRODE):
        z_temp_tar_e = np.random.multivariate_normal(
            mean=np.zeros([sg.N_LENGTH]), cov=np.eye(sg.N_LENGTH),
            size=z_tar_len
        )[..., np.newaxis] * sg.s_x_sq[e] + mean_fn_tar[e, ...]
        z_temp_ntar_e = np.random.multivariate_normal(
            mean=np.zeros([sg.N_LENGTH]), cov=np.eye(sg.N_LENGTH),
            size=z_ntar_len
        )[..., np.newaxis] * sg.s_x_sq[e] + mean_fn_ntar[e, ...]
        z_temp_tar.append(z_temp_tar_e)
        z_temp_ntar.append(z_temp_ntar_e)

    z_temp_tar = np.stack(z_temp_tar, axis=1)
    z_temp_ntar = np.stack(z_temp_ntar, axis=1)
    z_temp_pool = np.concatenate([z_temp_tar, z_temp_ntar], axis=0)
    z_temp_pool_ml = np.reshape(z_temp_pool, [z_tar_len + z_ntar_len, sg.NUM_ELECTRODE * sg.N_LENGTH])

    # Import ML-style dataset and ML-algorithm
    [signals_trun_train, eeg_type_1d_train, _] = ExistMLObj.import_sim_ml_trunc_dataset(
        'ML_down_TrueGen_train', sg.LETTER_DIM, sg.REPETITION_TRN, 1
    )

    # Logistic Regression
    ml_obj = LogisticRegression(max_iter=1000)
    ml_obj.fit(signals_trun_train, eeg_type_1d_train[0, :])
    # y_prob_temp_pool = ml_obj.predict_proba(z_temp_pool_ml)
    # print(y_prob_temp_pool)
    y_temp_pool = ml_obj.predict(z_temp_pool_ml)

    '''
    # swLDA
    swlda_wts_i = ExistMLObj.import_matlab_swlda_wts_train(
        sg.repet_num_fit, 'down', sg.scenario_name
    )
    b = swlda_wts_i['b']
    inmodel = swlda_wts_i['inmodel']
    ExistMLObj.plot_swlda_select_feature(inmodel, sg.repet_num_fit, sg.scenario_name)
    y_temp_pool = ExistMLObj.swlda_predict_y_prob(b, inmodel, z_temp_pool_ml)[:, 0]
    y_temp_pool = (y_temp_pool > 0) * 1
    # z_label_pool = np.concatenate([np.ones([z_tar_len]), -np.ones([z_ntar_len])], axis=0)
    '''

    y_temp_pool_tar_ids = np.where(y_temp_pool == 1)[0].tolist()
    y_temp_pool_ntar_ids = np.where(y_temp_pool == -1)[0].tolist()

    # Step 2: Regenerate Z given Y from y_temp_pool
    # Randomly select sg.TOTAL_STM_NUM * 0.2 from y_temp_pool_tar_ids
    y_random_tar = random.sample(y_temp_pool_tar_ids, int(sg.TOTAL_STM_NUM / 6))
    y_random_ntar = random.sample(y_temp_pool_ntar_ids, int(sg.TOTAL_STM_NUM * 5 / 6))
    z_value_tar = np.transpose(z_temp_pool[y_random_tar, ...], [1, 0, 2, 3])
    z_value_ntar = np.transpose(z_temp_pool[y_random_ntar, ...], [1, 0, 2, 3])
    permute_id_none = np.arange(sg.TOTAL_STM_NUM)
    SimData.generate_latency_signals(
        sg.s_x_sq, sg.rho, rho_s,
        z_value_tar, z_value_ntar,
        np.zeros_like(mean_fn_tar), np.zeros_like(mean_fn_ntar), permute_id_none,
        eeg_code_2d, eeg_type_2d, sg.LETTER_DIM, sg.NUM_REPETITION,
        sg.sim_common, sim_name_short,
        sg.sim_type + '_' + sg.scenario_name, save_plot
    )

else:
    # Modify eeg_type_1d with (prop*100)% mis-specification
    if 'Signal' in sg.scenario_name:
        eeg_type_1d = SimData.generate_multiple_mis_specified_type(
            eeg_type_1d, prop=0.1
        )

    type_tar_total = np.sum(eeg_type_1d).astype('int')
    type_ntar_total = len(eeg_type_1d) - type_tar_total

    permute_id = SimData.create_permute_beta_id(
        sg.LETTER_DIM, sg.NUM_REPETITION, eeg_type_1d
    )

    # Pure Model-based, without Z-level noise, use zero vector to separate tar and non-tar.
    flash_noise_tar = np.zeros([sg.NUM_ELECTRODE, type_tar_total, sg.N_LENGTH, 1])
    flash_noise_ntar = np.zeros([sg.NUM_ELECTRODE, type_ntar_total, sg.N_LENGTH, 1])
    print('flash_noise_tar has shape {}'.format(flash_noise_tar.shape))
    print('flash_noise_ntar has shape {}'.format(flash_noise_ntar.shape))

    # Notice that here eeg_type_2d, we still use original correct
    # But the corresponding latency signal is modified by mis-specified
    # when scenario_name contains Signal Keyword!!!
    SimData.generate_latency_signals(
        sg.s_x_sq, sg.rho, rho_s,
        flash_noise_tar,
        flash_noise_ntar,
        mean_fn_tar, mean_fn_ntar, permute_id,
        eeg_code_2d, eeg_type_2d, sg.LETTER_DIM, sg.NUM_REPETITION,
        sg.sim_common, sim_name_short,
        sg.sim_type + '_' + sg.scenario_name, save_plot
    )


# Generate super-seq and super-seq-truncate-seq
single_seq_bool = False
convol_bool = True
save_plot_bool = True
sg.sim_type = 'super_seq'

GenSuperSeq = BayesGenSeq(
    data_type=sg.data_type,
    num_repetition=sg.NUM_REPETITION,
    num_electrode=sg.NUM_ELECTRODE,
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE,
    local_bool=sg.local_use
)

design_x_train = GenSuperSeq.create_design_mat_gen_bayes_seq(sg.REPETITION_TRN)
# Notice that signals have already been tiled, so we don't need t_matrix here.

[beta_tar, beta_ntar, s_x_sq, rho, _,
 signals, eeg_code, eeg_type,
 message] = GenSuperSeq.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.NUM_REPETITION, 'latency_'+sg.scenario_name, 2
)

# ma = np.array([0.9])
# Split the entire dataset as training and test
# Train
signals_train = signals[:, :, :sg.REPETITION_TRN * sg.NUM_REP, ...]
GenSuperSeq.generate_super_seq_from_latency_signals(
    signals_train, design_x_train,
    eeg_code[:, :sg.REPETITION_TRN * sg.NUM_REP],
    eeg_type[:, :sg.REPETITION_TRN * sg.NUM_REP],
    sg.eta, s_x_sq, rho, None, beta_tar, beta_ntar,
    sg.normal_bool, sg.t_df,
    sg.LETTER_DIM, sg.REPETITION_TRN, message,
    sim_name_short, sg.sim_type + '_' + sg.scenario_name + '_train',
    convol_bool, single_seq_bool, save_plot_bool, noise_arq=None  # generate the noise inside the function
)


# Test
signals_test = signals[:, :, sg.REPETITION_TRN * sg.NUM_REP:, ...]
eeg_code_test = eeg_code[:, sg.REPETITION_TRN * sg.NUM_REP:]
eeg_type_test = eeg_type[:, sg.REPETITION_TRN * sg.NUM_REP:]

# Notice that under current design,
# subjects are given # number of testing sequences without pausing before swLDA makes prediction.
# So if we stick with it,
# we need to generate # testing cases separately.
# But we keep the noise the same to make the results comparable.
# For traditional ML methods using truncated signal segments as feature input
# we keep the original testing sequences.


design_x_test = GenSuperSeq.create_design_mat_gen_bayes_seq(sg.REPETITION_TEST)
len_test_rep_id = design_x_test.shape[0]
GenSuperSeq.generate_super_seq_from_latency_signals(
    signals_test, design_x_test,
    eeg_code_test, eeg_type_test,
    sg.eta, s_x_sq, rho, None,
    beta_tar, beta_ntar, sg.normal_bool, sg.t_df,
    sg.LETTER_DIM, sg.REPETITION_TEST, message,
    sim_name_short, sg.sim_type + '_' + sg.scenario_name + '_test',
    convol_bool, single_seq_bool, save_plot_bool,
    noise_arq=None
)
del GenSuperSeq, design_x_train, signals_train, design_x_test, signals_test


# Generate ML-type dataset
show_dim_bool = True
file_suffix = 'down'
print('This is a general signal pre-processing file for discriminant ML methods!\n')
# print('The simulation type here is independent of the sys.argv[2] input!\n')

# training set
ExistMLTrain = ExistMLPred(
    data_type=sg.data_type,
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    num_repetition=sg.REPETITION_TRN,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE_FIT, # we use the same length for swLDA
    local_bool=sg.local_use
)

[_, _, _, _, _,
 signals_train, eeg_code_train, eeg_type_train, _] = ExistMLTrain.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.REPETITION_TRN, sg.sim_type + '_' + sg.scenario_name+ "_train", 1
)
print('signals_train has shape {}'.format(signals_train.shape))

# Produce truncated eeg signals subset
if sg.N_MULTIPLE < sg.N_MULTIPLE_FIT:
    super_seq_trn_diff = sg.SUPER_SEQ_LENGTH_TRN_FIT - sg.SUPER_SEQ_LENGTH_TRN
    signals_train = np.concatenate(
        [signals_train,
         np.zeros([sg.NUM_ELECTRODE, sg.LETTER_DIM, super_seq_trn_diff, 1])], axis=-2
    )

eeg_signals_trun_train, _ = ExistMLTrain.create_truncate_segment_batch(
    signals_train, None, sg.LETTER_DIM, sg.REPETITION_TRN, show_dim_bool
)

[eeg_signals_trun_t_mean,
 eeg_signals_trun_nt_mean,
 eeg_signals_trun_t_cov,
 eeg_signals_trun_nt_cov] = ExistMLTrain.produce_trun_mean_cov_subset(
    eeg_signals_trun_train, eeg_type_train
)

ExistMLTrain.produce_mean_covariance_plots(
    eeg_signals_trun_t_mean, eeg_signals_trun_nt_mean, None, None,
    file_suffix + '_' + sg.scenario_name + '_train'
)

ExistMLTrain.save_sim_ml_trunc_dataset(
    eeg_signals_trun_train, eeg_type_train, eeg_code_train,
    file_suffix + '_' + sg.scenario_name + '_train'
)

# test set
ExistMLTest = ExistMLPred(
    data_type=sg.data_type,
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    num_repetition=sg.REPETITION_TEST,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE_FIT,
    local_bool=sg.local_use
)

[_, _, _, _, _,
 signals_test, eeg_code_test, eeg_type_test, _] = ExistMLTest.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.REPETITION_TEST,
    sg.sim_type + '_' + sg.scenario_name + "_test", 1
)

if sg.N_MULTIPLE < sg.N_MULTIPLE_FIT:
    super_seq_test_diff = sg.SUPER_SEQ_LENGTH_TEST_FIT - sg.SUPER_SEQ_LENGTH_TEST
    signals_test = np.concatenate(
        [signals_test,
         np.zeros([sg.NUM_ELECTRODE, sg.LETTER_DIM, super_seq_test_diff, 1])], axis=-2
    )
# Produce truncated eeg signals subset
eeg_signals_trun_test, _ = ExistMLTest.create_truncate_segment_batch(
    signals_test, None, sg.LETTER_DIM, sg.REPETITION_TEST, show_dim_bool
)

ExistMLTest.save_sim_ml_trunc_dataset(
    eeg_signals_trun_test, eeg_type_test, eeg_code_test,
    file_suffix + '_' + sg.scenario_name + '_test'
)