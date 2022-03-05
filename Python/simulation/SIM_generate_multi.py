from self_py_fun.BayesGenFun import *
from self_py_fun.ExistMLFun import *
import self_py_fun.GlobalSIM as sg
np.random.seed(sg.seed_index)
sns.set_context('notebook')

# Generate latency files
display_mean_fn_bool = True  # for pseudo mean fn
save_latency_plot = False  # for final pseudo signals
simple_array_bool = True  # whether we save redundant 1/0
specific_bool = False  # whether we fix the location of target flashes
row_loc = 3  # if specific_bool = True, the location of target flash 1
col_loc = 9  # if specific_bool = True, the location of target flash 2
sim_name_short = 'sim_' + str(sg.design_num + 1)

SimData = EEGGeneralFun(
    num_repetition=sg.NUM_REPETITION,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE,
    local_bool=sg.local_use
)

[eeg_code_1d, eeg_type_1d] = SimData.generate_multiple_letter_code_and_type(
    sg.LETTERS, simple_array_bool, specific_bool, row_loc, col_loc
)
eeg_code_2d = np.reshape(eeg_code_1d, [sg.LETTER_DIM, sg.NUM_REPETITION * sg.NUM_REP])
eeg_type_2d = np.reshape(eeg_type_1d, [sg.LETTER_DIM, sg.NUM_REPETITION * sg.NUM_REP])
print('eeg_code_2d has shape {}'.format(eeg_code_2d.shape))
print('eeg_type_2d has shape {}'.format(eeg_type_2d.shape))

# Mean functions of multi-dimension
mean_fn_tar, mean_fn_ntar = SimData.import_sim_mean_fn_multi(
    sg.mean_fn_type, display_mean_fn_bool, sg.scenario_name
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

sigma_s_sq = sg.s_x_sq  # scalar value
rho_t = sg.rho  # (1d-vector input)
rho_s = sg.rho_s  # fix

# Notice that here eeg_type_2d, we still use original correct
# But the corresponding latency signal is modified by mis-specified
# when scenario_name contains Signal Keyword!!!
SimData.generate_latency_signals(
    sigma_s_sq, rho_t, rho_s, flash_noise_tar, flash_noise_ntar,
    mean_fn_tar, mean_fn_ntar, permute_id,
    eeg_code_2d, eeg_type_2d, sg.LETTER_DIM, sg.NUM_REPETITION,
    sg.sim_common, sim_name_short,
    sg.sim_type + '_' + sg.scenario_name, save_latency_plot
)


# Generate super-seq and super-seq-truncate-seq
single_seq_bool = False
convolution_bool = True
save_super_seq_plot_bool = True
sg.sim_type = 'super_seq'  # do not delete

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

[beta_tar, beta_ntar, _, _, _,
 signals_latency, eeg_code, eeg_type,
 message] = GenSuperSeq.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.NUM_REPETITION, 'latency_' + sg.scenario_name, 2
)
signals_latency_train = signals_latency[:, :, :sg.REPETITION_TRN * sg.NUM_REP, ...]
print('signals_latency_train has shape {}'.format(signals_latency_train.shape))

GenSuperSeq.generate_super_seq_from_latency_signals(
    signals_latency_train, design_x_train,
    eeg_code[:, :sg.REPETITION_TRN * sg.NUM_REP],
    eeg_type[:, :sg.REPETITION_TRN * sg.NUM_REP],
    sg.eta, sigma_s_sq, rho_t, rho_s, beta_tar, beta_ntar,
    sg.normal_bool, sg.t_df,
    sg.LETTER_DIM, sg.REPETITION_TRN, message,
    sim_name_short, sg.sim_type + '_' + sg.scenario_name + '_train',
    convolution_bool, single_seq_bool, save_super_seq_plot_bool, noise_arq=None
    # generate the noise inside the function
)


# Test
signals_latency_test = signals_latency[:, :, sg.REPETITION_TRN * sg.NUM_REP:, ...]
eeg_code_test = eeg_code[:, sg.REPETITION_TRN * sg.NUM_REP:]
eeg_type_test = eeg_type[:, sg.REPETITION_TRN * sg.NUM_REP:]
print('signals_latency_test has shape {}'.format(signals_latency_test.shape))

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
    signals_latency_test, design_x_test,
    eeg_code_test, eeg_type_test,
    sg.eta, sigma_s_sq, rho_t, rho_s,
    beta_tar, beta_ntar, sg.normal_bool, sg.t_df,
    sg.LETTER_DIM, sg.REPETITION_TEST, message,
    sim_name_short, sg.sim_type + '_' + sg.scenario_name + '_test',
    convolution_bool, single_seq_bool, save_super_seq_plot_bool, noise_arq=None
)
del GenSuperSeq, design_x_train, signals_latency_train, \
    design_x_test, signals_latency_test


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
    n_multiple=sg.N_MULTIPLE_FIT,  # we use the same length for swLDA
    local_bool=sg.local_use
)

[_, _, _, _, _,
 signals_train, eeg_code_train, eeg_type_train, _] = ExistMLTrain.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.REPETITION_TRN, sg.sim_type + '_' + sg.scenario_name + "_train", 1
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
