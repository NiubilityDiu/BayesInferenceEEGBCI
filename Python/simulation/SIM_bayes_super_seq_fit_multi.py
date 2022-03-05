import self_py_fun.GlobalSIM as sg
from self_py_fun.BayesGenFun import *

# global constants
np.random.seed(sg.seed_index)
q_mcmc = 2
method_name = 'BayesGenq' + str(q_mcmc)
scenario_name = sg.scenario_name
sim_suffix = 'fit'
eeg_dat_type = 'super_seq'
sim_dat_bool = True
# seq_bool = False
file_subscript = '_'.join([sg.sim_type, sg.scenario_name, 'train'])
# kernel_option = 'gamma_exp'  # rbf or gamma_exp
kernel_option = sys.argv[6]
s_tar = np.array([0.5])
s_ntar = np.array([0.5])
var_tar = np.array([5.0])
var_ntar = np.array([0.5])
gamma_val = np.array([1.8])
s_sine = np.array([1.0])
periodicity = np.array([0.4])
alpha_s = 0.1
beta_s = 0.1
# cont_fit_bool = False
cont_fit_bool = (sys.argv[7] == 'T')
# zeta_0 = 0.5
zeta_0 = float(sys.argv[8])
# rho_level_num = 8
rho_level_num = int(sys.argv[9])

super_seq_length = sg.SUPER_SEQ_LENGTH_TRN_FIT
n_length_fit = sg.N_LENGTH_FIT
num_electrode = sg.NUM_ELECTRODE
letter_dim = sg.LETTER_DIM
repet_num_fit = sg.repet_num_fit  # originally we plan to vary training sample size
repet_num_pred = sg.REPETITION_TEST

BayesGenSeqObj = BayesGenSeq(
    data_type=sg.data_type,
    sub_folder_name=sg.sim_common,
    sub_name_short='sim_' + str(sg.design_num+1),
    num_repetition=repet_num_fit,
    num_electrode=num_electrode,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=letter_dim,
    n_multiple=sg.N_MULTIPLE_FIT,
    local_bool=sg.local_use
)

# Import super-seq-train
[true_beta_tar, true_beta_ntar,
 true_s_x_sq, true_rho_t, true_rho_s,
 signals_x, eeg_code, eeg_type,
 message] = BayesGenSeqObj.import_sim_bayes_gen_dataset(
    letter_dim, repet_num_fit, file_subscript, 1
)
zeta_true = 0 + (np.abs(true_beta_tar - true_beta_ntar) >= 1)

# Subset signals, code, and type
if super_seq_length < signals_x.shape[2]:
    print('We take the subset.')
    signals_x = signals_x[..., :super_seq_length, :]
    true_beta_tar = true_beta_tar[:, :sg.N_LENGTH_FIT, :]
    true_beta_ntar = true_beta_ntar[:, :sg.N_LENGTH_FIT, :]
elif super_seq_length > signals_x.shape[2]:
    print('We add zeros to the end.')
    super_seq_diff = super_seq_length - signals_x.shape[2]
    signals_x = np.concatenate(
        [signals_x, np.zeros([num_electrode, letter_dim, super_seq_diff, 1])], axis=-2
    )
    true_beta_tar = np.concatenate([true_beta_tar, np.zeros([num_electrode, super_seq_diff, 1])], axis=1)
    true_beta_ntar = np.concatenate([true_beta_ntar, np.zeros([num_electrode, super_seq_diff, 1])], axis=1)
else:
    print('We do nothing.')
    pass

s_tar = np.tile(s_tar, [num_electrode])
s_ntar = np.tile(s_ntar, [num_electrode])

if sg.mean_fn_type == 'multi_channel_2':
    var_tar = np.array([5, 3, 1])
    # since the mean ratios are different
else:
    var_tar = np.tile(var_tar, [num_electrode])
    # otherwise, it is the location of split and merge.
var_ntar = np.tile(var_ntar, [num_electrode])
gamma_val = np.tile(gamma_val, [num_electrode])
s_sine = np.tile(s_sine, [num_electrode])
periodicity = np.tile(periodicity, [num_electrode])

mcmc_hyper_params = BayesGenSeqObj.standard_mcmc_prepare(
    s_tar, s_ntar, var_tar, var_ntar, kernel_option, gamma_val,
    s_sine, periodicity, n_length_fit, super_seq_length, num_electrode,
    q_mcmc, rho_level_num
)
theta_prior_pres = mcmc_hyper_params['theta_prior_pres'][0, ...]
lower = mcmc_hyper_params['lower'][0, :]
upper = mcmc_hyper_params['upper'][0, :]
zeta_prior_mean = np.zeros([n_length_fit, 1]) + 0.5
zeta_prior_pres = np.eye(n_length_fit)


BayesGenSeqObj.standard_mcmc_multi(
    signals_x, eeg_type,
    s_tar, s_ntar, var_tar, var_ntar,
    sg.KAPPA, sg.BURN_IN, sg.NUM_INTERVAL,
    eeg_dat_type, rho_level_num, q_mcmc,
    theta_prior_pres, lower, upper,
    zeta_prior_mean, zeta_prior_pres,
    alpha_s, beta_s, repet_num_fit, scenario_name,
    sg.DEC_FACTOR, method_name, sim_suffix,
    cont_fit_bool, sim_dat_bool, zeta_0,
    true_beta_tar, true_beta_ntar, zeta_true
)
