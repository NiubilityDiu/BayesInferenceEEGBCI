# Treat scale and variance parameter fixed, not in the Bayes framework
import self_py_fun.GlobalSIM as sg
from self_py_fun.BayesGenFun import *

# global constants
np.random.seed(sg.seed_index)
mcmc_axis = 0
q_mcmc = 2
method_name = 'BayesGenq' + str(q_mcmc)
scenario_name = sg.scenario_name
sim_suffix = 'fit'
s_stpsize = sg.s_stepsize
burn_in_samples = 0
eeg_dat_type = 'super_seq'
sim_dat_bool = True
seq_bool = False
file_subscript = '_'.join([sg.sim_type, sg.scenario_name, 'train'])
# kernel_option = 'gamma_exp'  # rbf or gamma_exp
kernel_option = sys.argv[6]
gamma_val = 1.0
s_tar = np.array([0.3])
s_ntar = np.array([0.3])
var_tar = np.array([10.0])
var_ntar = np.array([0.5])
s_fix = np.stack([s_tar, s_ntar], axis=-1)
var_fix = np.stack([var_tar, var_ntar], axis=-1)
# cont_fit_bool = True
cont_fit_bool = (sys.argv[7] == 'T')
# zeta_0 = 0
zeta_0 = float(sys.argv[8])
# level_num = 8
level_num = int(sys.argv[9])

a = 1  # shape for s_x_sq
b = 1  # rate for s_x_sq

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
 true_s_z_sq, true_s_x_sq, true_rho,
 signals_x, eeg_code, eeg_type,
 message] = BayesGenSeqObj.import_sim_bayes_gen_dataset(
    letter_dim, repet_num_fit, file_subscript, 3
)
zeta_true = 0 + (np.abs(true_beta_tar - true_beta_ntar) >= 0.1)

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


# Fit the single super-sequence model
# Comment this section only to perform multi-super-sequence model.
rep_single_ids = np.arange(1, repet_num_fit+1)
[signals_x, eeg_type, eeg_code, _, _, _,
 single_seq_len] = BayesGenSeqObj.split_convol_super_seq_by_seq_ids(
    signals_x, eeg_type, eeg_code, rep_single_ids, rep_single_ids,
    odd_reshape=1, even_reshape=3
)
true_beta = np.concatenate([true_beta_tar, true_beta_ntar], axis=1)
print('true_beta has shape {}'.format(true_beta.shape))
letter_dim = letter_dim * repet_num_fit
super_seq_length = signals_x.shape[2]
repet_num_fit = 1
#

mcmc_hyper_params = BayesGenSeqObj.standard_mcmc_prepare(
    s_tar, s_ntar, var_tar, var_ntar, kernel_option, gamma_val,
    None, None,
    n_length_fit, super_seq_length, num_electrode, q_mcmc, level_num
)

BayesGenSeqObj.standard_mcmc(
    signals_x, eeg_type,
    s_tar, s_ntar, var_tar, var_ntar,
    mcmc_hyper_params, a, b, sg.KAPPA, sg.BURN_IN,
    letter_dim, repet_num_fit, n_length_fit, super_seq_length, scenario_name,
    num_electrode, eeg_dat_type, sg.DEC_FACTOR, method_name, kernel_option, seq_bool,
    sim_dat_bool, cont_fit_bool, zeta_0, true_beta_tar, true_beta_ntar
)
