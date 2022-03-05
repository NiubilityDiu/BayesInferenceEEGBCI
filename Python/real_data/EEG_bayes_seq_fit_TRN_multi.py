import self_py_fun.GlobalEEG as gc
from self_py_fun.BayesGenFun import *


# global constants
np.random.seed(gc.seed_index+2)
mcmc_axis = 0
reshape_1d_bool = False
eeg_dat_type = 'super_seq'
q_mcmc = 2
num_electrode = len(gc.channel_id)
method_name = 'BayesGenq' + str(q_mcmc)
burn_in_samples = 0
seq_bool = False
sim_dat_bool = False
n_length_fit = gc.N_LENGTH
# bp_low = 0.5
bp_low = float(sys.argv[9])
# bp_upp = 6
bp_upp = float(sys.argv[4])
if float(bp_upp) == int(bp_upp):
    bp_upp = int(bp_upp)
else:
    bp_upp = float(bp_upp)
if bp_upp < 0:
    eeg_file_suffix = 'raw'
else:
    eeg_file_suffix = 'raw_bp_{}_{}'.format(bp_low, bp_upp)
# kernel_option = 'gamma_exp'
kernel_option = sys.argv[5]
s_tar = np.array([0.5])
s_ntar = np.array([0.5])
# s_tar = np.array([float(sys.argv[10])])
# s_ntar = np.array([float(sys.argv[10])])
var_tar = np.array([1.0])
var_ntar = np.array([1.0])
gamma_val = np.array([1.5])
# gamma_val = np.array([float(sys.argv[11])])
s_sine = np.array([1.0])
periodicity = np.array([0.4])
alpha_s = 0.1
beta_s = 0.1
# cont_fit_bool = True
cont_fit_bool = (sys.argv[6] == 'T')
# zeta_0 = 0
zeta_0 = float(sys.argv[7])
# rho_level_num = 8
rho_level_num = int(sys.argv[8])
repet_num_single_seq = 1
repet_num_fit = len(gc.rep_odd_id)


BayesGenSeqObj = BayesGenSeq(
    data_type=gc.DATA_TYPE,
    sub_name_short=gc.sub_file_name[:4],
    sub_folder_name=gc.sub_file_name,
    num_repetition=gc.NUM_REPETITION,
    num_electrode=num_electrode,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)

print('We are using channel {} for prediction.'.format(gc.channel_id + 1))
channel_id_str = [str(e+1) for e in gc.channel_id]
if len(gc.channel_id) == 16:
    channel_name = 'all_channels'
else:
    channel_name = 'channel_' + '_'.join(channel_id_str)

gc_file_subscript = '{}_{}_from_{}'.format(
    gc.file_subscript, gc.DEC_FACTOR, eeg_file_suffix
)
# Import the training set without subsetting yet
[signal_x, code_3d, type_3d] = BayesGenSeqObj.import_eeg_processed_dat(
    gc_file_subscript, reshape_1d_bool
)
signal_x_sub = np.transpose(signal_x[:, gc.channel_id, ...], [1, 0, 2, 3])

# eeg_spatial_corr = BayesGenSeqObj.import_eeg_spatial_corr(eeg_file_suffix)
# eeg_spatial_corr_sub = eeg_spatial_corr[gc.channel_id, :][:, gc.channel_id]
eeg_spatial_corr_sub = None

# For real data training set, we split the dataset by odd/even number and test on the entire training set
# No need to worry about the scaling factor right now, make sure the method works
[signal_x_odd, eeg_type_odd_1d, eeg_code_odd_1d,
 signal_x_even, eeg_type_even_3d, eeg_code_even_3d,
 single_seq_len] = BayesGenSeqObj.split_convol_super_seq_by_seq_ids(
    signal_x_sub, type_3d, code_3d, gc.rep_odd_id, gc.rep_even_id,
    odd_reshape=1, even_reshape=3
)

s_tar = np.tile(s_tar, [num_electrode])
s_ntar = np.tile(s_ntar, [num_electrode])
var_tar = np.tile(var_tar, [num_electrode])
var_ntar = np.tile(var_ntar, [num_electrode])
gamma_val = np.tile(gamma_val, [num_electrode])
s_sine = np.tile(s_sine, [num_electrode])
periodicity = np.tile(periodicity, [num_electrode])

mcmc_hyper_params = BayesGenSeqObj.standard_mcmc_prepare(
    s_tar, s_ntar, var_tar, var_ntar, kernel_option, gamma_val,
    s_sine, periodicity, n_length_fit, single_seq_len, num_electrode,
    q_mcmc, rho_level_num
)
theta_prior_pres = mcmc_hyper_params['theta_prior_pres'][0, ...]
lower = mcmc_hyper_params['lower'][0, :]
upper = mcmc_hyper_params['upper'][0, :]
zeta_prior_mean = np.zeros([n_length_fit, 1]) + 0.5
zeta_prior_pres = np.eye(n_length_fit)


BayesGenSeqObj.standard_mcmc_multi(
    signal_x_odd, eeg_type_odd_1d,
    s_tar, s_ntar, var_tar, var_ntar,
    gc.KAPPA, gc.BURN_IN, gc.NUM_INTERVAL,
    eeg_dat_type, rho_level_num, q_mcmc,
    theta_prior_pres, lower, upper,
    zeta_prior_mean, zeta_prior_pres,
    alpha_s, beta_s, repet_num_single_seq, gc.channel_id,
    gc.DEC_FACTOR, method_name, eeg_file_suffix,
    cont_fit_bool, sim_dat_bool, zeta_0,
    signal_x_corr_s=eeg_spatial_corr_sub,
    # scale=s_tar[0], gamma=gamma_val[0]
)



