from self_py_fun.BayesGenFun import *

# Global Constants
local_use = (sys.argv[1].upper() == 'T')
# local_use = True
sim_type = sys.argv[2]
# sim_type = 'latency'
# sim_type = 'seq'
# sim_type = 'super_seq'
# sim_type = 'super_seq_trun_seq'
# sim_type = 'ML'
# repet_num_fit = 5
repet_num_fit = int(sys.argv[3])
data_type = 'SIM_files'
scenario_names = ['TrueGen', 'MisNoiseDist', 'MisLatencyLen25',
                  'MisLatencyLen35', 'MisSignalDist', 'MisTypeDist']
# scenario_name = scenario_names[1 - 1]
scenario_name = scenario_names[int(sys.argv[4]) - 1]
# mean_fn_type = 'multi_channel_2'
mean_fn_type = sys.argv[5]

if '25' in scenario_name:
    N_MULTIPLE = 5
elif '35' in scenario_name:
    N_MULTIPLE = 7
else:
    N_MULTIPLE = 6

FLASH_PAUSE_LENGTH = 5
N_MULTIPLE_FIT = 6
N_LENGTH_FIT = int(N_MULTIPLE_FIT * FLASH_PAUSE_LENGTH)
LETTER_DIM = 19
NUM_REP = 12
SEQ_LENGTH_FIT = (NUM_REP + N_MULTIPLE - 1) * FLASH_PAUSE_LENGTH

if 'Noise' in scenario_name:
    normal_bool = False
else:
    normal_bool = True

t_df = 5
num_electrode = 3
super_seq_length = 325  # fixed
n_length_fit = 30
letter_dim = 19
repet_num_pred = 5
q_mcmc = 2
method_name = 'BayesGenq' + str(q_mcmc)
sim_dat_bool = True
file_subscript = '_'.join([sim_type, scenario_name, 'train'])
# zeta_0 = 0.5
zeta_0 = float(sys.argv[6])
# rho_level_num = 8
rho_level_num = int(sys.argv[7])
design_num = int(sys.argv[8])
# design_num = 20
subset_sum = 100
job_id_dec_2 = 'binary_down_1_fit_zeta_{}'.format(zeta_0)
letter_dim_2 = repet_num_fit * LETTER_DIM

snr_mat = []

for subset_num in range(subset_sum):
    sim_common = 'sim_' + str(design_num + 1) + '_dataset_' + str(subset_num + 1)

    BayesGenSeqObj = BayesGenSeq(
        data_type=data_type,
        sub_folder_name=sim_common,
        sub_name_short='sim_' + str(design_num + 1),
        num_repetition=repet_num_fit,
        num_electrode=num_electrode,
        flash_and_pause_length=FLASH_PAUSE_LENGTH,  # fixed
        num_letter=letter_dim,
        n_multiple=N_MULTIPLE_FIT,  # fixed in the simulation fitting design
        local_bool=local_use
    )
    # Import super-seq-train
    [true_beta_tar, true_beta_ntar,
     true_s_x_sq, true_rho_t, true_rho_s,
     signals_x, eeg_code, eeg_type,
     message] = BayesGenSeqObj.import_sim_bayes_gen_dataset(
        letter_dim, repet_num_fit, file_subscript, 1
    )

    # Subset signals, code, and type
    if super_seq_length < signals_x.shape[2]:
        print('We take the subset.')
        signals_x = signals_x[..., :super_seq_length, :]
        true_beta_tar = true_beta_tar[:, :N_LENGTH_FIT, :]
        true_beta_ntar = true_beta_ntar[:, :N_LENGTH_FIT, :]
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

    d_mat = BayesGenSeqObj.create_design_mat_gen_bayes_seq(1)

    # Import mcmc
    [cov_s_mcmc, arg_rho_mcmc, zeta_mcmc, _,
     beta_tar_mcmc, beta_ntar_mcmc,
     scale_opt, var_opt, log_lkd_mcmc] = BayesGenSeqObj.import_mcmc(
        sim_type, method_name, repet_num_fit, scenario_name, job_id_dec_2, sim_dat_bool
    )
    mcmc_num = beta_tar_mcmc.shape[0]
    b_distance_multi = np.zeros([num_electrode])
    # s_x_sq_median_multi = np.zeros([gc.NUM_ELECTRODE])
    z_convol_multi = []
    for e in range(num_electrode):
        b_distance_multi[e] = BayesGenSeqObj.bhattacharyya_gaussian_distance(
            beta_tar_mcmc[:, e, :, 0], beta_ntar_mcmc[:, e, :, 0]
        )

    beta_mcmc_e = np.concatenate([beta_tar_mcmc, beta_ntar_mcmc], axis=2)  # (mcmc_num, 16, 100, 1)
    beta_mcmc_e = np.transpose(beta_mcmc_e, axes=(1, 0, 2, 3))  # (16, mcmc_num, 100, 1)
    for seq_id in range(letter_dim_2):
        t_mat_id = BayesGenSeqObj.create_transform_mat(
            eeg_type[seq_id * NUM_REP:(seq_id + 1) * NUM_REP], 1, 1, 'super_seq'
        )
        dt_mat_id = np.matmul(d_mat, t_mat_id)[np.newaxis, ...]  # (1, 1, 160, 100)
        z_e_odd_id = np.matmul(dt_mat_id, beta_mcmc_e)
        z_convol_multi.append(z_e_odd_id)

    z_convol_multi = np.transpose(np.reshape(
        np.stack(z_convol_multi, axis=0),
        [letter_dim_2, num_electrode, mcmc_num, SEQ_LENGTH_FIT, 1]
    ), axes=(1, 0, 2, 3, 4))
    # print('z_convol_multi has shape {}'.format(z_convol_multi.shape))
    signal_var = np.mean(np.var(z_convol_multi, axis=(-3, -2, -1)), axis=-1)
    observe_var = np.mean(np.var(signals_x, axis=(-2, -1)), axis=-1)
    print('signal_var = {}'.format(np.round(signal_var, decimals=2)))
    print('observe_var = {}'.format(np.round(observe_var, decimals=2)))
    snr_modify = signal_var / observe_var * 100
    print('Modified SNR % = {}'.format(np.round(snr_modify, decimals=2)))
    print('Channel Rank by SNR = {}'.format(np.argsort(snr_modify)[::-1] + 1))

    snr_mat.append(snr_modify)

snr_mat = np.stack(snr_mat, axis=0)
print(np.mean(snr_mat, axis=0))
print(np.std(snr_mat, axis=0))

