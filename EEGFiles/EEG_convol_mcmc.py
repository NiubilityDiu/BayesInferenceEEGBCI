import sys
sys.path.insert(0, './self_py_fun')
# from self_py_fun.MMAR_q_latent_z_multi import *
import self_py_fun.EEGConvolGlobal as gc
from self_py_fun.ConvolFun import *
print(os.getcwd())
tf.compat.v1.random.set_random_seed(612)
np.random.seed(612)

EEGWLSOpt = WLSOpt(
    # EEGPreFun class
    data_type=gc.data_type,
    sub_folder_name=gc.sub_file_name,
    # EEGGeneralFun class
    num_repetition=gc.num_repetition,
    num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    num_letter=gc.num_letter,
    n_multiple=gc.n_multiple,
    local_bool=gc.local_use
)
EEGWLSOpt.print_sub_trn_info(gc.trn_repetition)

'''MCMC part:'''
# trn_seq_length = EEGWLSOpt.compute_trn_seq_length(GC.trn_repetition)
# Import down-sample data and specify the part for training
# If not specify letter_dim and trn_repetition, the whole dataset is considered as training.
[eeg_signals, eeg_signals_trun_sub, eeg_code, eeg_type] = \
    EEGWLSOpt.import_eeg_processed_dat_wls(
        gc.file_subscript,
        letter_dim=gc.letter_dim,
        trn_repetition=gc.trn_repetition,
        reshape_to_1d=True)

print('eeg_signals has shape {}'.format(eeg_signals.shape))
print('eeg_signals_trun_sub has shape {}'.format(eeg_signals_trun_sub.shape))
print('eeg_code has shape {}'.format(eeg_code.shape))
print('eeg_type has shape {}'.format(eeg_type.shape))

# Initial values and hyper-parameters:
# Extract stratified sample mean and sample covariance,
# Notice that those are the summary statistics of the training set!
[eeg_t_mean, eeg_nt_mean, eeg_t_cov, eeg_nt_cov] = \
    EEGWLSOpt.produce_trun_mean_cov_subset(eeg_signals_trun_sub, eeg_type)

print(np.linalg.det(eeg_t_cov))
print(np.linalg.det(eeg_nt_cov))

# Perform MCMC on single channel
mcmc_chan = 0  # Channel 1
eeg_t_mean_2 = eeg_t_mean[np.newaxis, mcmc_chan, :]
eeg_nt_mean_2 = eeg_nt_mean[np.newaxis, mcmc_chan, :]
eeg_t_cov_2 = eeg_t_cov[np.newaxis, mcmc_chan, :]
eeg_nt_cov_2 = eeg_nt_cov[np.newaxis, mcmc_chan, :]
eeg_signals_2 = eeg_signals[..., mcmc_chan, np.newaxis]

print('signals_2 has shape {}'.format(eeg_signals_2.shape))
print('eeg_t_mean_2 has shape {}'.format(eeg_t_mean_2.shape))
print('eeg_nt_mean_2 has shape {}'.format(eeg_nt_mean_2.shape))
print('eeg_t_cov_2 has shape {}'.format(eeg_t_cov_2.shape))
print('eeg_nt_cov_2 has shape {}'.format(eeg_nt_cov_2.shape))

hyper_param_dict = EEGWLSOpt.provide_hyper_params(
    eeg_t_mean_2, eeg_nt_mean_2, hyper_delta_var=1.0, hyper_sigma_r_sq=1.0)

start_time = time.time()
with tf.Graph().as_default() as g:

    id_beta = EEGWLSOpt.rearrange.create_permute_beta_id(gc.letter_dim, gc.trn_repetition, eeg_type)
    design_x = EEGWLSOpt.rearrange.create_design_matrix_bayes(gc.letter_dim, gc.trn_repetition)

    target_log_prob_eeg_convol_fn = EEGWLSOpt.get_log_prob_eeg_convol_fn(
        gc.letter_dim, gc.trn_repetition, hyper_param_dict,
        eeg_signals=eeg_signals_2,
        id_beta=id_beta, design_x=design_x)

    para_mcmc = EEGWLSOpt.mcmc_sample_chain(
        target_log_prob_eeg_convol_fn,
        eeg_t_mean_2, eeg_nt_mean_2,
        eeg_t_cov_2, eeg_nt_cov_2,
        gc.n_samples, gc.n_burn_in,
        gc.num_steps_between_results, gc.step_size_init,
        gc.target_accept_prob, gc.num_leapfrog_steps)

    g.finalize()
end_time_1 = time.time()
print("Running tf static graphic using costs %g seconds" % (end_time_1 - start_time))

with tf.compat.v1.Session(graph=g) as mcmc_sess:
    # mcmc_sess.run(tf.compat.v1.global_variables_initializer())
    try:
        [para_mcmc_] = mcmc_sess.run([para_mcmc])

    except Exception as e:
        # Shorten the giant stack trace
        lines = str(e).split('\n')
        # print('\n'.join(lines[:5]+['...']+lines[-3:]))
        for _, message in enumerate(lines):
            print(message)

end_time_2 = time.time()
print("Evaluating tf costs another %g seconds" % (end_time_2 - end_time_1))
# acceptance_prob = np.exp(np.min(log_accept_ratio_, 0))
# print('The accpetance probability is {}'.format(acceptance_prob))

# Final convergence check
with tf.Graph().as_default() as gg:
    delta_1_r_hat = tfp.mcmc.potential_scale_reduction(para_mcmc_[0], independent_chain_ndims=2)
    delta_0_r_hat = tfp.mcmc.potential_scale_reduction(para_mcmc_[1], independent_chain_ndims=2)
    pres_chky_1_r_hat = tfp.mcmc.potential_scale_reduction(para_mcmc_[2], independent_chain_ndims=2)
    pres_chky_0_r_hat = tfp.mcmc.potential_scale_reduction(para_mcmc_[3], independent_chain_ndims=2)
    gg.finalize()

with tf.compat.v1.Session(graph=gg) as sess:
    print('delta_1_r_hat, channel {}:\n {}'.format(mcmc_chan, sess.run(delta_1_r_hat)))
    print('delta_0_r_hat, channel {}:\n {}'.format(mcmc_chan, sess.run(delta_0_r_hat)))
    print('pres_chky_1_r_hat, channel {}:\n {}'.format(mcmc_chan, sess.run(pres_chky_1_r_hat)))
    print('pres_chky_0_r_hat, channel {}:\n {}'.format(mcmc_chan, sess.run(pres_chky_0_r_hat)))

# Compute the norm of delta_1 and delta_0
delta_1_norm = np.linalg.norm(np.squeeze(para_mcmc_[0], axis=1), ord=2, axis=1)
delta_0_norm = np.linalg.norm(np.squeeze(para_mcmc_[1], axis=1), ord=2, axis=1)
print('The norm of delta_1:\n {}\n'.format(delta_1_norm))
print('The norm of delta_0:\n {}\n'.format(delta_0_norm))
# Compute the determinant of pres_chky_1
pres_chky_1 = EEGWLSOpt.prior.convert_1d_array_to_upper_triangular(para_mcmc_[2])
pres_chky_0 = EEGWLSOpt.prior.convert_1d_array_to_upper_triangular(para_mcmc_[3])

pres_chky_1_det = tf.linalg.det(tf.squeeze(pres_chky_1, axis=1))
pres_chky_0_det = tf.linalg.det(tf.squeeze(pres_chky_0, axis=1))
pres_chky_1_det = tf.keras.backend.eval(pres_chky_1_det)
pres_chky_0_det = tf.keras.backend.eval(pres_chky_0_det)
print('The determinant of the pres_chky_1:\n {}\n'.format(pres_chky_1_det))
print('The determinant of the pres_chky_0:\n {}\n'.format(pres_chky_0_det))

EEGWLSOpt.save_hmc_params_est(para_mcmc_, gc.message)