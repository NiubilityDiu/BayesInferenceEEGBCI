import sys
sys.path.insert(0, './self_py_fun')
# from self_py_fun.MMAR_q_latent_z_multi import *
import self_py_fun.EEGConvolGlobal as gc
from self_py_fun.ConvolFun import *
from self_py_fun.xDAFun import *
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

[eeg_signals, eeg_signals_trun_sub, eeg_code, eeg_type] = \
    EEGWLSOpt.import_eeg_processed_dat_wls(
        gc.file_subscript,
        letter_dim=gc.letter_dim,
        trn_repetition=gc.trn_repetition,
        reshape_to_1d=True)
eeg_signals = eeg_signals[..., np.newaxis]
print('eeg_signals has shape {}'.format(eeg_signals.shape))
print('eeg_signals_trun_sub has shape {}'.format(eeg_signals_trun_sub.shape))
print('eeg_code has shape {}'.format(eeg_code.shape))
print('eeg_type has shape {}'.format(eeg_type.shape))

# Initial values and hyper-parameters:
# Extract stratified sample mean and sample covariance,
# Notice that those are the summary statistics of the training set!
[eeg_t_mean, eeg_nt_mean, eeg_t_cov, eeg_nt_cov] = \
    EEGWLSOpt.produce_trun_mean_cov_subset(eeg_signals_trun_sub, eeg_type)

# design_x = EEGWLSOpt.rearrange.create_design_matrix_bayes(gc.letter_dim, gc.trn_repetition)
EEGGibbs = ConvolGibbs(
    n_multiple=gc.n_multiple,
    num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    mu_1_delta=np.zeros([gc.num_electrode, gc.n_length, 1]),
    mu_0_delta=np.zeros([gc.num_electrode, gc.n_length, 1]),
    sigma_sq_delta=10.0,
    nu_1=gc.n_length + 1, nu_0=gc.n_length + 1,
    scale_1=np.tile(np.eye(gc.n_length, gc.n_length)[np.newaxis, ...], [gc.num_electrode, 1, 1]),
    scale_0=np.tile(np.eye(gc.n_length, gc.n_length)[np.newaxis, ...], [gc.num_electrode, 1, 1]),
    alpha=5.0, beta=2.0,
    kappa=gc.kappa,
    letter_dim=gc.letter_dim,
    trn_repetition=gc.trn_repetition
)

# np.random.seed(612)
pres_matrix = np.tile(np.eye(gc.n_length)[np.newaxis, ...], [gc.num_electrode, 1, 1])
pres_lambda = np.random.uniform(size=(16, 1))
delta = np.random.uniform(size=(16,25,1))
gamma = np.random.binomial(1,0,size=(16,25))
mu_conditional_beta = np.random.normal(size=(16,25,1))

beta = EEGGibbs.update_individual_beta(
    pres_matrix, pres_lambda, delta, gamma, mu_conditional_beta
)
print(beta[0,:,0])

# Initialize parameters
[
    delta_1_mcmc, delta_0_mcmc,
    pi_1_mcmc, pi_0_mcmc,
    gamma_1_mcmc, gamma_0_mcmc,
    pres_mat_1_mcmc, pres_mat_0_mcmc,
    beta_iter_k,
    pres_lambda_mcmc
] = EEGGibbs.initialize_parameters(eeg_type)

beta_1_indices = np.where(eeg_type == 1)[0]
beta_0_indices = np.where(eeg_type == 0)[0]
append_zeros = np.zeros([gc.n_multiple-1, gc.num_electrode, gc.n_length, 1])
dm_cond_beta = EEGGibbs.create_partial_design_matrix_bayes()
dm_beta = EEGGibbs.create_design_matrix_bayes()

for k in range(EEGGibbs.kappa):
    if k % 10 == 0:
        print('gibbs_index = {}'.format(k+1))
    beta_sum_1_pre, beta_sum_0_pre = EEGGibbs.compute_stratified_beta_sum(
        beta_iter_k, beta_1_indices, beta_0_indices)

    delta_1_post, delta_0_post = EEGGibbs.update_delta_post(
        pres_mat_1_mcmc[k, ...], pres_mat_0_mcmc[k, ...],
        beta_sum_1_pre, beta_sum_0_pre)

    out_prod_1_pre, out_prod_0_pre = EEGGibbs.compute_outer_prod(
        beta_iter_k, delta_1_post, delta_0_post, beta_1_indices, beta_0_indices)
    pres_1_post, pres_0_post = EEGGibbs.update_precision_matrix_post(
        out_prod_1_pre, out_prod_0_pre)

    beta_iter_k_plus = []
    for letter_id in range(gc.letter_dim):
        # print('letter_id = {}'.format(letter_id+1))
        letter_low = letter_id * gc.trn_repetition * gc.num_rep
        letter_upp = letter_low + gc.trn_repetition * gc.num_rep
        beta_let_iter_k = np.concatenate([append_zeros, beta_iter_k[letter_low:letter_upp, ...],
                                          append_zeros], axis=0)
        # print('beta_let_iter_k has shape {}'.format(beta_let_iter_k.shape))
        beta_let_iter_k_plus = append_zeros
        for rep_id in range(gc.n_multiple-1, gc.trn_repetition*gc.num_rep+gc.n_multiple-1):
            # print('rep_id = {}'.format(rep_id))
            eeg_signal_slice = EEGGibbs.extract_eeg_signal_slice(
                eeg_signals, letter_id, rep_id - gc.n_multiple + 1)
            beta_slice_pre = beta_let_iter_k_plus[(1-gc.n_multiple):, ...]
            # print('beta_slice_pre has shape {}'.format(beta_slice_pre.shape))
            beta_slice_post = beta_let_iter_k[(rep_id+1): (rep_id+gc.n_multiple), ...]
            # print('beta_slice_post has shape {}'.format(beta_slice_post.shape))

            cond_beta_let_rep = EEGGibbs.compute_mu_conditional_beta(
                dm_cond_beta, eeg_signal_slice, beta_slice_pre, beta_slice_post)
            type_id = letter_id*gc.trn_repetition*gc.num_rep + rep_id-gc.n_multiple+1
            if eeg_type[type_id] == 1:
                beta_let_rep_iter_k_plus = EEGGibbs.update_individual_beta(
                    pres_1_post, pres_lambda_mcmc[k, ...], delta_1_post, cond_beta_let_rep)
            else:
                beta_let_rep_iter_k_plus = EEGGibbs.update_individual_beta(
                    pres_0_post, pres_lambda_mcmc[k, ...], delta_0_post, cond_beta_let_rep)
            beta_let_iter_k_plus = np.concatenate([beta_let_iter_k_plus,
                                                   beta_let_rep_iter_k_plus[np.newaxis, ...]], axis=0)
            # print('beta_let_iter_k_plus has shape {}'.format(beta_let_iter_k_plus.shape))
        beta_iter_k_plus.append(beta_let_iter_k_plus)
    beta_iter_k_plus = np.stack(beta_iter_k_plus, axis=0)
    beta_iter_k_plus = beta_iter_k_plus[:, (gc.n_multiple-1):, ...]
    # print('beta_iter_k_plus has shape {}'.format(beta_iter_k_plus.shape))

    predicted_signals = EEGGibbs.create_predicted_signals(
        dm_beta, beta_iter_k_plus)
    lambda_post = EEGGibbs.update_precision_lambda_post(
        gc.letter_dim, eeg_signals, predicted_signals)
    # print('lambda_post has shape {} and dtype {}'.format(lambda_post.shape, lambda_post.dtype))

    # Append the result of each iteration
    delta_1_mcmc = np.concatenate([delta_1_mcmc, delta_1_post[np.newaxis, ...]], axis=0)
    delta_0_mcmc = np.concatenate([delta_0_mcmc, delta_0_post[np.newaxis, ...]], axis=0)
    pres_mat_1_mcmc = np.concatenate([pres_mat_1_mcmc, pres_1_post[np.newaxis, ...]], axis=0)
    pres_mat_0_mcmc = np.concatenate([pres_mat_0_mcmc, pres_0_post[np.newaxis, ...]], axis=0)
    beta_iter_k = np.reshape(beta_iter_k_plus, [gc.letter_dim*gc.trn_repetition*gc.num_rep,
                                                gc.num_electrode,
                                                gc.n_length,
                                                1])
    pres_lambda_mcmc = np.concatenate([pres_lambda_mcmc, lambda_post[np.newaxis, ...]], axis=0)

burn_in = 11
gibbs_params_list = [
    delta_1_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),
    delta_0_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),
    pres_mat_1_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),
    pres_mat_0_mcmc[burn_in:, ...].astype(gc.DAT_TYPE),
    pres_lambda_mcmc[burn_in:, ...].astype(gc.DAT_TYPE)
]

EEGWLSOpt.save_gibbs_params_est(gibbs_params_list, gc.message)

'''
start_time = time.time()
with tf.Graph().as_default() as g:
    EEGGibbs = GibbsSample(
        n_multiple=gc.n_multiple,
        num_electrode=gc.num_electrode,
        flash_and_pause_length=gc.flash_and_pause_length,
        mu_1_delta=np.zeros([gc.num_electrode, gc.n_length, 1]).astype(gc.DAT_TYPE),
        mu_0_delta=np.zeros([gc.num_electrode, gc.n_length, 1]).astype(gc.DAT_TYPE),
        sigma_sq_delta=1.0,
        nu_1=gc.n_length + 1, nu_0=gc.n_length + 1,
        scale_1=np.tile(np.eye(gc.n_length, gc.n_length)[np.newaxis, ...], [gc.num_electrode, 1, 1]),
        scale_0=np.tile(np.eye(gc.n_length, gc.n_length)[np.newaxis, ...], [gc.num_electrode, 1, 1]),
        alpha=5.0, beta=2.0,
        kappa=gc.kappa,
        letter_dim=gc.letter_dim,
        trn_repetition=gc.trn_repetition
    )

    id_beta_ph = tf.compat.v1.placeholder(tf.int32,
                                          shape=[gc.letter_dim*gc.trn_repetition*gc.num_rep])
    dm_cond_beta_ph = tf.compat.v1.placeholder(tf.float32, shape=[2*gc.n_multiple-2,
                                                                  gc.num_electrode,
                                                                  gc.n_length,
                                                                  gc.n_length])
    dm_beta_ph = tf.compat.v1.placeholder(tf.float32, shape=[gc.letter_dim,
                                                             gc.num_electrode,
                                                             gc.trn_total_seq_length,
                                                             gc.trn_repetition*gc.num_rep*gc.n_length])
    eeg_signals_ph = tf.compat.v1.placeholder(
        tf.float32, shape=[gc.letter_dim,
                           gc.num_electrode,
                           gc.trn_total_seq_length,
                           1])

    id_beta_tf = tf.Variable(id_beta_ph)
    dm_cond_beta_tf = tf.Variable(dm_cond_beta_ph)
    dm_beta_tf = tf.Variable(dm_beta_ph)
    eeg_signals_tf = tf.Variable(eeg_signals_ph)

    

# dm_cond_beta_ = EEGGibbs.create_partial_design_matrix_bayes()
# dm_beta_ = EEGGibbs.create_design_matrix_bayes()
with tf.compat.v1.Session(graph=g) as gibbs_sess:
    id_beta_ = EEGWLSOpt.rearrange.create_permute_beta_id(
        gc.letter_dim, gc.trn_repetition, eeg_type)
    dm_cond_beta_ = EEGGibbs.create_partial_design_matrix_bayes()
    dm_beta_ = EEGGibbs.create_design_matrix_bayes()
    gibbs_sess.run(tf.compat.v1.global_variables_initializer(),
                   feed_dict={
                       id_beta_ph: id_beta_,
                       dm_cond_beta_ph: dm_cond_beta_,
                       dm_beta_ph: dm_beta_,
                       eeg_signals_ph: eeg_signals[..., np.newaxis]
                   })
    [delta_1_mcmc_, delta_0_mcmc_,
     pres_mat_1_mcmc_, pres_mat_0_mcmc_,
     pres_lambda_mcmc_] = gibbs_sess.run([delta_1_mcmc, delta_0_mcmc,
                                          pres_mat_1_mcmc, pres_mat_0_mcmc,
                                          pres_lambda_mcmc])
end_time = time.time()
print("Evaluating tf costs another %g seconds" % (end_time - start_time))

lambda_mat = np.tile(np.eye(25)[np.newaxis, ...], [2,3,4,1,1])
eta_vec = np.random.normal(size=[2,3,4,25])
[mean_vec, sigma_mat_half] = EEGGibbs.recover_normal_canonical_form(
    lambda_mat, eta_vec)
    
delta_1_post, delta_0_post = EEGGibbs.update_delta_post(
    eeg_t_cov, eeg_nt_cov,
    np.random.uniform(size=[gc.num_electrode, gc.n_length, 1]).astype(gc.DAT_TYPE),
    np.random.uniform(size=[gc.num_electrode, gc.n_length, 1]).astype(gc.DAT_TYPE)
)
print('delta_1_post has shape {}'.format(delta_1_post.shape))
print('delta_0_post has shape {}'.format(delta_0_post.shape))

pres_1_post, pres_0_post = EEGGibbs.update_precision_matrix_post(eeg_t_cov, eeg_nt_cov)
print('pres_1_post has shape {} and dtype {}'.format(pres_1_post.shape, pres_1_post.dtype))
print('pres_0_post has shape {}'.format(pres_0_post.shape))

beta_sum_1, beta_sum_0 = EEGGibbs.compute_stratified_beta_sum(
    np.random.uniform(size=[720,16,25,1]),
    np.where(eeg_type==1)[0],
    np.where(eeg_type==0)[0]
)

print('beta_sum_1 has shape {}'.format(beta_sum_1.shape))
print('beta_sum_0 has shape {}'.format(beta_sum_0.shape))

out_prod_1, out_prod_0 = EEGGibbs.compute_outer_prod(
    np.random.uniform(size=[3420,16,25,1]), delta_1_post, delta_0_post,
    np.where(eeg_type==1)[0], np.where(eeg_type==0)[0]
)
print('out_prod_1 has shape {}'.format(out_prod_1.shape))
print('out_prod_0 has shape {}'.format(out_prod_0.shape))

lambda_post = EEGGibbs.update_precision_lambda_post(
    gc.letter_dim, eeg_signals[..., np.newaxis], 0.9*eeg_signals[..., np.newaxis])
print('lambda_post has shape {} and dtype {}'.format(lambda_post.shape, lambda_post.dtype))

dm = EEGGibbs.create_partial_design_matrix_bayes()

indi_beta = EEGGibbs.update_individual_beta(pres_1_post, lambda_post, delta_1_post, delta_0_post)
print('indi_beta has shape {}'.format(indi_beta.shape))

mu_conditional_beta = EEGGibbs.compute_mu_conditional_beta(
    np.ones([8, 16, 25, 25], dtype=gc.DAT_TYPE),
    np.random.normal(size=[16, 25, 1]).astype(gc.DAT_TYPE),
    np.random.normal(size=[4, 16, 25, 1]).astype(gc.DAT_TYPE),
    np.random.normal(size=[4, 16, 25, 1]).astype(gc.DAT_TYPE)
)
print('mu_conditional_beta has shape {}'.format(mu_conditional_beta.shape))

gamma_1 = np.random.binomial(1,0.5,[16,25])
radius = 2
pi_1_post, _ = EEGGibbs.update_pi_post(gamma_1, gamma_1, radius)
print('pi_1_post has shape {}'.format(pi_1_post.shape))
print(pi_1_post[0,:])
'''