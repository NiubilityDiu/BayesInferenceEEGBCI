from self_py_fun.BayesGenFun import *
import self_py_fun.GlobalSIM as sg

np.random.seed(sg.seed_index)
q_mcmc = 2
method_name = 'BayesGenq' + str(q_mcmc)
sim_name_short = 'sim_' + str(sg.design_num + 1)
thinning = 2
reshape_option = sg.sim_type  # 'super_seq'
homo_var_bool = True
mean_fn_type = sg.mean_fn_type
scenario_name = sg.scenario_name
# kernel_option = 'gamma_exp'
kernel_option = sys.argv[6]
# zeta_0 = 0.5
zeta_0 = float(sys.argv[7])
single_pred_bool = False
sim_dat_bool = True
# rho_level_num = 8
rho_level_num = int(sys.argv[8])


BayesGenSeqObj = BayesGenSeq(
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

rho_set, rho_level, _, _, _ = BayesGenSeqObj.produce_pre_compute_rhos(q_mcmc, 10, rho_level_num)
job_id_dec_2 = 'binary_down_{}_zeta_{}'.format(kernel_option, zeta_0)

[s_x_sq_fix, arg_rho_fix, _, _,
 beta_tar_mcmc, beta_ntar_mcmc,
 scale_opt, var_opt, log_lkd] = BayesGenSeqObj.import_mcmc(
    sg.sim_type, method_name,
    sg.repet_num_fit, scenario_name, job_id_dec_2
)
s_x_sq_fix = np.squeeze(s_x_sq_fix, axis=-1)
arg_rho_fix = np.squeeze(arg_rho_fix, axis=-1)
rho_fix = [rho_set[arg_rho_fix[e]] for e in range(sg.NUM_ELECTRODE)]
r_fix = 1.0
beta_comb_mcmc = np.concatenate([beta_tar_mcmc, beta_ntar_mcmc], axis=-2)
mcmc_num = beta_comb_mcmc.shape[0]

log_prior_prob = np.log(1 / 36 * np.ones([36]))
prob_dist_total = []
letter_dist_total = []
target_letter_rank = []
letter_dim = sg.LETTER_DIM
# letter_dim = 2
repet_num = sg.REPETITION_TEST
# repet_num = 2

# Classification on testing set for each repet_num
[_, _, _, _, _,
 signals_test, code_test_3d, type_test_3d,
 _] = BayesGenSeqObj.import_sim_bayes_gen_dataset(
    sg.LETTER_DIM, sg.REPETITION_TEST,
    sg.sim_type + '_' + scenario_name + "_test", 3
)

# Subset signals, code, and type
super_seq_length = sg.SUPER_SEQ_LENGTH_TEST_FIT
if super_seq_length < signals_test.shape[2]:
    signals_test = signals_test[..., :super_seq_length, :]
    print('We take the subset.')
elif super_seq_length > signals_test.shape[2]:
    super_seq_diff = super_seq_length - signals_test.shape[2]
    signals_test = np.concatenate(
        [signals_test, np.zeros([sg.NUM_ELECTRODE, sg.LETTER_DIM, super_seq_diff, 1])], axis=-2
    )
    print('We append the zeros to the end.')
else:
    print('We do nothing.')
    pass

# Classification
for repet_num_test in range(0, repet_num):
    print('repetitions of testing = {}'.format(repet_num_test + 1))
    d_mat = BayesGenSeqObj.create_design_mat_gen_bayes_seq(repet_num_test + 1)
    super_seq_length_id = d_mat.shape[0]
    # print('signals_test has shape {}'.format(signals_test.shape))
    type_temp_1d = np.tile(np.concatenate([np.ones(2), np.zeros(10)], axis=0), repet_num_test + 1)
    t_mat = BayesGenSeqObj.create_transform_mat(type_temp_1d, 1, repet_num_test + 1, 'super_seq')
    dt_mat = d_mat @ t_mat
    r_mat_neghalf_fix = BayesGenSeqObj.create_ratio_mat_inv_half(r_fix, dt_mat)

    if q_mcmc == 1:
        _, pres_chky_t, _ = BayesGenSeqObj.create_std_ar1_pres_candidate(
            rho_fix, super_seq_length_id
        )
    else:
        _, pres_chky_t, _ = BayesGenSeqObj.create_std_ar2_pres_candidate(
            rho_fix, super_seq_length_id
        )
    eta_repet_fix = np.zeros([sg.NUM_ELECTRODE, super_seq_length_id, 1])

    for letter_index in range(letter_dim):
        print('Correct letter = {}'.format(sg.LETTERS[letter_index]))
        letter_test_table = []
        for i in range(0, mcmc_num, thinning):
            log_lkd_test, letter_test = BayesGenSeqObj.bayes_generate_pred_latency_full(
                beta_comb_mcmc[i, ...], eta_repet_fix,
                s_x_sq_fix, pres_chky_t, r_mat_neghalf_fix,
                code_test_3d[letter_index, np.newaxis, :repet_num_test + 1, :],
                signals_test[:, letter_index, np.newaxis, :super_seq_length_id, ...],
                d_mat, 1, repet_num_test + 1,
                reshape_option, True, 0, None, log_prior_prob
            )  # For prediction, we should always stick with normal assumption
            letter_test_table.append(letter_test)
        letter_test_table = np.stack(letter_test_table, axis=0)

        unique_test, counts_test = np.unique(letter_test_table, return_counts=True)
        test_reorder = unique_test[np.argsort(counts_test)[::-1]]
        print('The first {} possible letters = \n {}'.format(
            min(len(test_reorder), 11), test_reorder[:11])
        )
        letter_dist_total.append(test_reorder[0])

        order_repet_num = np.where(test_reorder == sg.LETTERS[letter_index].upper())[0]
        if len(order_repet_num) == 0:
            target_letter_rank.append(12)  # assume it is ranking no.12.
        else:
            target_letter_rank.append(order_repet_num[0] + 1)  # index from 1 for reading

        letter_dist_table = np.stack(
            [np.sum(letter_test_table == l) for i, l in enumerate(BayesGenSeqObj.letter_table)])
        letter_dist_table = letter_dist_table + 0.1  # In case of 0's
        letter_dist_table = letter_dist_table / np.sum(letter_dist_table)
        # log_prior_prob = np.log(letter_dist_table)
        print('posterior letter prob = \n {}'.format(np.reshape(letter_dist_table, [6, 6])))

        prob_dist_total.append(letter_dist_table)
        # log_prior_prob = np.log(1 / 36 * np.ones([36]))  # scale to zero for a new letter

prob_dist_total = np.transpose(np.reshape(
    np.stack(prob_dist_total, axis=0), [repet_num, letter_dim, 36]), [1, 0, 2]
)
letter_dist_total = np.reshape(
    np.stack(letter_dist_total, axis=0), [repet_num, letter_dim]
).T
target_letter_rank = np.reshape(
    np.stack(target_letter_rank, axis=0), [repet_num, letter_dim]
).T

bayes_result_dict = {
    'cum_max_letter': letter_dist_total,
    'cum_dist': prob_dist_total,
    'target_letter_rank':  target_letter_rank,
    'sample_num': int(np.ceil(mcmc_num/thinning)),
    'scale': scale_opt,
    'var': var_opt,
    'screen_ids': None
}

BayesGenSeqObj.save_bayes_results(
    bayes_result_dict, sg.repet_num_fit, repet_num, method_name,
    sg.sim_type+'_test', sg.LETTERS[:letter_dim], scenario_name,
    job_id_dec_2, single_pred_bool, sim_dat_bool
)
