import sys
sys.path.insert(0, './self_py_fun')
import self_py_fun.EEGConvolGlobal as gc
from self_py_fun.swLDAFun import *
# print(os.getcwd())
# tf.compat.v1.random.set_random_seed(612)
np.random.seed(612)

EEGswLDAObj = SWLDAPred(
    data_type=gc.data_type,
    sub_folder_name=gc.sub_file_name,
    # EEGGeneralFun class
    sampling_rate=gc.sampling_rate,
    num_repetition=gc.num_repetition,
    num_electrode=gc.num_electrode,
    flash_and_pause_length=gc.flash_and_pause_length,
    p300_flash_strength=gc.p300_flash_strength,
    p300_pause_strength=gc.p300_pause_strength,
    non_p300_strength=gc.non_p300_strength,
    num_letter=gc.num_letter,
    n_multiple=gc.n_multiple,
    local_bool=gc.local_use
)

# Import the training set without subset
[eeg_signals, eeg_code,
 eeg_type] = EEGswLDAObj.import_eeg_processed_dat(
    gc.file_subscript, reshape_to_1d=False)

print('eeg_signals have shape {}'.format(eeg_signals.shape))
print('eeg_code has shape {}'.format(eeg_code.shape))
print('eeg_type has shape {}'.format(eeg_type.shape))

for trn_repetition in range(15, 16):
    EEGswLDAObj.print_sub_trn_info(trn_repetition)

    # Produce truncated eeg signals subset
    eeg_signals_trun_sub, eeg_type_sub = EEGswLDAObj.create_truncate_segment_batch(
        np.squeeze(eeg_signals, axis=-1), eeg_type, letter_dim=gc.num_letter,
        trn_repetition=trn_repetition)

    # Import swlda wts
    swlda_wts_i = EEGswLDAObj.import_matlab_swlda_wts_trn_repetition(
        trn_repetition, gc.file_subscript)

    b = swlda_wts_i['b']
    inmodel = swlda_wts_i['inmodel']
    # pval = swlda_wts_i['pval']
    # se = swlda_wts_i['se']
    # stats = swlda_wts_i['stats']

    EEGswLDAObj.plot_swlda_select_feature(inmodel, gc.sub_file_name[:4], trn_repetition)

    print('The number of features selected by swLDA is {}'.format(np.sum(inmodel[0, :])))
    print('swLDA prediction 1:\n')

    # b_inmodel = b[:, 0] * inmodel[0, :]
    # b_inmodel = np.reshape(b_inmodel, [16, 25])
    # for i in range(gc.num_electrode):
    #     plt.figure()
    #     plt.plot(b_inmodel[i, :])
    #     plt.title('Channel {}'.format(i+1))
    #     plt.show()

    [eeg_signals_mean_1_sub,
     eeg_signals_mean_0_sub,
     eeg_signals_cov_sub] = EEGswLDAObj.compute_selected_sample_stats(
        eeg_signals_trun_sub, eeg_type_sub, inmodel
    )
    print(eeg_signals_mean_1_sub.shape)
    print(eeg_signals_mean_0_sub.shape)

    # eeg_signals_trun_sub_2, _ = EEGswLDAObj.create_truncate_segment_batch(
    #     np.squeeze(eeg_signals, axis=-1), None, letter_dim=gc.num_letter,
    #     trn_repetition=gc.num_repetition)
    # section_num_2, _, _ = eeg_signals_trun_sub_2.shape
    #
    # # print('eeg_signals_trun_sub_2 has shape {}'.format(eeg_signals_trun_sub_2.shape))
    # eeg_signals_trun_sub_2 = np.reshape(eeg_signals_trun_sub_2,
    #                                     [section_num_2,
    #                                      gc.num_electrode * gc.n_length])
    # eeg_signals_trun_sub_2 = eeg_signals_trun_sub_2[:, inmodel[0, :] == 1]
    #
    # pred_letter_mat_log_prob = EEGswLDAObj.swlda_produce_two_step_estimation(
    #     eeg_signals_trun_sub_2, eeg_code,
    #     eeg_signals_mean_1_sub, eeg_signals_mean_0_sub, eeg_signals_cov_sub, trn_repetition
    # )
    # swlda_accuracy_log_prob = [np.around(np.mean(pred_letter_mat_log_prob[:, i] == np.array(list(gc.target_letters))),
    #                            decimals=2)
    #                            for i in range(gc.num_repetition)]
    # print(swlda_accuracy_log_prob)
    #
    # swLDA prediction directly
    print('swLDA prediction 2:\n')
    eeg_signals_trun, _ = EEGswLDAObj.create_truncate_segment_batch(
        np.squeeze(eeg_signals, axis=-1), None, letter_dim=gc.num_letter,
        trn_repetition=gc.num_repetition)

    # Collapse the electrode dimension
    eeg_signals_trun = np.reshape(eeg_signals_trun,
                                  [gc.num_letter*gc.num_repetition*gc.num_rep,
                                   gc.num_electrode*gc.n_length])

    pred_letter_mat_direct = EEGswLDAObj.swlda_pred(
        b, inmodel, trn_repetition, eeg_signals_trun, eeg_code)

    swlda_accuracy_direct = [np.around(np.mean(pred_letter_mat_direct[:, i] == np.array(list(gc.target_letters))),
                                       decimals=2)
                             for i in range(gc.num_repetition)]
    print(swlda_accuracy_direct)
    # EEGswLDAObj.save_swlda_pred_results(
    #     swlda_accuracy_direct, gc.sub_file_name, trn_repetition)


