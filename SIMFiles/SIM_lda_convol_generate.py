import sys
sys.path.insert(0, './self_py_fun')
from self_py_fun.ConvolFun import *
import self_py_fun.SIMConvolGlobal as sg
print(os.getcwd())
# tf.compat.v1.random.set_random_seed(612)
# np.random.seed(612)

sim_ids = 2020010200 + np.array([1, 2, 3, 4, 5]) + 15
num_electrodes = 2 * np.array([1, 1, 1, 1, 1])

for i, sim_id in enumerate(sim_ids):
    sim_name = 'sim_' + str(sim_id)
    print(sim_name)

    PData = EEGGeneralFun(
        num_repetition=sg.num_repetition, num_electrode=num_electrodes[i],
        flash_and_pause_length=sg.flash_and_pause_length,
        num_letter=sg.letter_dim,
        n_multiple=sg.n_multiple
    )
    CPrior = PriorModel(n_length=sg.n_length, num_electrode=num_electrodes[i])
    CArrange = ReArrangeBetaSigma(
        n_multiple=sg.n_multiple, num_electrode=sg.num_electrode,
        flash_and_pause_length=sg.flash_and_pause_length)

    # Borrow direct signals from SIM_lda_generate.py
    [beta_tar, beta_ntar,
     cov_mat, _,
     signals, eeg_code, eeg_type,
     message] = PData.import_simulation_results(sim_name, reshape_to_3d=False, as_pres=False)

    # Reshape signals to satisfy the requirement of matrix operation:
    signals = np.transpose(signals, [0, 2, 1, 3, 4])
    signals = np.reshape(signals, [sg.letter_dim,
                                   num_electrodes[i],
                                   sg.num_repetition * sg.num_rep * sg.n_length,
                                   1])

    print('signals have shape {}'.format(signals.shape))
    design_x = CArrange.create_design_matrix_bayes(sg.letter_dim, sg.num_repetition,
                                                   channel_dim=num_electrodes[i])
    print('design_x has shape {}'.format(design_x.shape))

    # Since previously the signals have already been permuted, we don't need to permute it again.
    beta_tilta = CArrange.create_joint_beta_tilta(
        sg.letter_dim, sg.num_repetition, signals,
        id_beta=None, design_x=design_x, channel_dim=num_electrodes[i]
    )
    print('convoluted sequence has shape {}'.format(beta_tilta.shape))

    # Preview the convoluted signals, and mark the eeg_type:
    PData.save_simulation_results(
        sim_name + '_convol',
        beta_tar, beta_ntar,
        cov_mat, cov_mat,
        beta_tilta,
        eeg_code=eeg_code, eeg_type=eeg_type,
        convol=True, as_pres=False, save_plots=True,
        message=message + '_convol'
    )

