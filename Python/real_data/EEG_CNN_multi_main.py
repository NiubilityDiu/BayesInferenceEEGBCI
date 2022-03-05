from self_py_fun.CNNMultiDataloader import *
from self_py_fun.CNNMultiTrain import *
from self_py_fun.CNNMultiModel import *
from self_py_fun.ExistMLFun import *
import self_py_fun.GlobalEEG as gc
sns.set_context('notebook')

# ----------------------------------------------
if gc.local_use:
    DATA_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data/TRN_files'
    SAVE_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data/TRN_files/'
else:
    DATA_DIC = '/home/mtianwen/EEG_MATLAB_data/TRN_files'
    SAVE_DIC = '/home/mtianwen/EEG_MATLAB_data/TRN_files/'

SAVE_DIC = SAVE_DIC + 'K' + str(gc.K_num) + '/'


# ----------------------------------------------


def get_neuralnetwork_results(
        simnum,
        datasetnum,  # id for sim_data, dec_factor for real data
        length,
        out_channels0=10,
        out_channels=100,
        kernel_size=3,
        maxpool_size=5,
        linear_dim=20,
        val_proportion=0.2,
        epochs=200,
        batch_size=50,
        lr=0.0005,
        datadic=DATA_DIC,
        savedic=SAVE_DIC,
        bp_upp=6,
        channel_id=None,
        sim_bool=False
):
    module = EEGBCIModule(
        datadic, savedic, simnum, datasetnum, length, out_channels0, out_channels,
        kernel_size, maxpool_size, linear_dim,
        val_proportion, epochs, batch_size, lr, bp_upp, channel_id, sim_bool
    )
    return {
        'train_loss': module.loss_train,
        'train_accuracy': module.acc_train,
        'test_loss': module.loss_test,
        'test_accuracy': module.acc_test,
        'train_probability': module.train_pb,
        'test_probability': module.test_pb
    }


def tune_parameter(simnum=11,
                   datasetnum=1,
                   oc0list=[2, 5, 10],
                   oclist=[20, 50, 100],
                   kslist=[3, 6, 9, 12, 20],
                   mslist=[3, 5, 7, 10],
                   ldlist=[10, 20, 50],
                   datadic=DATA_DIC,
                   savedic=SAVE_DIC,
                   sim_bool=True):
    bestpar_oc0 = 0
    bestpar_oc = 0
    bestpar_ks = 0
    bestpar_ms = 0
    bestpar_ld = 0
    minloss = 100
    acc = 0
    for oc0 in oc0list:
        for oc in oclist:
            for ks in kslist:
                for ms in mslist:
                    for ld in ldlist:

                        a = EEGBCIModule(datadic=datadic,
                                         savedic=savedic,
                                         design_num=simnum,
                                         dataset_num=datasetnum,
                                         length=gc.N_LENGTH,
                                         out_channels=oc,
                                         out_channels0=oc0,
                                         kernel_size=ks,
                                         maxpool_size=ms,
                                         linear_dim=ld,
                                         sim_bool=sim_bool)

                        if a.min_val_loss < minloss:
                            minloss = a.min_val_loss
                            bestpar_oc0 = oc0
                            bestpar_oc = oc
                            bestpar_ks = ks
                            bestpar_ms = ms
                            bestpar_ld = ld
                            acc = a.best_val_acc
                        print(bestpar_oc0, bestpar_oc, bestpar_ks, bestpar_ms, bestpar_ld, acc, minloss)
    return {
        'best_oc0': bestpar_oc0,
        'best_oc': bestpar_oc,
        'best_ks': bestpar_ks,
        'best_ms': bestpar_ms,
        'best_ld': bestpar_ld,
        'best_validation_accuracy': acc,
        'minimum_loss': float(minloss.detach().numpy())
    }


# Main code starts here.
rep_num_fit = len(gc.rep_odd_id)
rep_num_pred = len(gc.rep_even_id)
bp_low = 0.5
bp_upp = float(sys.argv[4])
# bp_upp = 6
if float(bp_upp) == int(bp_upp):
    bp_upp = int(bp_upp)
else:
    bp_upp = float(bp_upp)
if bp_upp < 0:
    eeg_file_suffix = 'raw'
else:
    eeg_file_suffix = 'raw_bp_{}_{}'.format(bp_low, bp_upp)

file_subscript = 'ML_down_{}'.format(gc.DEC_FACTOR)
sim_dat_bool = False
method_name = 'EEGCNN'
file_subscript = file_subscript + '_from_' + eeg_file_suffix
file_subscript_2 = 'ML_down_{}_{}'.format(gc.DEC_FACTOR, eeg_file_suffix)

# print('We are using channel {} for prediction.'.format(gc.channel_id + 1))
channel_id_str = [str(e + 1) for e in gc.channel_id]
if len(gc.channel_id) == 16:
    channel_name = 'all_channels'
else:
    channel_name = 'channel_' + '_'.join(channel_id_str)

results = get_neuralnetwork_results(
    simnum=gc.K_num, datasetnum=gc.DEC_FACTOR, length=gc.N_LENGTH,
    channel_id=gc.channel_id, bp_upp=bp_upp
)

# print(results.keys())
train_loss = results['train_loss']
train_accuracy = results['train_accuracy']
test_loss = results['test_loss']
test_accuracy = results['test_accuracy']
train_prob = results['train_probability']
test_prob = results['test_probability']


# Convert binary probs to letter-based probs
NNTrain = ExistMLPred(
    data_type=gc.DATA_TYPE,
    sub_folder_name=gc.sub_file_name,
    sub_name_short=gc.sub_file_name[:4],
    num_repetition=gc.NUM_REPETITION,
    num_electrode=gc.NUM_ELECTRODE,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)

[_, eeg_code_odd, _] = NNTrain.import_eeg_odd_even_dat(
    file_subscript + '_odd'
)
print('eeg_code_odd_1d has shape {}'.format(eeg_code_odd.shape))
# channel_name = 'channel_{}'.format(gc.channel_id + 1)
# print(channel_name)

nn_pred_dict = NNTrain.ml_predict(
    train_prob[:, np.newaxis], eeg_code_odd, gc.LETTER_DIM, rep_num_fit
)
NNTrain.save_exist_ml_pred_results(
    nn_pred_dict, rep_num_fit, rep_num_fit, gc.TARGET_LETTERS,
    method_name, channel_name, file_subscript_2,
    train_bool=True, sim_dat_bool=sim_dat_bool
)


# Test
NNTest = ExistMLPred(
    data_type=gc.DATA_TYPE,
    sub_folder_name=gc.sub_file_name,
    sub_name_short=gc.sub_file_name[:4],
    num_repetition=rep_num_pred,
    num_electrode=gc.NUM_ELECTRODE,
    flash_and_pause_length=gc.FLASH_AND_PAUSE_LENGTH,
    num_letter=gc.LETTER_DIM,
    n_multiple=gc.N_MULTIPLE,
    local_bool=gc.local_use
)

# Test combined
[_, eeg_code_even, _] = NNTest.import_eeg_odd_even_dat(
    file_subscript + '_even'
)
print('eeg_code_even_1d has shape {}'.format(eeg_code_even.shape))

nn_pred_dict_test = NNTest.ml_predict(
    test_prob[:, np.newaxis], eeg_code_even, gc.LETTER_DIM, rep_num_pred
)
NNTest.save_exist_ml_pred_results(
    nn_pred_dict_test, rep_num_fit, rep_num_pred,
    gc.TARGET_LETTERS, method_name, channel_name, file_subscript_2,
    train_bool=False, sim_dat_bool=sim_dat_bool
)