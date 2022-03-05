from self_py_fun.CNNMultiDataloader import *
from self_py_fun.CNNMultiTrain import *
from self_py_fun.CNNMultiModel import *
from self_py_fun.ExistMLFun import *
import self_py_fun.GlobalSIM as sg
sns.set_context('notebook')

#----------------------------------------------
if sg.local_use:
    DATA_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data/SIM_files'
    SAVE_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data/SIM_files/'
else:
    DATA_DIC = '/home/mtianwen/EEG_MATLAB_data/SIM_files'
    SAVE_DIC = '/home/mtianwen/EEG_MATLAB_data/SIM_files/'
#----------------------------------------------


def get_neuralnetwork_results(
        simnum,
        datasetnum,
        length = 30,
        out_channels0 = 10,
        out_channels = 100,
        kernel_size = 3,
        maxpool_size = 5,
        linear_dim = 20,
        val_proportion = 0.2,
        epochs = 200,
        batch_size = 50,
        lr = 0.0005,
        datadic = DATA_DIC,
        savedic = SAVE_DIC,
        sim_bool = True
):
    module = EEGBCIModule(datadic, savedic, simnum, datasetnum, length, out_channels0, out_channels,
                          kernel_size, maxpool_size, linear_dim,
                          val_proportion, epochs, batch_size, lr, sim_bool)
    return {
        'train_loss': module.loss_train,
        'train_accuracy': module.acc_train,
        'test_loss': module.loss_test,
        'test_accuracy': module.acc_test,
        'train_probability': module.train_pb,
        'test_probability': module.test_pb
    }


def tune_parameter(simnum = 11, 
                   datasetnum = 1, 
                   oc0list = [2,5,10],
                   oclist = [20,50,100], 
                   kslist = [3,6,9,12,20], 
                   mslist = [3,5,7,10], 
                   ldlist = [10,20,50], 
                   datadic = DATA_DIC, 
                   savedic = SAVE_DIC,
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
                        
                        a = EEGBCIModule(datadic = datadic,
                                         savedic = savedic,
                                         design_num= simnum,
                                         dataset_num= datasetnum,
                                         length=sg.N_LENGTH_FIT,
                                         out_channels = oc,
                                         out_channels0=oc0,
                                         kernel_size = ks,
                                         maxpool_size = ms,
                                         linear_dim = ld,
                                         sim_bool=sim_bool)

                        if a.min_val_loss < minloss:
                            minloss = a.min_val_loss
                            bestpar_oc0 = oc0
                            bestpar_oc = oc
                            bestpar_ks = ks
                            bestpar_ms = ms
                            bestpar_ld = ld
                            acc = a.best_val_acc
                        print(bestpar_oc0,bestpar_oc,bestpar_ks,bestpar_ms,bestpar_ld, acc, minloss)
    return {
        'best_oc0':bestpar_oc0,
        'best_oc':bestpar_oc,
        'best_ks':bestpar_ks,
        'best_ms':bestpar_ms,
        'best_ld':bestpar_ld,
        'best_validation_accuracy': acc,
        'minimum_loss': float(minloss.detach().numpy())
    }


# Main code starts here.
sim_dat_bool = True
sim_name_short = 'sim_' + str(sg.design_num + 1)
method_name = 'EEGCNN'
sim_type_2 = sg.sim_type + '_down_' + sg.scenario_name


results = get_neuralnetwork_results(sg.design_num + 1, sg.subset_num + 1)
# print(results.keys())
train_loss = results['train_loss']
train_accuracy = results['train_accuracy']
test_loss = results['test_loss']
test_accuracy = results['test_accuracy']
train_prob = results['train_probability']
test_prob = results['test_probability']

# Convert binary probs to letter-based probs
NNTrain = ExistMLPred(
    data_type='SIM_files',
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    num_repetition=sg.repet_num_fit,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE_FIT,
    local_bool=sg.local_use
)


[_, _, eeg_code_1d] = NNTrain.import_sim_ml_trunc_dataset(
        sim_type_2 + '_train',
        sg.LETTER_DIM, sg.REPETITION_TRN, 3
)

eeg_code_1d = np.reshape(
        eeg_code_1d[:, :sg.repet_num_fit, :],
        [1, sg.LETTER_DIM * sg.repet_num_fit * sg.NUM_REP]
)
nn_pred_dict = NNTrain.ml_predict(
        train_prob[:, np.newaxis], eeg_code_1d, sg.LETTER_DIM, sg.repet_num_fit
)
NNTrain.save_exist_ml_pred_results(
    nn_pred_dict, sg.repet_num_fit, sg.repet_num_fit, sg.LETTERS,
    method_name, sg.scenario_name, 'down', True, sim_dat_bool
)

NNTest = ExistMLPred(
    data_type='SIM_files',
    sub_folder_name=sg.sim_common,
    sub_name_short=sim_name_short,
    num_repetition=sg.REPETITION_TEST,
    num_electrode=sg.NUM_ELECTRODE,
    flash_and_pause_length=sg.FLASH_PAUSE_LENGTH,
    num_letter=sg.LETTER_DIM,
    n_multiple=sg.N_MULTIPLE_FIT,
    local_bool=sg.local_use
)

# Test combined
[signals_trun_test, _, eeg_code_test_1d] = NNTest.import_sim_ml_trunc_dataset(
    sim_type_2 + '_test',
    sg.LETTER_DIM, sg.REPETITION_TEST, 3
)
nn_pred_dict_test = NNTest.ml_predict(
    test_prob[:, np.newaxis], eeg_code_test_1d, sg.LETTER_DIM, sg.REPETITION_TEST
)
NNTest.save_exist_ml_pred_results(
    nn_pred_dict_test, sg.repet_num_fit, sg.REPETITION_TEST,
    sg.LETTERS, method_name, sg.scenario_name, 'down', False, sim_dat_bool
)
