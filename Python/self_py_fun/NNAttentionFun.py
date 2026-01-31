import os
import scipy.io as sio
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0)

DATA_DIC = 'data_dic'
SAVE_DIC = 'save_dic'


class EEGBCIData():
    def __init__(self, dic, path, scenario_name):
        self.train = sio.loadmat(dic + path + '/sim_dat_ML_down_' + str(scenario_name) + '_train')
        self.test = sio.loadmat(dic + path + '/sim_dat_ML_down_' + str(scenario_name) + '_test')
        self.train_num = self.train['eeg_signals'].shape[0]
        self.test_num = self.test['eeg_signals'].shape[0]
        self.dim = self.train['eeg_signals'].shape[1]


class EEGBCINNAttention(nn.Module):
    def __init__(self,
                 in_dim = 30,
                 out_channels = 10,
                 kernel_size = 6,
                 maxpool_size = 3,
                 attention_head = 3,
                 linear_dim = 10
                 ):
        """Initialize class parameters.
        Args:
            in_dim: dimension of input EEG signal
            out_channels: number of channels of CNN
            kernel_size: kernel size of CNN
            maxpool_size: kernel size of max pooling
            linear_dim: dimension of output of linear layer
        Return:
            None.
        """
        super(EEGBCINNAttention, self).__init__()
        self.CNN =nn.Conv1d(1, out_channels, kernel_size = kernel_size)
        self.MaxPooling = nn.MaxPool1d(maxpool_size)
        self.tanh = nn.Tanh()
        self.attention = nn.Linear(out_channels * int((in_dim - kernel_size + 1) / maxpool_size), attention_head, bias = False)
        self.Linear1 = nn.Linear(attention_head * out_channels * int((in_dim - kernel_size + 1) / maxpool_size), linear_dim)
        self.Linear2 = nn.Linear(linear_dim, 2)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, EEGBCIData):
        """Forward process.
        Args:
            EEGBCIData: EEGBCI data, a dictionary that contains eeg_type, eeg_signals, etc
        Return:
            The probability of each signal being 0 or 1 (before normalized), A FloatTensor of shape [eeg_number, 2]
        """
        data = torch.FloatTensor(EEGBCIData)
        d1 = data.shape[0]
        d2 = data.shape[1]
        data = data.reshape(data.shape[0] * data.shape[1], 1, data.shape[2])
        ans_CNN = self.CNN(data)
        ans_maxpool = self.MaxPooling(ans_CNN)
        ans_maxpool = self.tanh(ans_maxpool)
        ans_maxpool = ans_maxpool.reshape(ans_maxpool.shape[0], -1)
        ans_maxpool = ans_maxpool.reshape(d1,d2,-1)
        alpha = self.softmax(self.attention(ans_maxpool))
        ans_attention = torch.matmul(alpha.permute(0,2,1), ans_maxpool)
        ans_attention = ans_attention.reshape(ans_attention.shape[0], -1)
        ans_linear1 = self.Linear1(ans_attention)
        ans_linear1 = self.ReLU(ans_linear1)
        ans_linear2 = self.Linear2(ans_linear1)

        return ans_linear2


class EEGBCIAttnModule():
    def __init__(self,
                 datadic,
                 savedic,
                 simnum,
                 datasetnum,
                 scenario_name,
                 out_channels=10,
                 kernel_size=6,
                 maxpool_size=3,
                 attention_head=3,
                 linear_dim=10,
                 val_proportion=0.5,
                 epochs=200,
                 batch_size=10,
                 lr=0.001,
                 ):
        '''Initialize class parameters.
        Args:
            datadic: the path of folder SIM_files
            savedic: the path to save the model
            simnum: the number of sim to get the data
            datasetnum: the number of dataset to get the data
            scenario_name: string, misspecification type
            out_channels: number of channels of CNN
            kernel_size: kernel size of CNN
            maxpool_size: kernel size of max pooling
            linear_dim: dimension of output of linear layer
            val_proportion: The proportion of the validation set
            epochs: Maximum number of epoch trained
            batch_size: batch size
            lr: learning rate
        Return:
            None.
        '''
        self.datadic = datadic
        self.savedic = savedic
        self.simnum = simnum
        self.datasetnum = datasetnum
        self.scenario_name = scenario_name
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.attention_head = attention_head
        self.linear_dim = linear_dim
        self.val_proportion = val_proportion
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sim_common = 'sim_' + str(self.simnum) + '_dataset_' + str(self.datasetnum)
        self.data = EEGBCIData(datadic, '/SIM_files/sim_' + str(self.simnum) + '/' + self.sim_common, self.scenario_name)
        self.dim = self.data.train['eeg_signals'].shape[1]
        self.model = EEGBCINNAttention(self.dim, out_channels, kernel_size, maxpool_size, attention_head, linear_dim)
        self.train(val_proportion, epochs, batch_size, lr)
        self.test()

    def train(self, val_proportion=0.2,
              epochs=200,
              batch_size=10,
              lr=0.001,
              ):
        '''Train the model.
        Args:
            val_proportion: The proportion of the validation set
            epochs: Maximum number of epoch trained
            batch_size: batch size
            lr: learning rate
        Save:
            A torch model file contains the best model.
            A figure of training and validation error
        Return:
            A tuple, minimum validation loss and corresponding validation accuracy
        '''
        # if not os.path.exists(self.savedic + "/NeuralNetworksave"):
        #     os.mkdir(self.savedic + "/NeuralNetworksave")
        # if not os.path.exists(self.savedic + "/NeuralNetworksave/models"):
        #     os.mkdir(self.savedic + "/NeuralNetworksave/models")
        # if not os.path.exists(self.savedic + "/NeuralNetworksave/models/sim_" + str(self.simnum)):
        #     os.mkdir(self.savedic + "/NeuralNetworksave/models/sim_" + str(self.simnum))
        # savefilepath = self.savedic + "/NeuralNetworksave/models/sim_" + str(self.simnum) + '/sim_' + str(
        #     self.simnum) + '_dataset_' + str(self.datasetnum)

        savefilepath = self.savedic + "sim_" + str(self.simnum) + '/' + self.sim_common + '/NNAttn/'

        if not os.path.exists(savefilepath):
            os.mkdir(savefilepath)

        dat = self.data.train['eeg_signals']
        self.train_num = int(dat.shape[0] / 12 / 19)
        code = self.data.train['eeg_code'][0]
        label = self.data.train['eeg_type'][0]
        dat_code = []
        label_code = []
        for i in range(1, 13):
            dt = dat[np.where(code == i)]
            lb = label[np.where(code == i)]
            for j in range(19):
                dat_code.append(dt[self.train_num * j:self.train_num * (j + 1)])
                label_code.append(lb[self.train_num * j])
        dat_code = np.array(dat_code)
        label_code = np.array(label_code)
        np.random.seed(1)
        numbers = list(range(dat_code.shape[0]))
        np.random.shuffle(numbers)
        val_rows = numbers[0:int(val_proportion * len(numbers))]
        train_rows = numbers[int(val_proportion * len(numbers)):]
        train_data = dat_code[train_rows, :, :]
        val_data = dat_code[val_rows, :, :]
        train_label = label_code[train_rows]
        val_label = label_code[val_rows]

        print('train: sim ' + str(self.simnum) + ', dataset ' + str(self.datasetnum),
              'scenario ', self.scenario_name)
        print('method: NN with attention.')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_function = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        # process_bar = tqdm.tqdm(range(epochs))
        process_bar = range(epochs)
        self.train_loss = []
        self.val_loss = []
        self.min_val_loss = 100
        self.best_val_acc = -1
        self.best_epoch = -1
        for epoch in process_bar:
            order = list(range(train_data.shape[0]))
            np.random.shuffle(order)
            for batches in range(int(np.ceil(train_data.shape[0] / batch_size))):
                bt = order[batches * batch_size:(batches + 1) * batch_size]
                train_set = train_data[bt]
                train_lab = train_label[bt]
                self.optimizer.zero_grad()
                pb = self.model(train_set)
                loss = self.loss_function(pb, torch.LongTensor((train_lab + 1) / 2))
                loss.backward()
                self.optimizer.step()
            train_res = self.model(train_data)
            val_res = self.model(val_data)
            loss_train = self.loss_function(train_res, torch.LongTensor((train_label + 1) / 2))
            loss_val = self.loss_function(val_res, torch.LongTensor((val_label + 1) / 2))
            acc_train = 1 - sum(
                ((self.softmax(train_res).detach().numpy()[:, 1] > 0.5) - (train_label + 1) / 2) ** 2) / len(
                train_label)
            acc_val = 1 - sum(((self.softmax(val_res).detach().numpy()[:, 1] > 0.5) - (val_label + 1) / 2) ** 2) / len(
                val_label)
            self.train_loss.append(loss_train.detach().numpy())
            self.val_loss.append(loss_val.detach().numpy())
            # process_bar.set_description("Train Loss: %0.6f, Validation Loss: %0.6f, Train Accuracy: %0.6f, Validation Accuracy: %0.6f" %
            #                             (loss_train,
            #                              loss_val, acc_train, acc_val))
            if self.min_val_loss > loss_val:
                self.min_val_loss = loss_val
                self.best_val_acc = acc_val
                self.best_epoch = epoch
                torch.save(self.model, savefilepath + "/" + "oc%dks%dms%dah%dld%dlr%f.model" % (
                self.out_channels, self.kernel_size, self.maxpool_size, self.attention_head, self.linear_dim, lr))

                self.best_model = torch.load(savefilepath + "/" + "oc%dks%dms%dah%dld%dlr%f.model" % (
                self.out_channels, self.kernel_size, self.maxpool_size, self.attention_head, self.linear_dim, lr))
        # print(sum(sum(self.best_model.attention.weight != self.model.attention.weight)))
        print('Minimum validation loss: %0.6f, Best validation accuracy: %0.6f, best epoch: %d' % (
        self.min_val_loss, self.best_val_acc, self.best_epoch))
        f_train, = plt.plot(self.train_loss, 'r')
        f_val, = plt.plot(self.val_loss, 'b')
        self.fig = plt.legend([f_train, f_val], ['train', 'validation'])
        plt.savefig(savefilepath + "/" + "oc%dks%dms%dah%dld%dlr%f.png" % (
        self.out_channels, self.kernel_size, self.maxpool_size, self.attention_head, self.linear_dim, lr))
        plt.close()
        return (self.min_val_loss, self.best_val_acc)

    def test(self):
        test_dat = self.data.test['eeg_signals']
        self.test_num = int(test_dat.shape[0] / 12 / 19)
        test_code = self.data.test['eeg_code'][0]
        test_label = self.data.test['eeg_type'][0]
        test_dat_code = []
        test_label_code = []
        test_code_code = []
        for i in range(1, 13):
            dt = test_dat[np.where(test_code == i)]
            lb = test_label[np.where(test_code == i)]
            cd = test_code[np.where(test_code == i)]
            for j in range(19):
                test_dat_code.append(dt[self.test_num * j:self.test_num * (j + 1)])
                test_label_code.append(lb[self.test_num * j])
                test_code_code.append(cd[self.test_num * j])
        test_dat_code = np.array(test_dat_code)
        test_label_code = np.array(test_label_code)
        test_code_code = np.array(test_code_code)
        self.loss_test = []
        self.acc_test = []
        self.test_pb = []
        for n in range(1, self.test_num + 1):
            test_res = self.best_model(test_dat_code[:, 0:n, :])
            self.loss_test.append(
                float(self.loss_function(test_res, torch.LongTensor((test_label_code + 1) / 2)).detach().numpy()))
            self.acc_test.append(
                1 - sum(((self.softmax(test_res).detach().numpy()[:, 1] > 0.5) - (test_label_code + 1) / 2) ** 2) / len(
                    test_label_code))
            self.test_pb.append(self.softmax(test_res).detach().numpy()[:, 1])
        self.real_label = []
        self.res_label = []
        for i in range(19):
            self.real_label.append(test_label_code[list(range(i, 228, 19))])
        self.real_label = np.array(self.real_label)
        for i in range(self.test_num):
            pb = self.test_pb[i]
            res_lab = []
            for j in range(19):
                ppb = pb[list(range(j, 228, 19))]
                rl = ppb[0:6]
                cl = ppb[6:12]
                r = np.where(rl == max(rl))[0][0]
                c = np.where(cl == max(cl))[0][0]
                res = [-1] * 12
                res[int(r)] = 1
                res[int(c) + 6] = 1
                res_lab.append(res)
            self.res_label.append(res_lab)
        self.res_label = np.array(self.res_label)
        self.test_acc = []
        for i in range(self.test_num):
            res = self.res_label[i, :, :]
            acc = sum(((res == self.real_label).sum(axis=1)) == 12) / 19
            self.test_acc.append(acc)
        self.test_acc = np.array(self.test_acc)


def get_nn_attention_results(simnum,
                             datasetnum,
                             scenario_name,
                             num_rep_fit,
                             num_rep_pred,
                             out_channels=10,
                             kernel_size=12,
                             maxpool_size=10,
                             attention_head=2,
                             linear_dim=20,
                             val_proportion=0.5,
                             epochs=200,
                             batch_size=10,
                             lr=0.001,
                             datadic=DATA_DIC,
                             savedic=SAVE_DIC):

    module = EEGBCIAttnModule(datadic, savedic, simnum, datasetnum,
                              scenario_name, out_channels, kernel_size,
                              maxpool_size,attention_head, linear_dim,
                              val_proportion, epochs, batch_size, lr)

    real_label = module.real_label
    test_label = module.res_label
    test_accuracy = module.test_acc

    sim_common = 'sim_' + str(simnum) + '_dataset_' + str(datasetnum)
    save_output_path = datadic + '/NNAttn/sim_' + str(simnum)
    if not os.path.exists(save_output_path):
        os.mkdir(save_output_path)
    save_output_path = save_output_path + '/' + sim_common
    if not os.path.exists(save_output_path):
        os.mkdir(save_output_path)
    save_output_path = save_output_path + '/' + scenario_name
    if not os.path.exists(save_output_path):
        os.mkdir(save_output_path)

    output_name = save_output_path + '/' + sim_common + \
                  '_train_' + str(num_rep_fit) + '_pred_test_' + str(num_rep_pred) + '.mat'
    sio.savemat(output_name,
                {
                    'test_accuracy': test_accuracy
                })

    return {'real_label': real_label,
            'test_label': test_label,
            'test_accuracy': test_accuracy,
            'module': module,
            'dir': output_name
            }


def nn_attention_tune_parameter(simnum=1,
                                datasetnum=1,
                                scenario_name='TrueGen',
                                oclist=[2, 5, 10],
                                kslist=[3, 6, 9, 12, 20],
                                mslist=[3, 5, 7, 10],
                                ahlist=[1, 2, 3],
                                ldlist=[10, 20, 50],
                                datadic=DATA_DIC,
                                savedic=SAVE_DIC):
    bestpar_oc = 0
    bestpar_ks = 0
    bestpar_ms = 0
    bestpar_ld = 0
    bestpar_ah = 0
    minloss = 100
    acc = 0
    for oc in oclist:
        for ks in kslist:
            for ms in mslist:
                for ah in ahlist:
                    for ld in ldlist:

                        a = EEGBCIAttnModule(
                            datadic=datadic,
                            savedic=savedic,
                            simnum=simnum,
                            datasetnum=datasetnum,
                            scenario_name=scenario_name,
                            out_channels=oc,
                            kernel_size=ks,
                            maxpool_size=ms,
                            attention_head=ah,
                            linear_dim=ld)

                        if a.min_val_loss < minloss:
                            minloss = a.min_val_loss
                            bestpar_oc = oc
                            bestpar_ks = ks
                            bestpar_ms = ms
                            bestpar_ah = ah
                            bestpar_ld = ld
                            acc = a.best_val_acc
                        print(bestpar_oc, bestpar_ks, bestpar_ms, bestpar_ah, bestpar_ld, acc, minloss)

    return {'best_oc': bestpar_oc,
            'best_ks': bestpar_ks,
            'best_ms': bestpar_ms,
            'best_ah': bestpar_ah,
            'best_ld': bestpar_ld,
            'best_validation_accuracy': acc,
            'minimum_loss': float(minloss.detach().numpy())}
