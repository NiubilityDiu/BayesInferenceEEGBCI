import os
import scipy.io as sio
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(0)

DATA_DIC = 'data_dic'
SAVE_DIC = 'save_dic'


class EEGBCISimData:
    def __init__(self, dic, path, scenario_name):
        self.train = sio.loadmat(dic + path + '/sim_dat_ML_down_' + str(scenario_name) + '_train')
        self.test = sio.loadmat(dic + path + '/sim_dat_ML_down_' + str(scenario_name) + '_test')
        self.train_num = self.train['eeg_signals'].shape[0]
        self.test_num = self.test['eeg_signals'].shape[0]
        self.dim = self.train['eeg_signals'].shape[1]


class EEGBCIRealData:
    def __init__(self, dic, k_num, dec_factor, file_subscript, channel_id):
        self.train = sio.loadmat('{}/{}/{}_001_BCI_TRN_eeg_dat_ML_down_{}_from_{}_odd.mat'.format(
            dic, k_num, k_num, dec_factor, file_subscript
        ))
        self.train['eeg_signals'] = self.train['eeg_signals'][:, channel_id, :]
        self.test = sio.loadmat('{}/{}/{}_001_BCI_TRN_eeg_dat_ML_down_{}_from_{}_even.mat'.format(
            dic, k_num, k_num, dec_factor, file_subscript
        ))
        self.test['eeg_signals'] = self.test['eeg_signals'][:, channel_id, :]
        self.train_num = self.train['eeg_signals'].shape[0]
        self.test_num = self.test['eeg_signals'].shape[0]
        # self.num_electrode = self.train['eeg_signals'].shape[1]
        self.dim = self.train['eeg_signals'].shape[1]


class EEGBCINNSimOrdinary(nn.Module):
    def __init__(self,
                 in_dim=30,
                 out_channels=10,
                 kernel_size=6,
                 maxpool_size=3,
                 linear_dim=10
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
        super(EEGBCINNSimOrdinary, self).__init__()
        self.CNN = nn.Conv1d(1, out_channels, kernel_size=kernel_size)
        self.MaxPooling = nn.MaxPool1d(maxpool_size)
        self.Linear1 = nn.Linear(out_channels * int((in_dim - kernel_size + 1) / maxpool_size), linear_dim)
        self.Linear2 = nn.Linear(linear_dim, 2)
        self.ReLU = nn.ReLU()

    def forward(self, EEGBCISimData):
        """Forward process.
        Return:
            The probability of each signal being 0 or 1 (before normalized), A FloatTensor of shape [eeg_number, 2]
        """
        data = torch.FloatTensor(EEGBCISimData)
        # print(data.shape)
        data = data.reshape(data.shape[0], 1, data.shape[1])
        # print(data.shape)
        ans_CNN = self.CNN(data)
        ans_maxpool = self.MaxPooling(ans_CNN)
        ans_maxpool = self.ReLU(ans_maxpool)
        ans_maxpool = ans_maxpool.reshape(ans_maxpool.shape[0], -1)
        ans_linear1 = self.Linear1(ans_maxpool)
        ans_linear1 = self.ReLU(ans_linear1)
        ans_linear2 = self.Linear2(ans_linear1)

        return ans_linear2


class EEGBCINNRealOrdinary(nn.Module):
    def __init__(self,
                 in_dim=25,
                 out_channels=10,
                 kernel_size=6,
                 maxpool_size=3,
                 linear_dim=10
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
        super(EEGBCINNRealOrdinary, self).__init__()
        self.CNN = nn.Conv1d(1, out_channels, kernel_size=kernel_size)
        self.MaxPooling = nn.MaxPool1d(maxpool_size)
        self.Linear1 = nn.Linear(out_channels * int((in_dim - kernel_size + 1) / maxpool_size), linear_dim)
        self.Linear2 = nn.Linear(linear_dim, 2)
        self.ReLU = nn.ReLU()

    def forward(self, EEGBCIRealData):
        """Forward process.
        Return:
            The probability of each signal being 0 or 1 (before normalized), A FloatTensor of shape [eeg_number, 2]
        """
        data = torch.FloatTensor(EEGBCIRealData)
        # print(data.shape)
        data = data.reshape(data.shape[0], 1, data.shape[1])
        ans_CNN = self.CNN(data)
        ans_maxpool = self.MaxPooling(ans_CNN)
        ans_maxpool = self.ReLU(ans_maxpool)
        ans_maxpool = ans_maxpool.reshape(ans_maxpool.shape[0], -1)
        ans_linear1 = self.Linear1(ans_maxpool)
        ans_linear1 = self.ReLU(ans_linear1)
        ans_linear2 = self.Linear2(ans_linear1)

        return ans_linear2


class EEGBCINNOrdModule:
    def __init__(self,
                 data_dic, save_dic, sim_dat_bool,
                 sim_num=1, dataset_num=1, scenario_name='TrueGen',
                 k_num=114, dec_factor=8, eeg_file_suffix='raw_bp_0.5_5.5', channel_id=0,
                 out_channels=10,
                 kernel_size=6,
                 maxpool_size=3,
                 linear_dim=10,
                 val_proportion=0.2,
                 epochs=200,
                 batch_size=50,
                 lr=0.0005
                 ):

        ''' Initialize class parameters.
        Args:
            data_dic: the path of folder SIM_files
            save_dic: the path to save the model
            sim_dat_bool: bool variable
            sim_num: the number of sim to get the data
            dataset_num: the number of dataset to get the data
            scenario_name: string
            k_num: integer
            dec_factor: positive real number (integer here)
            eeg_file_suffix: string
            channel_id: integer between 0 and 15
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
        self.data_dic = data_dic
        self.save_dic = save_dic
        self.sim_dat_bool = sim_dat_bool

        self.sim_num = sim_num
        self.dataset_num = dataset_num
        self.scenario_name = scenario_name

        self.k_num = k_num
        self.dec_factor = dec_factor
        self.eeg_file_suffuix = eeg_file_suffix
        self.channel_id = channel_id

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.linear_dim = linear_dim
        self.val_proportion = val_proportion 
        self.epochs = epochs
        self.batch_size = batch_size 
        self.lr = lr
        self.sim_common = 'sim_' + str(self.sim_num) + '_dataset_' + str(self.dataset_num)

        if self.sim_dat_bool:
            self.data = EEGBCISimData(
                data_dic, '/SIM_files/sim_' + str(self.sim_num) + '/' + self.sim_common,
                self.scenario_name
            )
        else:
            self.data = EEGBCIRealData(
                save_dic, 'K'+str(self.k_num), self.dec_factor, self.eeg_file_suffuix, self.channel_id,
            )
        self.dim = self.data.dim
        if self.sim_dat_bool:
            self.model = EEGBCINNSimOrdinary(self.dim, out_channels, kernel_size, maxpool_size, linear_dim)
        else:
            self.model = EEGBCINNRealOrdinary(self.dim, out_channels, kernel_size, maxpool_size, linear_dim)
        self.train(val_proportion, epochs, batch_size, lr)
        self.test()

    def train(
            self, val_proportion=0.2, epochs=200, batch_size=50, lr=0.0005,
    ):

        r'''Train the model.
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
        if self.sim_dat_bool:
            save_file_path = self.save_dic + "sim_" + str(self.sim_num) + '/' + self.sim_common + '/NNOrd/'

            if not os.path.exists(save_file_path):
                os.mkdir(save_file_path)
            dat = self.data.train['eeg_signals']
            label = self.data.train['eeg_type'][0]
        else:
            save_file_path = '{}/K{}/NNOrd/'.format(self.save_dic, self.k_num)
            if not os.path.exists(save_file_path):
                os.mkdir(save_file_path)
            dat = self.data.train['eeg_signals']
            label = self.data.train['eeg_type'][0, :]

        np.random.seed(1)
        numbers = list(range(dat.shape[0]))
        np.random.shuffle(numbers)
        train_rows = numbers[0:int(val_proportion * len(numbers))]
        val_rows = numbers[int(val_proportion * len(numbers)):]
        train_data = dat[train_rows,]
        val_data = dat[val_rows,]
        train_label = label[train_rows]
        val_label = label[val_rows]
        if self.sim_dat_bool:
            print('train: sim ' + str(self.sim_num) + ', dataset ' + str(self.dataset_num) +
                  ', scenario ' + self.scenario_name, '\nmethod: neural network without attention.')
        else:
            print('train: K' + str(self.k_num) + ', channel ' + str(self.channel_id + 1) +
                  '\nmethod: neural network without attention.')
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
                if self.sim_dat_bool:
                    torch.save(self.model, save_file_path + "/" + "oc%dks%dms%dld%dlr%f.model" % (
                        self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
                    self.best_model = torch.load(save_file_path + "/" + "oc%dks%dms%dld%dlr%f.model" % (
                        self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
                else:
                    torch.save(self.model, save_file_path + "/" + 'channel_' + str(self.channel_id + 1) +
                               '_' + "oc%dks%dms%dld%dlr%f.model" % (
                        self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
                    self.best_model = torch.load(save_file_path + "/" + 'channel_' + str(self.channel_id + 1) +
                                                 '_' + "oc%dks%dms%dld%dlr%f.model" % (
                        self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
        print('Minimum validation loss: %0.6f, Best validation accuracy: %0.6f, best epoch: %d' % (
        self.min_val_loss, self.best_val_acc, self.best_epoch))
        f_train, = plt.plot(self.train_loss, 'r')
        f_val, = plt.plot(self.val_loss, 'b')
        self.fig = plt.legend([f_train, f_val], ['train', 'validation'])
        if self.sim_dat_bool:
            plt.savefig(save_file_path + "/" + "oc%dks%dms%dld%dlr%f.png" % (
            self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
        else:
            plt.savefig(save_file_path + "/" + 'channel_' + str(self.channel_id + 1) +
                        '_' + "oc%dks%dms%dld%dlr%f.png" % (
            self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
        plt.close()
        return self.min_val_loss, self.best_val_acc

    def test(self):
        if self.sim_dat_bool:
            test_res = self.best_model(self.data.test['eeg_signals'])
            test_label = self.data.test['eeg_type'][0]
        else:
            test_res = self.best_model(self.data.test['eeg_signals'])
            test_label = self.data.test['eeg_type'][0, :]
        self.loss_test = float(self.loss_function(test_res, torch.LongTensor((test_label + 1) / 2)).detach().numpy())
        self.acc_test = 1 - sum(
            ((self.softmax(test_res).detach().numpy()[:, 1] > 0.5) - (test_label + 1) / 2) ** 2) / len(test_label)
        self.test_pb = self.softmax(test_res).detach().numpy()[:, 1]

        train_res = self.best_model(self.data.train['eeg_signals'])
        train_label = self.data.train['eeg_type'][0]
        self.loss_train = float(self.loss_function(train_res, torch.LongTensor((train_label + 1) / 2)).detach().numpy())
        self.acc_train = 1 - sum(
            ((self.softmax(train_res).detach().numpy()[:, 1] > 0.5) - (train_label + 1) / 2) ** 2) / len(train_label)
        self.train_pb = self.softmax(train_res).detach().numpy()[:, 1]


def get_nn_ordinary_results(
        sim_dat_bool,
        simnum, datasetnum, scenario_name,
        k_num, dec_factor, eeg_file_suffix, channel_id,
        out_channels=10, kernel_size=6, maxpool_size=3,
        linear_dim=10, val_proportion=0.2, epochs=200,
        batch_size=50, lr=0.0005,
        datadic=DATA_DIC, savedic=SAVE_DIC):

    module = EEGBCINNOrdModule(
        datadic, savedic, sim_dat_bool,
        simnum, datasetnum, scenario_name,
        k_num, dec_factor, eeg_file_suffix, channel_id,
        out_channels, kernel_size, maxpool_size, linear_dim,
        val_proportion, epochs, batch_size, lr
    )

    return {'train_loss': module.loss_train,
            'train_accuracy': module.acc_train,
            'test_loss': module.loss_test,
            'test_accuracy': module.acc_test,
            'train_probability': module.train_pb,
            'test_probability': module.test_pb}


def nn_ordinary_tune_parameter(
        sim_dat_bool=True,
        simnum=1, datasetnum=1, scenario_name='TrueGen',
        k_num=114, dec_factor=8, eeg_file_suffix='raw_bp_0.5_5.5', channel_id=0,
        oclist=np.array([2, 5, 10]),
        kslist=np.array([3, 6, 9, 12, 20]),
        mslist=np.array([3, 5, 7, 10]),
        ldlist=np.array([10, 20, 50]),
        datadic=DATA_DIC,
        savedic=SAVE_DIC
):
    bestpar_oc = 0
    bestpar_ks = 0
    bestpar_ms = 0
    bestpar_ld = 0
    minloss = 100
    acc = 0
    for oc in oclist:
        for ks in kslist:
            for ms in mslist:
                for ld in ldlist:

                    a = EEGBCINNOrdModule(
                        data_dic=datadic, save_dic=savedic,
                        sim_dat_bool=sim_dat_bool,
                        sim_num=simnum, dataset_num=datasetnum, scenario_name=scenario_name,
                        k_num=k_num, dec_factor=dec_factor, eeg_file_suffix=eeg_file_suffix, channel_id=channel_id,
                        out_channels=oc,
                        kernel_size=ks,
                        maxpool_size=ms,
                        linear_dim=ld
                    )
                    if a.min_val_loss < minloss:
                        minloss = a.min_val_loss
                        bestpar_oc = oc
                        bestpar_ks = ks
                        bestpar_ms = ms
                        bestpar_ld = ld
                        acc = a.best_val_acc
                    print(bestpar_oc, bestpar_ks, bestpar_ms, bestpar_ld, acc, minloss)

    return {'best_oc': bestpar_oc,
            'best_ks': bestpar_ks,
            'best_ms': bestpar_ms,
            'best_ld': bestpar_ld,
            'best_validation_accuracy': acc,
            'minimum_loss': float(minloss.detach().numpy())}
