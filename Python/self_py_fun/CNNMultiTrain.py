from self_py_fun.CNNMultiModel import neuralnetwork_EEGBCI
from self_py_fun.CNNMultiDataloader import EEGBCISimData, EEGBCIData
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class EEGBCIModule:
    def __init__(self,
                 datadic,
                 savedic,
                 design_num,
                 dataset_num,
                 length,
                 out_channels0,
                 out_channels,
                 kernel_size=6,
                 maxpool_size=3,
                 linear_dim=10,
                 val_proportion=0.2,
                 epochs=200,
                 batch_size=50,
                 lr=0.0005,
                 bp_upp=6,
                 channel_id=None,
                 sim_bool=True):

        '''Initialize class parameters.
        Args:
            datadic: the path of folder SIM_files
            savedic: the path to save the model
            design_num: the number of sim to get the data
            dataset_num: the number of dataset to get the data
            out_channels: number of channels of CNN
            kernel_size: kernel size of CNN
            maxpool_size: kernel size of max pooling
            linear_dim: dimension of output of linear layer
            val_proportion: The proportion of the validation set
            epochs: Maximum number of epoch trained
            batch_size: batch size
            lr: learning rate
            sim_bool: bool_variable
        Return:
            None.
        '''
        self.datadic = datadic
        self.savedic = savedic
        self.dataset_num = dataset_num
        self.design_num = design_num
        self.length = length
        self.out_channels0 = out_channels0
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.linear_dim = linear_dim
        self.val_proportion = val_proportion 
        self.epochs = epochs
        self.batch_size = batch_size 
        self.lr = lr
        self.bp_upp = bp_upp
        self.channel_id = channel_id
        self.sim_bool = sim_bool
        if self.sim_bool:
            self.data = EEGBCISimData(
                self.datadic,
                '/sim_' + str(self.design_num) + '/sim_' + str(self.design_num) + '_dataset_' + str(self.dataset_num),
                self.length
            )
            self.dim = self.data.train['eeg_signals'].shape[2]
            self.channels = self.data.train['eeg_signals'].shape[1]
            print('Training signals have shape {}'.format(self.data.train['eeg_signals'].shape))
            print('Testing signals have shape {}'.format(self.data.test['eeg_signals'].shape))
        else:
            self.data = EEGBCIData(
                self.datadic, '/K' + str(self.design_num) + '/', self.dataset_num, self.bp_upp, self.length
            )
            self.dim = self.data.train['eeg_signals'].shape[2]

            if self.channel_id is None or len(self.channel_id) == 16:
                print('No subseting is performed. We are using all channels.')
                self.channels = self.data.train['eeg_signals'].shape[1]
                self.channel_name = 'all_channels'
            else:
                print('Subseting is performed. We are using channels {}'.format(self.channel_id + 1))
                self.channels = len(self.channel_id)
                self.data.train['eeg_signals'] = self.data.train['eeg_signals'][:, self.channel_id, :]
                self.data.test['eeg_signals'] = self.data.test['eeg_signals'][:, self.channel_id, :]
                # Rename channels
                self.channel_id_str = [str(e+1) for e in self.channel_id]
                self.channel_name = 'channel_' + '_'.join(self.channel_id_str)

            print('Training signals have shape {}'.format(self.data.train['eeg_signals'].shape))
            print('Testing signals have shape {}'.format(self.data.test['eeg_signals'].shape))

        self.model = neuralnetwork_EEGBCI(
            self.channels,self.dim, out_channels0,
            out_channels, kernel_size,
            maxpool_size, linear_dim
        )
        self.train(val_proportion, epochs, batch_size, lr)
        self.test()
        
    def train(self, val_proportion=0.2, epochs=200, batch_size=50, lr=0.0005):
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
        if self.sim_bool:
            if not os.path.exists(self.savedic + "/NeuralNetworksave"):
                os.mkdir(self.savedic + "/NeuralNetworksave")
            if not os.path.exists(self.savedic + "/NeuralNetworksave/models"):
                os.mkdir(self.savedic + "/NeuralNetworksave/models")
            if not os.path.exists(self.savedic + "/NeuralNetworksave/models/sim_" + str(self.design_num)):
                os.mkdir(self.savedic + "/NeuralNetworksave/models/sim_" + str(self.design_num))

            savefilepath = self.savedic + "/NeuralNetworksave/models/sim_" + str(self.design_num) + '/sim_' + \
                           str(self.design_num) + '_dataset_' + str(self.dataset_num)
            if not os.path.exists(savefilepath):
                os.mkdir(savefilepath)
        else:
            if not os.path.exists(self.savedic + "/CNN"):
                os.mkdir(self.savedic + "/CNN")
            if not os.path.exists(self.savedic + "/CNN/models"):
                os.mkdir(self.savedic + "/CNN/models")
            if not os.path.exists(self.savedic + "/CNN/models/" + str(self.channel_name)):
                os.mkdir(self.savedic + "/CNN/models/" + str(self.channel_name))
            savefilepath = self.savedic + 'CNN/models/' + str(self.channel_name)

        dat = self.data.train['eeg_signals']
        label = self.data.train['eeg_type'][0]
        # np.random.seed(1)
        numbers = list(range(dat.shape[0]))
        np.random.shuffle(numbers)
        train_rows = numbers[0:int((1 - val_proportion) * len(numbers))]
        val_rows = numbers[int((1-val_proportion) * len(numbers)):]
        train_data = dat[train_rows, :,:]
        val_data = dat[val_rows,: ,:]
        train_label = label[train_rows]
        val_label = label[val_rows]
        if self.sim_bool:
            print('train: sim ' + str(self.design_num) + ', dataset ' + str(self.dataset_num)
                  + ', scenario TrueGen')
            # TrueGen by default, may be extended to other scenarios if necessary
        else:
            print('train: K' + str(self.design_num) + ', decimation factor ' + str(self.dataset_num))

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = 0.1)
        self.loss_function = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
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
                bt = order[batches * batch_size:(batches+1)*batch_size]
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
            acc_train = 1 - sum(((self.softmax(train_res).detach().numpy()[:,1] > 0.5) - (train_label + 1) / 2)**2) / len(train_label)
            acc_val = 1 - sum(((self.softmax(val_res).detach().numpy()[:,1] > 0.5) - (val_label + 1) / 2)**2) / len(val_label)
            self.train_loss.append(loss_train.detach().numpy())
            self.val_loss.append(loss_val.detach().numpy())
            # process_bar.set_description("Train Loss: %0.6f, Validation Loss: %0.6f, Train Accuracy: %0.6f, Validation Accuracy: %0.6f" %
            #                             (loss_train,
            #                              loss_val, acc_train, acc_val))
            if self.min_val_loss > loss_val:
                self.min_val_loss = loss_val
                self.best_val_acc = acc_val
                self.best_epoch = epoch
                torch.save(self.model, savefilepath + "/" + "oc0%doc%dks%dms%dld%dlr%f.model" % (self.out_channels0,self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
                self.best_model = torch.load(savefilepath + "/" + "oc0%doc%dks%dms%dld%dlr%f.model" % (self.out_channels0,self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
        print('Minimum validation loss: %0.6f, Best validation accuracy: %0.6f, best epoch: %d' % (self.min_val_loss, self.best_val_acc, self.best_epoch))
        f_train, = plt.plot(self.train_loss, 'r')
        f_val, = plt.plot(self.val_loss, 'b')
        self.fig = plt.legend([f_train, f_val], ['train', 'validation'])
        plt.savefig(savefilepath + "/" + "oc0%doc%dks%dms%dld%dlr%f.jpg" % (self.out_channels0,self.out_channels, self.kernel_size, self.maxpool_size, self.linear_dim, lr))
        plt.close()
        return self.min_val_loss, self.best_val_acc
    
    def test(self):
        test_res = self.best_model(self.data.test['eeg_signals'])
        test_label = self.data.test['eeg_type'][0]
        self.loss_test = float(self.loss_function(test_res, torch.LongTensor((test_label + 1) / 2)).detach().numpy())
        self.acc_test = 1 - sum(((self.softmax(test_res).detach().numpy()[:,1] > 0.5) - (test_label + 1) / 2)**2) / len(test_label)
        self.test_pb = self.softmax(test_res).detach().numpy()[:,1]
    
        train_res = self.best_model(self.data.train['eeg_signals'])
        train_label = self.data.train['eeg_type'][0]
        self.loss_train = float(self.loss_function(train_res, torch.LongTensor((train_label + 1) / 2)).detach().numpy())
        self.acc_train = 1 - sum(((self.softmax(train_res).detach().numpy()[:,1] > 0.5) - (train_label + 1) / 2)**2) / len(train_label)
        self.train_pb = self.softmax(train_res).detach().numpy()[:, 1]