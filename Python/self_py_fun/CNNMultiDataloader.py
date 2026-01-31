import scipy.io as sio
import glob
import numpy as np


class EEGBCISimData:
    def __init__(self, dic, path, length):
        self.train = sio.loadmat(dic + path + '/sim_dat_ML_down_TrueGen_train')
        self.test = sio.loadmat(dic + path + '/sim_dat_ML_down_TrueGen_test')
        self.train_num = self.train['eeg_signals'].shape[0]
        self.test_num = self.test['eeg_signals'].shape[0]
        self.ydim = self.train['eeg_signals'].shape[1]
        self.dim = length
        self.channel = int(self.ydim / length)
        lists = []
        for i in range(self.channel):
            lists.append(list(range(i * length, (i+1)*length)))
            
        self.train['eeg_signals'] = self.train['eeg_signals'][:,lists]
        self.test['eeg_signals'] = self.test['eeg_signals'][:,lists]


class EEGBCIData:
    def __init__(self, dic, path, dec_factor, bp_upp, length):

        match_train = glob.glob(
            dic + path + '*eeg_dat_ML_down_' + str(dec_factor) +
            '_from_raw_bp_0.5_' + str(bp_upp) + '_odd.mat*'
        )
        match_test = glob.glob(
            dic + path + '/*eeg_dat_ML_down_' + str(dec_factor) +
            '_from_raw_bp_0.5_' + str(bp_upp) + '_even.mat*'
        )
        self.train = sio.loadmat(match_train[0])
        self.test = sio.loadmat(match_test[0])
        # print(self.train.keys())

        self.train_num = self.train['eeg_signals'].shape[0]
        self.test_num = self.test['eeg_signals'].shape[0]
        self.ydim = self.train['eeg_signals'].shape[1]
        self.dim = length
        # self.channel = int(self.ydim / length)
        # print(self.ydim)
        # print(self.dim)
        # print(self.channel)
        # lists = []
        # for i in range(self.channel):
        #     lists.append(list(range(i * length, (i + 1) * length)))
        #
        # self.train['eeg_signals'] = self.train['eeg_signals'][:, lists]
        # self.test['eeg_signals'] = self.test['eeg_signals'][:, lists]