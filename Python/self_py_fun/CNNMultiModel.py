# -*- coding: utf-8 -*-
import torch
torch.manual_seed(0)
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class neuralnetwork_EEGBCI(nn.Module):
    def __init__(self, 
                 in_channel = 3, 
                 in_dim = 30, 
                 out_channels0 = 3,
                 out_channels = 30, 
                 kernel_size = 6, 
                 maxpool_size = 3, 
                 linear_dim = 10
                 ):
        """Initilize class parameters.
        Args:
            in_dim: dimension of input EEG signal
            out_channels: number of channels of CNN
            kernel_size: kernel size of CNN
            maxpool_size: kernel size of max pooling
            linear_dim: dimension of output of linear layer
        Return:
            None.
        """
        super(neuralnetwork_EEGBCI, self).__init__()
        self.CNN0 = nn.Conv2d(1, out_channels0, kernel_size = (in_channel, 1))
        self.CNN =nn.Conv1d(out_channels0, out_channels, kernel_size = kernel_size)
        self.MaxPooling = nn.MaxPool1d(maxpool_size)
        self.Linear1 = nn.Linear(out_channels * int((in_dim - kernel_size + 1) / maxpool_size), linear_dim)
        self.Linear2 = nn.Linear(linear_dim, 2)
        self.ReLU = nn.ReLU()

    def forward(self, EEGBCI_data):
        """Forward process.
        Args:
            EEGBCI_data: EEGBCI data, a dictionary that contains eeg_type, eeg_signals, etc
        Return:
            The probability of each signal being 0 or 1 (before normalized), A FloatTensor of shape [eeg_number, 2]
        """
        data = torch.FloatTensor(EEGBCI_data)
        data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
        ans_CNN0 = self.CNN0(data)
        ans_CNN0 = self.ReLU(ans_CNN0)
        ans_CNN0 = ans_CNN0.reshape(ans_CNN0.shape[0], ans_CNN0.shape[1], ans_CNN0.shape[3])
        ans_CNN = self.CNN(ans_CNN0)
        ans_maxpool = self.MaxPooling(ans_CNN)
        ans_maxpool = self.ReLU(ans_maxpool)
        ans_maxpool = ans_maxpool.reshape(ans_maxpool.shape[0], -1)
        ans_linear1 = self.Linear1(ans_maxpool)
        ans_linear1 = self.ReLU(ans_linear1)
        ans_linear2 = self.Linear2(ans_linear1)
        return ans_linear2