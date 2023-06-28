#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score 
# sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import os
import random
import numpy as np
import pandas as pd


# %%


import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch.nn.init as init


# %%


class Generator(nn.Module):
    
    def __init__(self, noise_size=3):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(
            input_size= 1, 
            hidden_size=50, 
            num_layers=1,
            batch_first=True
            )
        
        self.lstmx = nn.LSTM(
            input_size= 1, 
            hidden_size=50, 
            num_layers=1,
            batch_first=True
            )
        
        self.lstmy = nn.LSTM(
            input_size= 1, 
            hidden_size=50, 
            num_layers=1,
            batch_first=True
            )
        
        self.lstmz = nn.LSTM(
            input_size= 1, 
            hidden_size=50, 
            num_layers=1,
            batch_first=True
            )
        
        self.lstm2 = nn.LSTM(
            input_size= 3600, 
            hidden_size=300, 
            num_layers=1,
            batch_first=True
            )
        
        self.lstmxyz = nn.LSTM(
            input_size= 150, 
            hidden_size=300, 
            num_layers=1,
            batch_first=True
            )
        self.output = nn.Linear(300, 300)

    def forward(self, input, sigmoid=True):
        #input = input.flatten(start_dim=1)
        
        inputx = input[:,:,0].reshape(input.shape[0], input.shape[1], 1)
        inputy = input[:,:,1].reshape(input.shape[0], input.shape[1], 1)
        inputz = input[:,:,2].reshape(input.shape[0], input.shape[1], 1)
        
        #print(input.shape)
        #print(inputx.shape)
        #print(inputy.shape)
        #print(inputz.shape)
        
        d_outputsx, _ = self.lstm(inputx)
        d_outputsy, _ = self.lstm(inputy)
        d_outputsz, _ = self.lstm(inputz)
        
        #d_outputs, _ = self.lstm(input)
        #d_outputs, _ = self.lstm2(d_outputs)
        #x = np.concatenate(d_outputs, axis =0)
        #print(d_outputs.shape)
        #d_outputs = d_outputs[-300: ]
        d_outputs = torch.cat([d_outputsx, d_outputsy, d_outputsz], axis=2)
        #print(d_outputs.shape)
        #d_outputs, _ = self.lstmxyz(d_outputs)
        d_outputs=torch.flatten(d_outputs, start_dim=1, end_dim = -1)
        d_outputs= d_outputs[:, -300:]
        #d_outputs = self.output(d_outputs.sum(1))
        #print('final:', d_outputs.shape)
        return d_outputs.reshape((-1, 100, 3))


# %%

# %%
"""
g = Generator()
g.cuda()
v = torch.rand(32, 100, 3)
print(v.shape)

gg = g(v.cuda()) """


# %%

# %%


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# %%


class Discriminator(nn.Module):
   
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lstm=nn.LSTM(
                          input_size = 3, 
                          hidden_size = 300, 
                          num_layers=1,
                          batch_first=True )
        self.rnn = nn.GRU(
            input_size= 300, 
            hidden_size=300, 
            num_layers=1,
            batch_first=True
            )
        
        self.rnn2 = nn.GRU(
            input_size= 300, 
            hidden_size=100, 
            num_layers=1,
            batch_first=True
            )
        self.fc_hidden = nn.Linear(100,50)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.lstm(input)
        d_outputs, _ = self.rnn(d_outputs)
        d_outputs, _ = self.rnn2(d_outputs)
        d_outputs = d_outputs 
        Y_hat = self.fc_hidden(d_outputs)
        final = self.fc(Y_hat)
        Y_hat = self.sigmoid(final)
        return Y_hat


# %%
