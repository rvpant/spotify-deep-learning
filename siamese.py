import torch
from torch import nn
import torchaudio
import numpy as np
# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from model import CNNLayer

class SiameseLayer(nn.Module):

    def __init__(self, input_channels, output_channels, shape, stride, pooling, mp_stride, dropout=True):
        super(SiameseLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=shape, stride=stride, padding=0)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU(inplace=True) #why Leaky over regular ReLU? Be sure to discuss.
        #NOTE: should we tune the parameters of the relu?
        self.max_pool = nn.MaxPool2d(kernel_size=pooling, stride=mp_stride, padding=0)
        if dropout:
            self.dropout = nn.Dropout(0.1)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out

class Siamese(nn.Module):

    def __init__(self, sample_rate=22050, f_min=0.0, f_max=11025.0, num_mels=128):
        super(Siamese, self).__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024,
                                                            f_min=f_min, 
                                                            f_max=f_max, 
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_batch_norm = nn.BatchNorm2d(1) #TODO: tune this param?
        self.conv = nn.Sequential(SiameseLayer(1, 96, 11, 4, 3, 2),
                                  SiameseLayer(96, 256, 5, 1, 2, 2),
                                  nn.Conv2d(256, 384, kernel_size=5, stride=25),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(384))
  
        self.dense = nn.Sequential(nn.Linear(1152, 512),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(512, 128),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(128, 8),
                                   nn.LeakyReLU(inplace=True))
        
    def one_side_forward(self, wav):

        out = self.amplitude_to_db(self.melspec(wav)) #Loads in correctly formatted audio melspec.
        out = out.unsqueeze(1)

        out = self.input_batch_norm(out)
        out = self.conv(out)
        out = out.reshape(len(out), -1)

        out = self.dense(out)
        # print("fwd done")
        return out
    
    def forward(self, x1, x2): #Canonical forward method that uses the submethod defined above.
        out1 = self.one_side_forward(x1)
        out2 = self.one_side_forward(x2)
        return out1, out2

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        #First calculate the pairwise distance using Euclidean distance. Add a cosine similarity function in here eventually?
        pnorm = nn.PairwiseDistance(2, keepdim=True)
        dist = pnorm(x1,x2)
        loss = torch.mean((label)*dist**2 + (1-label)*torch.where(self.margin - dist >= 0, self.margin-dist, 0)**2)

        return loss