import torch
from torch import nn
import torchaudio
import numpy as np
# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

classes = 10 #for GTZAN dataset, will have to change if we use this model on our own Spotify data set.

#NOTES (2/3/24): might look to hyper sweep over sample rate, max/min pooling, and dropout rates?

class CNNLayer(nn.Module):

    def __init__(self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU() #why Leaky over regular ReLU? Be sure to discuss.
        #NOTE: should we tune the parameters of the relu?
        self.max_pool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.dropout(out)
        return out

class CNN_Model(nn.Module):

    def __init__(self, sample_rate=22050,
                        n_fft=1024, 
                        f_min=0.0, 
                        f_max=11025.0, 
                        num_mels=128,
                        num_classes=classes):
        
        super(CNN_Model, self).__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024,
                                                            f_min=f_min, 
                                                            f_max=f_max, 
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_batch_norm = nn.BatchNorm2d(1) #TODO: tune this param?

        #Build the convolutional layers. Start with tutorial shapes and alter if needed.
        self.layer1 = CNNLayer(1, 16, pooling=(2, 3))
        self.layer2 = CNNLayer(16, 16, pooling=(3, 4))
        self.layer3 = CNNLayer(16, 32, pooling=(2, 5))
        self.layer4 = CNNLayer(32, 32, pooling=(3, 3))
        self.layer5 = CNNLayer(32, 64, pooling=(3, 4))

        #Add dense layers -- as earlier, can change the shape and number of these later if needed.
        self.dense1 = nn.Linear(64,64)
        self.dense_batch_norm = nn.BatchNorm1d(64)
        self.dense2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5) #tune this param?
        self.relu = nn.LeakyReLU()

    def forward(self, wav):
        # print("starting fwd pass")
        # print("base fwd wav shape: ", wav.shape)
        #Process audio input into the mel spectrogram.
        out = self.melspec(wav)
        out = self.amplitude_to_db(out)

        #Perform batch normalization of the inputs.
        # print("Shape of input into batch normalization: ", out.shape)
        # print("base shape before unsqueeze: ", out.shape)
        out = out.unsqueeze(1)
        # print("base shape after unsqueeze: ", out.shape)
        out = self.input_batch_norm(out)

        #Pass through convolutional layers.
        out = self.layer1(out)
        # print("after layer 1", out.shape)
        out = self.layer2(out)
        # print("after layer 2", out.shape)
        out = self.layer3(out)
        # print("after layer 3", out.shape)
        out = self.layer4(out)
        # print("after layer 4", out.shape)
        out = self.layer5(out)
        # print("after layer 5", out.shape)
        
        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        #CURRENTLY: [1, num_channels, ??, ??] for some reason...
        # print("Shape before reshaping: ", out.shape)
        
        out = out.reshape(len(out), -1)

        # print("Shape after reshaping: ", out.shape)

        #Pass through dense layers.
        out = self.dense1(out)
        out = self.dense_batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        # print("completed forward pass")

        return out

