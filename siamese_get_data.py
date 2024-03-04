import torch
from torch import nn
import torchaudio
import numpy as np
#import pandas as pd
#simport seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import random

gtzan_path = '/Users/rpant/Desktop/spotify_project_2/genres' #Genres are directories, each with the corresponding audio samples.
#gtzan_labels = list(os.walk())[0][1]
gtzan_labels = [f for f in os.listdir(gtzan_path) if not f.startswith('.')]
#print('GTZAN labels: ', gtzan_labels)
spotify_path = '/Users/rpant/Desktop/spotify_project_2/spotify_data' #Specifies the path for the Spotify data. 
spotify_labels = []

class SiameseDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, split, num_samples=22050*29, num_pairs = 1000, transform=None):
        super(SiameseDataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.genres = [p for p in os.listdir(data_path)if not p.startswith('.')] #Carryover from get_data, we will use to design labels.
        self.label_dict = {g:i for i,g in enumerate(self.genres)} #As above.
        self.transform = transform
        self.get_data_pairs()

    def get_data_pairs(self):
        pairs = []
        for i in range(self.num_pairs):
            gen1_idx = np.random.randint(0, len(self.genres))
            selector = np.random.randint(0,high=2)
            label = None
            #We introduce the selector here to approximately get 50% positive and 50% negative samples.
            if selector > 0.5:
                gen2_idx = gen1_idx
                label = 1
            else:
                gen2_idx = np.random.randint(0, len(self.genres))
                label = 0
            gen1 = self.genres[gen1_idx]
            gen2 = self.genres[gen2_idx]
            gen1_path = os.path.join(self.data_path, gen1)
            gen2_path = os.path.join(self.data_path, gen2)
            x1 = random.choice([f for f in os.listdir(gen1_path)])
            x2 = random.choice([f for f in os.listdir(gen2_path)])
            pairs.append((os.path.join(gen1_path, x1), os.path.join(gen2_path, x2), label))
        print("Data pairs loaded.")
        self.data_pairs = pairs
    
    def normalize_audio_length(self, wv):
        # print("Normalizing waveform length", len(wv))
        if self.split == 'train':
            #print("wv length: ", len(wv))
            ridx = np.random.randint(0, len(wv)-self.num_samples-1)
            wv = wv[ridx:ridx+self.num_samples]
        else:
            h = len(wv) - self.num_samples
            wv = wv[h:h+self.num_samples]

        return wv

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        x1, x2, label = self.data_pairs[idx]
        gen1_name = x1.split('/')[-2] #Hard coded, as we learned from building get_data.
        gen2_name = x2.split('/')[-2]
        wv1, rt1 = torchaudio.load(x1)
        wv2, rt2 = torchaudio.load(x2)
        wv1, wv2 = self.normalize_audio_length(wv1[0]), self.normalize_audio_length(wv2[0])
        
        #Here we should look to add random augmentations/transforms to the data.

        if self.transform:
            wv1, wv2 = self.transform(wv1), self.transform(wv2)

        # print("Requested item: ", gen1_name, gen2_name, label)
        return (wv1, wv2, label, gen1_name, gen2_name)
    
def load_data(split='train'):
    if split=='train':
        batch_size = 32
    else:
        batch_size = 1
    ds = SiameseDataset(gtzan_path, split=split)
    data_loader = torch.utils.data.DataLoader(ds, shuffle=(ds.split=='train'), batch_size=batch_size)
    return data_loader

def show_batch(img):
    npimg = img.numpy()
    plt.axis("off")
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig("example_siamese_batch.py") 

def main(split='train'):
    dataset = load_data(split)
    return dataset
    
def tester():
    print("Running testing ...")

    print("TRAIN DATA TESTING")
    train_dataloader = load_data()
    iter_train_loader = iter(train_dataloader)
    print("testing next functionality: ", next(iter_train_loader))
    print("NEXT shape: ", next(iter_train_loader)[0].shape)
    train_wav1, train_wav2, train_label = next(iter_train_loader)
    print("first wavelength shape:", (train_wav1.shape))
    print("second wavelength shape: ", train_wav2.shape)

    # mel1 = torchaudio.transforms.MelSpectrogram(train_wav1[0])
    # mel2 = torchaudio.transforms.MelSpectrogram(train_wav2[0])
    # concat = torch.cat((mel1, mel2), 0)
    # print("Concat shape: ", concat.shape)
    # show_batch(concat)
    print("Labels of above batch: ")
    print(train_label)

    print("Done!")

if __name__ == '__main__':
    main()