import torch
from torch import nn
import torchaudio
import numpy as np
import pandas as pd
#simport seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

## DEFINE THESE HERE -- not used until the main function call. 
gtzan_path = '/Users/rpant/Desktop/spotify_project_2/genres' #Genres are directories, each with the corresponding audio samples.
gtzan_labels = [f for f in os.listdir(gtzan_path) if not f.startswith('.')]
#print('GTZAN labels: ', gtzan_labels)
spotify_path = '/Users/rpant/Desktop/spotify_project_2/spotify_song_data' #Specifies the path for the Spotify data. 
spotify_labels = np.loadtxt('spotify_genre_labels.csv', delimiter=',').astype(int)

# print(spotify_labels)
# print(len(spotify_labels), type(spotify_labels), type(spotify_labels[0]))


class GTZANDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, split, num_samples=22050*29, transform=None):
        self.data_path = data_path
        self.split = split
        self.num_samples = num_samples
        # self.num_chunks = num_chunks
        self.genres = [p for p in os.listdir(data_path)if not p.startswith('.')]
        self.label_dict = {g:i for i,g in enumerate(self.genres)}
        self.transform = transform
        self.get_song_list()

    def get_song_list(self):
        song_list = []
        # for root, dirs, files in os.walk(gtzan_path):
        #     for n in files:
        #         song_list.append(os.path.join(root, n))
        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            for song in os.listdir(genre_path):
                # song_list.append((os.path.join(genre_path, song), genre))
                song_list.append(os.path.join(genre_path, song))
        # print('Length of dataset: ', len(song_list))
        self.song_list = song_list

    def normalize_audio_length(self, wv):
        # print("Normalizing waveform length", len(wv))
        if self.split == 'train':
            ridx = np.random.randint(0, len(wv)-self.num_samples-1)
            wv = wv[ridx:ridx+self.num_samples]
        else:
            h = len(wv) - self.num_samples
            wv = wv[h:h+self.num_samples]

        return wv


    def __len__(self):
        return len(self.song_list)

    def __getitem__(self, idx):
        item = self.song_list[idx]
        genre_name = item.split('/')[-2] #Hard codes the access to the genre name in the file path. 
        wv, rate = torchaudio.load(item)
        # print("getitem wv: ", wv[0])
        wv = self.normalize_audio_length(wv[0])
        if self.transform:
            wv = self.transform(wv)
        
        # print("Genre of requested item: {}".format(label))
        return wv, self.label_dict[genre_name]
        

class SpotifyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split, num_samples=22050*29, transform=None):
        self.data_path = data_path
        self.split = split
        self.num_samples = num_samples
        self.genres = spotify_labels
        # self.label_dict = {g:i for i,g in enumerate(self.genres)}
        self.transform = transform
        self.get_song_list()

    def get_song_list(self):
        song_list = []
        for song in os.listdir(self.data_path):
            song_list.append(os.path.join(self.data_path, song))
        # print('Length of dataset: ', len(song_list))
        self.song_list = song_list

    def normalize_audio_length(self, wv):
        if self.split == 'train':
            ridx = np.random.randint(0, len(wv)-self.num_samples-1)
            wv = wv[ridx:ridx+self.num_samples]
        else:
            h = len(wv) - self.num_samples
            wv = wv[h:h+self.num_samples]

        return wv

    def __len__(self):
        return len(self.song_list)
    
    def __getitem__(self, idx):
        item = self.song_list[idx]
        genre_name = self.genres[idx] #Accesses the genre from the hand-saved array...worth changing eventually. 
        wv, rate = torchaudio.load(item)
        # print("getitem wv: ", wv[0])
        wv = self.normalize_audio_length(wv[0])
        if self.transform:
            wv = self.transform(wv)
        
        # print("Genre of requested item: {}".format(label))
        return wv, genre_name

def load_data(mode='GTZAN', split='train'):
    if mode=='GTZAN':
        batch_size = 16
        ds = GTZANDataset(gtzan_path, split=split)
        data_loader = torch.utils.data.DataLoader(ds, shuffle=(ds.split=='train'), batch_size=batch_size)
        return data_loader
    elif mode=='spotify':
        print("LOADING SPOTIFY DATA")
        batch_size = 16
        ds = SpotifyDataset(spotify_path, split=split)
        data_loader = torch.utils.data.DataLoader(ds, shuffle=(ds.split=='train'), batch_size=batch_size)
        return data_loader
    else:
        print('Invalid data mode passed: ', mode)
        print("Supported options are: 'GTZAN', 'spotify'")


def main(mode='GTZAN', split='train'):
    dataset = load_data(mode,split)
    #iter_train_data = iter(dataset)
    return dataset

def tester():
    train_dataloader = load_data(mode='spotify')
    iter_train_loader = iter(train_dataloader)
    print("testing next functionality: ", next(iter_train_loader))
    print("NEXT shape: ", next(iter_train_loader)[0].shape)
    train_wav, train_genre = next(iter_train_loader)
    print("first wavelength shape: ", train_wav[0].shape)

    valid_loader = load_data(mode='spotify', split='valid')
    iter_valid_loader = iter(valid_loader)
    valid_wav, valid_genre = next(iter_valid_loader)

    print("training data shape: ", train_wav.shape)
    print("training data genres: ", train_genre)

    print("valid data shape: ", valid_wav.shape)
    print("valid data genres: ", valid_genre)

if __name__ == '__main__':
    tester()