import torch
from torch import nn
import torchaudio
import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from model import CNN_Model
import get_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn = CNN_Model().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
training_losses = []
validation_losses = []
num_epochs = 30 #Tune this later.

def train(data_mode, train_data, validation_data):
    for epoch in range(num_epochs):
        print("Epoch {} training...".format(epoch))
        losses = []

        # Train
        cnn.train()
        for (wav, genre_index) in train_data:
            # print("Input waveform shape, base", wav.shape)
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # Forward
            out = cnn(wav)
            loss = loss_function(out, genre_index)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        training_losses.append(np.mean(losses).item())
        print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, num_epochs, np.mean(losses)))

        # Validation
        cnn.eval()
        y_true = []
        y_pred = []
        losses = []
        for (wav, genre_index) in validation_data:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # reshape and aggregate chunk-level predictions
            # print("Valid wav saize and wav", wav.size(), wav)
            b, t = wav.size(); c=1
            logits = cnn(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            loss = loss_function(logits, genre_index)
            losses.append(loss.item())
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, num_epochs, valid_loss, accuracy))

        # Save model
        validation_losses.append(valid_loss.item())
        if np.argmin(validation_losses) == epoch:
            print('Best model save: epoch %d' % epoch)
            torch.save(cnn.state_dict(), '{}_model.pt'.format(data_mode))

    return losses, validation_losses

def test(data_mode, test_ds):
    model = torch.load('{}_model.pt'.format(data_mode))
    cnn.load_state_dict(model); print('{} model successfully loaded.'.format(data_mode))

    cnn.eval()
    ground_truth = []
    preds = []

    with torch.no_grad():
        for wv, label in test_ds:
            wv = wv.to(device)
            label = label.to(device)

            b, t = wv.size(); c=1
            logits = cnn(wv.view(-1,t))
            logits = logits.view(b,c,-1).mean(dim=1)
            _, prd = torch.max(logits.data,dim=1)

            ground_truth.extend(label.tolist())
            preds.extend(prd.tolist())
    
    #Display some results; save the loss graphs and confusion matrix to the directory.
            
    accuracy = accuracy_score(ground_truth, preds)
    confusion = confusion_matrix(ground_truth, preds)
    fig, ax = plt.subplots()
    cmatrix = sns.heatmap(confusion, annot=True, ax=ax)
    plt.savefig('{}_confusion_matrix.png'.format(data_mode))
    print("Confusion matrix saved for {} model.".format(data_mode))
    print("{} model testing accuracy: {}".format(data_mode, accuracy))

    return None

def main(mode='train', data_mode='GTZAN'):
    train_ds = get_data.main(mode=data_mode, split=mode)
    valid_ds = get_data.main(mode=data_mode, split=mode)
    test_ds = get_data.main(mode=data_mode, split=mode)
    if mode=='train':
        training_loss, valid_loss = train(data_mode, train_ds, valid_ds)
    else:
        print('RUNNING IN TESTING MODE')
        test(data_mode, test_ds)

if __name__ == '__main__':
    main('train', 'spotify')
