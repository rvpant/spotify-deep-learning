import torch
from torch import nn
import torchaudio
import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from siamese import Siamese, ContrastiveLoss
import siamese_get_data
from model import CNN_Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Siamese().to(device)
loss_function = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
training_losses = []
num_epochs = 30 #Tune this later.

def train(train_data, data_mode):
    for epoch in range(num_epochs):
        print("Epoch {} training...".format(epoch))
        losses = []

        #We begin by training the model.
        model.train()
        for (wav1, wav2, label, g1, g2) in train_data:

            #load data onto the device
            wav1 = wav1.to(device)
            wav2 = wav2.to(device)
            label = label.to(device)

            #fwdpass
            out1, out2 = model(wav1, wav2)
            loss = loss_function(out1, out2, label)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item()) #This stores batch loss.
        training_losses.append(np.mean(losses).item()) #This adds the mean of the batch loss to the overall training loss.
        print('Epoch: [%d/%d], Train loss: %.3f' % (epoch+1, num_epochs, np.mean(losses)))
        if np.argmin(training_losses) == epoch:
            print('Siamese saved at  epoch %d' % epoch)
            torch.save(model.state_dict(), 'siamese_{}_model.pt'.format(data_mode))

    return training_losses

def test(test_ds, data_mode):
    siamese = torch.load('siamese_{}_model.pt'.format(data_mode))
    model.load_state_dict(siamese); print('{} Siamese Model successfully loaded.'.format(data_mode))

    model.eval()
    same_genre_similarity = {}
    diff_genre_similarity = {}
    t1 = []
    t2 = []
    with torch.no_grad():
        data_iterator = iter(test_ds)
        for i in range(8):
            ref1, ref2, label, g1, g2 = next(data_iterator)
            for j in range(8):
                x1, x2, l, gen1, gen2 = next(data_iterator)
                joined = torch.cat((x1,x2),0)
                o1, o2 = model(ref1, x2)
                twonorm = nn.PairwiseDistance(keepdim=False)
                dissimilarity = twonorm(o1, o2)

                if g1 == gen2:
                    t1.append(dissimilarity.numpy())
                else:
                    t2.append(dissimilarity.numpy())
    print("Average in-genre dissimilarity score: ", np.mean(t1))
    print("Average diff-genre dissimilarity score: ", np.mean(t2))

    eval_similarity_from_model(test_ds, data_mode)

    return None

def eval_similarity_from_model(test_ds, data_mode):
    '''Function that loads a classification model and uses that as the encoder for two songs.
    We investigate whether such a model performs more successfully than the custom-trained few-show Siamese learner.'''

    #Written assuming that our test dataset remains the siamese one. Can rewrite to include the original test set.

    cnn_model = CNN_Model().to(device)
    # cnn = torch.load('{}_model.pt'.format(data_mode))
    cnn = torch.load('best_model.ckpt'.format(data_mode))
    cnn_model.load_state_dict(cnn)
    cnn_model.eval()

    t1 = []
    t2 = []
    with torch.no_grad():
        data_iterator = iter(test_ds)
        for i in range(8):
            ref1, ref2, label, g1, g2 = next(data_iterator)
            for j in range(8):
                x1, x2, l, gen1, gen2 = next(data_iterator)
                # joined = torch.cat((x1,x2),0)
                o1, o2 = cnn_model(ref1), cnn_model(x2)
                twonorm = nn.PairwiseDistance(keepdim=False)
                dissimilarity = twonorm(o1, o2)

                if g1 == gen2:
                    t1.append(dissimilarity.numpy())
                else:
                    t2.append(dissimilarity.numpy())
    
    print("Average in-genre dissimilarity score (classification model, {}): ".format(data_mode), np.mean(t1))
    print("Average diff-genre dissimilarity score (classification model, {}): ".format(data_mode), np.mean(t2))

    return None


def main(mode='train', data_mode='gtzan'):
    train_ds = siamese_get_data.main()
    test_ds = siamese_get_data.main(split='test')
    if mode=='train':
        print("Running training...")
        training_loss = train(train_ds, data_mode)
        fig, ax = plt.subplots()
        ax.plot(range(len(training_loss)), training_loss)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Contrastive Training Loss for Siamese Model')
        plt.savefig('siamese_training_losses.png')
        print("Training loss image saved.")
    else:
        print('RUNNING IN TESTING MODE')
        test(test_ds, data_mode)

if __name__ == '__main__':
    main(mode='test')