# File to get spotify data from the API directly, for setup with other files.

from pyexpat import features
import spotipy as sp
import pandas as pd
import numpy as np
from pprint import pprint
import librosa
import skimage
import os

from urllib.request import urlretrieve
from sklearn.preprocessing import minmax_scale
from pathlib import Path

client_id = '33b5f51cd6be42b398665d1ee9c933e4' #Replace this with your client ID from Spotify Dev.
client_secret = 'd742cc12736e41758c1711b399a1dff4' #Replace this with your client secret from Spotify Dev.

def spotify_login(client_id, client_secret):
    client_credentials_manager = sp.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret = client_secret)
    return sp.Spotify(client_credentials_manager = client_credentials_manager, requests_timeout=10, retries=5)



def build_track_data():
    '''DEPRECATED: function to download and save a variety of song features in textfile format. '''
    spot = spotify_login(client_id, client_secret)
    dat = spot.search('Country Gold', type='playlist')
    playlist_id = dat['playlists']['items'][0]['id']

    song_ids = []
    song_urls = []
    song_names = []
    song_features =  []
    song_artists = []
    song_artist_genre = []
    song_artist_related = []

    playlist_songs = spot.playlist_tracks(playlist_id, limit=25)

    for i in range(len(playlist_songs['items'])):
        print(i)
        track = playlist_songs['items'][i]['track']
        # print(track['name'])
        artist = playlist_songs['items'][i]['track']['artists'][0]
        # print(artist['name'], artist['id'])
        if len(spot.search(artist['id'], type='artist', limit=1)['artists']['items']) == 0:
            artist_genre = spot.search(artist['name'], type='artist', limit=1)['artists']['items'][0]['genres'][0]
        else:
            artist_genre = spot.search(artist['id'], type='artist', limit=1)['artists']['items'][0]['genres'][0]
        # print(artist_genre)
        if len(spot.artist_related_artists(artist['id'])['artists']) != 0:
            artist_related = [art['name'] for art in spot.artist_related_artists(artist['id'])['artists']]
        else:
            artist_related = []
        song_ids.append(track['id'])
        song_urls.append(track['preview_url'])
        song_names.append(track['name'])
        song_features.append(spot.audio_features(track['id']))
        song_artists.append(artist['name'])
        song_artist_genre.append(artist_genre)
        song_artist_related.append(artist_related)

        #next, iterate through the artist's albums, and add any songs to our data set that have not already been included
        artist_albums = spot.artist_albums(artist['id'])
        for album in artist_albums['items']:
            album_id = album['id']
            album_tracks = spot.album_tracks(album_id)
            for track in album_tracks['items']:
                if track['id'] not in song_ids:
                    song_ids.append(track['id'])
                    song_urls.append(track['preview_url'])
                    song_names.append(track['name'])
                    song_features.append(spot.audio_features(track['id']))
                    song_artists.append(artist['name'])
                    song_artist_genre.append(artist_genre)
                    song_artist_related.append(artist_related)


        #finally, iterate through the related artists and add their songs/info to the data set if not already included
        for rel in artist_related[:5]:
            rel_artist = spot.search(rel, type='artist')['artists']['items'][0]
            rel_artist_id = rel_artist['id']
            if len(spot.search(rel_artist['id'], type='artist', limit=1)['artists']['items']) == 0:
                rel_artist_genre = spot.search(rel_artist['name'], type='artist', limit=1)['artists']['items'][0]['genres'][0]
            else:
                rel_artist_genre = spot.search(rel_artist['id'], type='artist', limit=1)['artists']['items'][0]['genres'][0]
            # print(rel_artist_genre)
            if len(spot.artist_related_artists(rel_artist['id'])['artists']) != 0:
                rel_artist_related = [art['name'] for art in spot.artist_related_artists(rel_artist['id'])['artists']]
            else:
                rel_artist_related = []
            if not rel in song_artists:
                rel_artist_albums = spot.artist_albums(rel_artist_id)
                for album in rel_artist_albums['items']:
                    rel_album_id = album['id']
                    rel_album_tracks = spot.album_tracks(rel_album_id)
                    for track in rel_album_tracks['items']:
                        if track['id'] not in song_ids:
                            song_ids.append(track['id'])
                            song_urls.append(track['preview_url'])
                            song_names.append(track['name'])
                            song_features.append(spot.audio_features(track['id']))
                            song_artists.append(rel_artist['name'])
                            song_artist_genre.append(rel_artist_genre)
                            song_artist_related.append(rel_artist_related)

    song_df = pd.DataFrame({'Song ID': song_ids, 'Song URL': song_urls, 'Song Name': song_names, 'Song Features': song_features, 'Artist': song_artists, 
    'Genre': song_artist_genre, 'Related Artists': song_artist_related})

    song_df = song_df[(song_df['Genre'] != 'alt z') & (song_df['Genre'] != 'dance pop')]
    song_df = song_df[~song_df['Song Name'].str.contains('Commentary')]
    song_df.reset_index(inplace=True)
    arr = []
    for i in range(len(song_df)):
        if 'contemporary' in song_df.loc[i]['Genre']:
            arr.append(1)
        elif 'alberta country' in song_df.loc[i]['Genre'] or 'black americana' in song_df.loc[i]['Genre'] or 'canadian americana' in song_df.loc[i]['Genre']:
            arr.append(1)
        elif 'arkansas country' in song_df.loc[i]['Genre'] or 'kentucky indie' in song_df.loc[i]['Genre']:
            arr.append(2)
        else:
            arr.append(3)
    song_df['Adj Genre'] = arr

    song_df.to_excel(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/songdata.xlsx')
    np.savetxt(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/features.csv', np.array([np.array(list(d[0].values())[:11]) for d in song_features if d[0] is not None]), delimiter=',')
    np.savetxt(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/adjlabels.csv', arr, delimiter=',')
    return song_df

def genre_build_data():

    #Initialize the Spotify client and hard code the genres that we want to highlight.
    #Here, we choose pop country, outlaw (folksier country), and bluegrass style as things to differentiate.
    spot = spotify_login(client_id, client_secret)
    playlists = ['Pop Country', 'Outlaw', 'Folk Country']
    test_playlist = 'Hot Country'

    #Store arrays for returning ids, labels.
    song_ids = []
    song_labels = []

    for i in range(len(playlists)):
        plist = playlists[i]
        dat = spot.search(plist, type='playlist', limit=1)
        print("Playlist Name: ", dat['playlists']['items'][0]['name'])
        testid = dat['playlists']['items'][0]['id']
        print("Playlist ID: ", testid)
        test_tresp = spot.playlist_tracks(testid)
        urllist = [test_tresp['items'][j]['track']['preview_url'] for j in range(len(test_tresp['items']))if test_tresp['items'][j]['track']['preview_url'] is not None]
        song_ids.extend(urllist)
        labels = np.full(len(urllist), i)
        song_labels.extend(labels)
    # song_labels = np.where(song_labels == 3, 2, song_labels)
    song_labels = np.array(song_labels, dtype=int)
    
    test_id = spot.search(test_playlist, type='playlist', limit=1)
    np.savetxt('spotify_song_urls.csv', song_ids, delimiter=',', fmt='%s')
    np.savetxt('spotify_genre_labels.csv', song_labels, delimiter=',')
    
    return song_ids, song_labels

def test_spotify_api_info():
    '''Testing function to understand the Spotify API calls and the returned objects.'''
    spot = spotify_login(client_id, client_secret)
    dat = spot.search('Pop Country', type='playlist', limit=1)
    # print(spot.categories())

    #  pprint(dat)

    print("Playlist Name: ", dat['playlists']['items'][0]['name'])
    testid = dat['playlists']['items'][0]['id']
    print("Playlist ID: ", testid)
    test_tresp = spot.playlist_tracks(testid)
    urllist = [test_tresp['items'][i]['track']['preview_url'] for i in range(len(test_tresp['items']))]
    # print((test_tresp['items'][0]['track']['id']))
    print(urllist)
    urlretrieve(urllist[0], './test.mp3')
    # while test_tresp['next']:
    #     test_tresp = spot.next(test_tresp)
    #     testout.extend(test_tresp['items'])
    # print("Output Tracks v0: ", testout)
    # for i in range(len(dat['artists']['items'])):
    #     print ("ARTIST: ", dat['artists']['items'][i]['name'])
    #     print("Genres: ", dat['artists']['items'][i]['genres'])
    #     print("Related Artists: ", [artist['name'] for artist in spot.artist_related_artists(dat['artists']['items'][i]['id'])['artists']])
    #     print("Artist Albums: ", [album['name'] for album in spot.artist_albums(dat['artists']['items'][i]['id'])['items']])


    return None

def download_data():
    # urls = pd.read_csv(r'spotify_song_urls.csv', sep=',', header=None) #.dropna().reset_index()
    urls = np.genfromtxt('spotify_song_urls.csv', delimiter=',', dtype='str')
    directory = r'spotify_song_data'
    files = Path(directory).glob('*') #Clears any existing data in the directory.
    for f in files:
        os.remove(f)
    for i in range(len(urls)):
        print("Pulling song: {}/{}".format(i, len(urls)))
        url = urls[i]
        urlretrieve(url, '{}/{}{}'.format(directory, '{}'.format(i), '.mp3'))
    return directory

def get_spectrograms():
    '''DEPRECATED: function that was originally used (on an old system) to extract and store spectrograms using Librosa.'''

    #Downloads spectrograms into folder, returns Adj Genre array from dataframe.
    df = pd.read_excel(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/songdata.xlsx').dropna().reset_index()
    directory = r'C:/Users/ragha/OneDrive/Desktop/spotify_project/song_data'
    savepath = r'C:/Users/ragha/OneDrive/Desktop/spotify_project/spec_images/'
    mels = []
    raw_labels = np.transpose(np.array(df['Adj Genre']))
    #song_features = np.array(df['Song Features'])
    files = Path(directory).glob('*')
    count = 0
    for f in files:
        count += 1
        y, sr = librosa.load(f)
        spec = librosa.feature.melspectrogram(y=y,sr=sr)
        if spec is not None:
            spec = np.flip(255*minmax_scale(spec)).astype(np.uint8)
            if len(spec.flatten()) == 165376:
                mels.append(np.array(spec.flatten()))
                print("{}/{}".format(count, len(os.listdir(directory))))
                skimage.io.imsave('{}/{}{}.png'.format(savepath, 'spec', str(count)), spec)
            else:
                pass
        else:
            pass
    np.savetxt(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/spectrograms.csv', np.stack(np.array(mels)).T)
    return raw_labels


def get_data(model):
    '''DEPRECATED: function on an old system that was used to load stored data and return in either standard song/label format
    or as a pair with similarity label for the Siamese model. Now using Torch DataLoaders.'''
    mels = np.loadtxt(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/spectrograms.csv', skiprows=5000)
    raw_labels = np.loadtxt(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/adjlabels.csv', skiprows=5000)
    features = np.loadtxt(r'C:/Users/ragha/OneDrive/Desktop/spotify_project/features.csv', delimiter=',', skiprows=5000)
    print("Mels Length", "Labels Length", "Features Length")
    if model == 'cnn':
        return (features, mels, raw_labels)
    if model == '':
        return (mels, raw_labels)
    if model == "siamese":
        adj_labels = np.where(raw_labels != 1, 1, 0)
        second_filter = np.nonzero(adj_labels)
        second_data = np.array(mels)[second_filter]
        first_data = [m for m in mels if m not in second_data]
        zero_pair_data = []
        zero_pair_labels = []
        for k in range(len(second_data)):
            #generate zero-pairs of mismatched data
            pair = np.zeros(2)
            pair[0] = first_data[k]
            pair[1] = second_data[k]
            zero_pair_data.append(pair)
            zero_pair_labels.append(0)
        one_pair_data = []
        one_pair_labels = []
        for k in range(len(second_data) - len(second_data)%2):
            #add matched pairs from second classifier
            pair = np.zeros(2)
            pair[0] = second_data[k]
            pair[1] = second_data[k+1]
            one_pair_data.append(pair)
            one_pair_labels.append(1)
        for k in range(len(second_data) - len(second_data)%2):
            #repeat loop to add matched pairs from first classifier
            pair = np.zeros(2)
            dat = first_data
            np.random.shuffle(dat)
            pair[0] = dat[k]
            pair[1] = dat[k+1]
            one_pair_data.append(pair)
            one_pair_labels.append(1)
        # print(np.array(zero_pair_data).shape, np.array(zero_pair_labels).shape)
        final_data = np.concatenate((zero_pair_data, one_pair_data))
        final_labels = np.concatenate((zero_pair_labels, one_pair_labels))
        # print(final_data.shape, final_labels.shape)
        return (final_data, final_labels)


def main():
    genre_build_data()
    download_data()

    return None

if __name__ == '__main__':
    main()