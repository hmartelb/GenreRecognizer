import json
import os

import keras
import numpy as np
import requests
import soundcloud
import youtube_dl


class Generator(keras.utils.Sequence):
    def __init__(self, filenames, genres, batch_size=32, dim=(44100, 1), shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.filenames = filenames
        self.genres = genres
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]         # Generate indexes of the batch
        filenames_temp = [self.filenames[k] for k in indexes]                           # Find list of filenames
        x,y = self.__data_generation(filenames_temp)                                    # Generate data
        return x,y

    def get_label(self, filename):
        folder = os.path.split(os.path.dirname(filename))[-1]
        label  = self.genres.index(folder)
        return keras.utils.to_categorical(label, num_classes=len(self.genres))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames_temp):
        'Generates data containing batch_size samples'                                  
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size)
        # Generate data
        for i, filename in enumerate(filenames_temp):
            x[i,] = np.load(filename)[:,0:self.dim[1]]
            y[i,] = self.get_label(filename)
        return x,y

def get_top_songs(genre, limit, token):
    top = requests.get('https://api-v2.soundcloud.com/charts?kind=top&genre=soundcloud:genres:' + genre + '&limit=' + str(limit) + '&oauth_token=' + token).json()
    return top

def download_from_url(url, destination):
    def dl_hook(d):
        if d['status'] == 'finished':
            print('Download finished "' + d['filename'] + '" [' + d['_total_bytes_str'] + ']')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': destination+'/%(title)s.%(ext)s',      
        'download_archive': 'downloaded_songs.txt',  
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'progress_hooks': [dl_hook],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_title = info_dict.get('title', None)
        ydl.download([url])
        return video_title + '.mp3'

def downloader(directory, limit, validation_split, verbose=0):
    with open(os.path.join(directory, 'genres.txt')) as f:
        genres_list = f.readlines()
        for i in range(len(genres_list)):
            genres_list[i] = genres_list[i].split(',')[0]

    for genre in genres_list:
        if(verbose):
            print('Downloading top ', str(limit), ' songs for: ', genre)

        genre_training_directory = os.path.join(directory, 'training', genre)
        genre_validation_directory = os.path.join(directory, 'validation', genre)
        if(not os.path.isdir(genre_training_directory)):
            os.makedirs(genre_training_directory)
        if(not os.path.isdir(genre_validation_directory)):
            os.makedirs(genre_validation_directory)
        
        token = ''
        song_url_list = get_top_songs(genre, limit, token)
        # for i in range(len(song_url_list)):
        #     if(i < len(song_url_list) * (1-validation_split)):
        #         destination = genre_training_directory
        #     else:
        #         destination = genre_validation_directory

        #     song_title = download_from_url(song_url_list[i], destination)

if __name__ == '__main__':
    directory = os.path.join('..','dataset')
    
    downloader(directory, validation_split=0.1, verbose=1)
