import os
import youtube_dl
import json

def generator():
    return "Not implemented"

def get_top_50(genre):
    return []

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

def downloader(directory, validation_split, verbose=0):
    with open(os.path.join(directory, 'genres.txt')) as f:
        genres_list = f.readlines()
        for i in range(len(genres_list)):
            genres_list[i] = genres_list[i].split(',')[0]

    for genre in genres_list:
        if(verbose):
            print("Downloading top 50 songs for: ", genre)

        genre_training_directory = os.path.join(directory, 'training', genre)
        genre_validation_directory = os.path.join(directory, 'validation', genre)
        if(not os.path.isdir(genre_training_directory)):
            os.makedirs(genre_training_directory)
        if(not os.path.isdir(genre_validation_directory)):
            os.makedirs(genre_validation_directory)
        
        song_url_list = get_top_50(genre)
        for i in range(len(song_url_list)):
            if(i < len(song_url_list) * (1-validation_split)):
                destination = genre_training_directory
            else:
                destination = genre_validation_directory

            song_title = download_from_url(song_url_list[i], destination)
