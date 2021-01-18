import csv
import os
import io
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

def extractFeaturesToCSV():
    header = 'title chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' genre'
    header = header.split()  
    file = open('features.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    for class_dir in os.listdir('Samples'):
        for file in os.listdir('Samples/'+class_dir):
            songname = f'Samples/{class_dir}/{file}'
            y, sr = librosa.load(songname, mono=True, duration=5)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{file} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {class_dir}'
            file = open('features.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
    
def saveSpectrogramsAsPNG():
    if not os.path.exists(f'img_data'):
            os.mkdir(f'img_data')
    np.seterr(divide = 'ignore')
    cmap = plt.get_cmap('inferno')
    for class_dir in os.listdir('Samples'):
        if not os.path.exists(f'img_data/{class_dir}'):
            os.mkdir(f'img_data/{class_dir}')
        for count, file in enumerate(os.listdir('Samples/'+class_dir), 1):
            songname = f'Samples/{class_dir}/{file}'
            y, sr = librosa.load(songname, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'img_data/{class_dir}/{file[:-4]}.png')
            plt.clf()

def create_spectogram(track_id):
    filename = track_id
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def plot_spect(track_id):
    spect = create_spectogram(track_id)
    print(spect.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def saveSpectrogramsAsNPZ():
    X_spect = np.empty((0, 640, 128))
    count = 0
    genres = []
    for class_dir in os.listdir('Samples'):
        count = 0
        for file in os.listdir('Samples/'+class_dir):
            try:
                count += 1
                spect = create_spectogram(f'Samples/{class_dir}/{file}')
                spect = spect[:640, :]
                X_spect = np.append(X_spect, [spect], axis=0)
                genres.append(dict_genres[class_dir])
                if count % 100 == 0:
                    print("Currently processing: ", count)
            except:
                print("Couldn't process: ", count)
                continue
    y_arr = np.array(genres)
    lb = LabelBinarizer()
    y_arr = lb.fit_transform(y_arr)
    np.savez('data_arr', X_spect, y_arr)