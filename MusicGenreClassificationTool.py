import csv
import cv2
import os
import io
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer   
from sklearn.metrics import confusion_matrix
import itertools

def extractFeaturesToCSV():
    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
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
            to_append = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {class_dir}'
            file = open('features.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

def loadFeatures(indir):
    data = pd.read_csv(indir)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(genre_list)
    scaler = StandardScaler()
    samples = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    return samples, labels                

def saveSpectrogramsToPNG():
    if not os.path.exists(f'img_data'):
            os.mkdir(f'img_data')
    np.seterr(divide = 'ignore')
    cmap = plt.get_cmap('inferno')
    for class_dir in os.listdir('Samples'):
        if not os.path.exists(f'img_data/{class_dir}'):
            os.mkdir(f'img_data/{class_dir}')
        for count, file in enumerate(os.listdir('Samples/'+class_dir), 1):
            songname = f'Samples/{class_dir}/{file}'
            y, sr = librosa.load(songname, mono=True)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'img_data/{class_dir}/{file[:-4]}.png')
            plt.clf()
            
def loadImages(indir, image_size):
    samples = []
    labels = []
    for class_dir in os.listdir(indir):
        the_class = class_dir
        for file in os.listdir(indir+'/'+class_dir):
            image = cv2.imread("{}/{}/{}".format(indir,class_dir,file))
            image = cv2.resize(image, image_size)
            samples.append(image)
            labels.append(the_class)
    samples = np.array(samples)
    labels = np.array(labels)
    print('loaded',len(samples),' samples')
    print('classes',set(labels))
    
    # one-hot labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    print("Labels shape",labels.shape)
    labels = labels.astype(float)
    
    return samples,labels

def makeSpectrogram(track_id):
    filename = track_id
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def plotSpectrogram(track_id):
    spect = makeSpectrogram(track_id)
    print(spect.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def saveSpectrogramsToNPZ():
    X_spect = np.empty((0, 640, 128))
    count = 0
    genres = []
    for class_dir in os.listdir('Samples'):
        count = 0
        X_spect_temp = np.empty((0, 640, 128))
        for file in os.listdir('Samples/'+class_dir):
            try:
                count += 1
                spect = makeSpectrogram(f'Samples/{class_dir}/{file}')
                spect = spect[:640, :]
                X_spect_temp = np.append(X_spect_temp, [spect], axis=0)
                genres.append(class_dir)
                if count % 100 == 0:
                    print("Currently processing: ", count, " in", class_dir)
            except:
                print("Couldn't process: ", count)
                continue
        X_spect = np.concatenate((X_spect,X_spect_temp),axis=0)
    y_arr = np.array(genres)
    lb = LabelBinarizer()
    y_arr = lb.fit_transform(y_arr)
    np.savez('data_arr', X_spect, y_arr)

def loadArrays(indir):
    npzfile = np.load(indir)
    print(npzfile.files)
    samples = npzfile['arr_0']
    labels = npzfile['arr_1']
    print(samples.shape, labels.shape)
    return samples, labels
    
def makeDictionary():
    dict_genres = {}

    for i, genre_name in enumerate(os.listdir('Samples'), 0):
        dict_genres[genre_name] = i

    return dict_genres   

def showSummaryStats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()