import numpy as np
import pickle
import os
from scipy.io.wavfile import read
from sklearn import mixture
import python_speech_features as mfcc
from sklearn import preprocessing
import parameters as p
import tensorflow as tf
import python_speech_features as mfcc

node = True

def extract_features(audio_path):
    file_path = os.path.join(p.source_train, audio_path)
    num_mfcc = 40  # Number of MFCC coefficients
    num_spectrogram_bins = 124  # Number of spectrogram bins
    winlen = int(0.025 * p.sample_rate)  # Window length in number of samples
    winstep = int(0.01 * p.sample_rate)  # Window step size in number of samples

    # Load the audio waveform
    waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)

    # Compute spectrogram using TensorFlow tutorial method
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.image.resize(spectrogram, [num_spectrogram_bins, num_spectrogram_bins])

    # Compute MFCCs from the spectrogram
    mfccs = mfcc.mfcc(signal=spectrogram, samplerate=sample_rate, winlen=winlen, winstep=winstep,
                      numcep=num_mfcc, nfilt=num_spectrogram_bins)
    mfccs = preprocessing.scale(mfccs)

    return mfccs

def featureExtraction(directoryName):
    features = np.asarray(()) #we created Array
    path = os.path.join(p.source_train, directoryName) 
    
    sources = [] #create a new list. We will take the .wav files in the folders in the training data/Username folder into this list.

    for name in os.listdir(p.source_train + directoryName): #TrainingData/x where x is the folder in it. This function will work for each folder.
        if name.endswith('.wav'): #If it is a wav file in TrainingData/x;
            nn = "{}".format(directoryName)+"/"+"{}".format(name) #Path
            sources.append(nn) #Adding it to our list.

    print('Sources', sources)


    for path in sources:    
        path = path.strip()   
        print("Path", path)
        # Read the voice
        sr,audio = read(p.source_train + path)
        print('sr extracted', sr)
        # sr = 16000
        print("Source+path", p.source_train + path)
        vector   = extract_features(audio,sr)
        if features.size == 0: #If we doesn't have any data
            features = vector 
            print("No featureS")#Features will equal to vector and program ends.
        else: 
            features = np.vstack((features, vector)) #We stack arrays vertically (on a row basis) sequentially.
            print("features.size", features.size)
        if node == True:    
            gmm = mixture.GaussianMixture(n_components = p.n_components, max_iter = p.max_iter,  covariance_type='diag',n_init = p.n_init) #We are calling gmm function.
            gmm.fit(features)
            # We save the models we calculated to the folder
            picklefile = path.split("/")[0]+".gmm"
            print('picklefile', picklefile)
            pickle.dump(gmm,open(p.modelpath + picklefile,'wb'))
            print("  >> Modeling complete for file: ",picklefile,' ',"| Data Point = ",features.shape   )
            features = np.asarray(()) 
