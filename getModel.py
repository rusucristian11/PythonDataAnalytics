import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
import parameters as p
import tensorflow as tf
# from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

node = True
s_train_control = os.path.join(p.source_train, 'Control/')
s_train_parkinson = os.path.join(p.source_train, 'Parkinson/')
modelpath = p.modelpath

# Trains the data for a given word (ka-ka-ka or ta-ta-ta)
def enroll(name):
    # Path
    path = os.path.join(modelpath, name)

    # Create directory
    try:
        os.mkdir(path)
        print("Directory '%s' created" % name)
    except OSError as error:
        print(error)
    return path

def extract_features(file_path):
    # Load audio file
    waveform, _ = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)

    # Convert waveform to spectrogram
    spectrogram = tf.signal.stft(waveform, frame_length=256, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    # Convert spectrogram to log mel spectrogram
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=16000,
        lower_edge_hertz=0.0,
        upper_edge_hertz=8000.0
    )
    mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_spectrogram)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return mel_spectrogram

def featureExtraction(path, directoryName, m_path):
    features = np.asarray([])  # Initialize features as an empty array

    path = os.path.join(p.source_train, word)
    path += '/'
    sources = []
    for name in os.listdir(path + directoryName):
        if name.endswith('.wav'):
            nn = "{}".format(directoryName) + "/" + "{}".format(name)
            sources.append(nn)

    for path2 in sources:
        path2 = path2.strip()
        audio_path = os.path.join(path, path2)
        vector = extract_features(audio_path)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if node:
            gmm = mixture.GaussianMixture(n_components=p.n_components, max_iter=p.max_iter, covariance_type='diag',
                                          n_init=p.n_init)
            gmm.fit(features)

            # Save the models calculated to the folder
            picklefile = path2.split("/")[0] + ".gmm"
            pickle.dump(gmm, open(os.path.join(m_path, picklefile), 'wb'))
            print("  >> Modeling complete for file:", picklefile, "| Data Point =", features.shape[0])

            features = np.asarray([])

word = "ka-ka-ka"
m_path = enroll(word)
path_test = os.path.join(p.source_train, word)
# extract features from training data and put them in the folder "word"
featureExtraction(path_test, 'Parkinson/', m_path)
featureExtraction(path_test, 'Control/', m_path)
