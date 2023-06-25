import os
import pickle
import numpy as np
import shutil
import librosa
import tensorflow as tf
import parameters as p

path = p.modelpath

def extract_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_normalized = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    # Pad or truncate the feature vector to match the expected dimensions
    if mfccs_normalized.shape[1] < 128:
        mfccs_normalized = np.pad(mfccs_normalized, ((0, 0), (0, 128 - mfccs_normalized.shape[1])))
    elif mfccs_normalized.shape[1] > 128:
        mfccs_normalized = mfccs_normalized[:, :128]
    return mfccs_normalized

def test(word, test_audio):
    waveform, sample_rate = librosa.load(test_audio, sr=None)
    feature_vector = extract_features(test_audio)

    modelpath = os.path.join(path, word)
    print("modelpath", modelpath)
    gmmModels = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

    models = [pickle.load(open(fname, 'rb')) for fname in gmmModels]
    person = [fname.split("/")[-1].split(".gmm")[0] for fname in gmmModels]
    log = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score_samples(feature_vector))
        print('Score for i =', i, 'is:', scores)
        log[i] = scores.sum()
    
    winner = np.argmax(log)
    print("=> Detected as person:", person[winner])
    speaker_detected = person[winner]
    print("Detected:", speaker_detected)
    return speaker_detected

word = "ka-ka-ka"
test(word, 'audio.wav') # Replace 'audio.wav' with your audio file path
