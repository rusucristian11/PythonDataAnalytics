import os
import pickle
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import parameters as p
from feature_extraction import extract_features as extract_features


### Test 70% (35 audio) -----control&parkinson
### Validare 30% (15 audio)


from prediction import test as pred

word = "ka-ka-ka" #ka-ka-ka
# test(word, 'test.wav')

class GetFiles:

    def __init__(self,dataset_path):
        # dataset: dataset is the root path for dataset(test,train,predict)
        self.dataset_path = dataset_path

    def getTestFiles(self):

        data_frame_row = []

        data_frame = pd.DataFrame()
        print('Root +++++++++++++++++++++++++++++++++++++')
        # flag = "test"

        # root test dierctory files listing
        speaker_audio_folder = os.listdir(self.dataset_path)
        print('Root test files', speaker_audio_folder)

        for folders in speaker_audio_folder:

            audio_files = os.listdir(self.dataset_path+"/"+folders)
            # listing of sub directory

            for files in audio_files:
                # path_to_audio =  self.dataset_path+"/"+folders+"/"+flag+"/"+files
                path_to_audio =  self.dataset_path+"/"+folders+"/"+files
                print('path_to_audio', path_to_audio)
                data_frame_row.append([path_to_audio,folders])

            data_frame = pd.DataFrame(data_frame_row,columns=['audio_path','actual'])


        # for folders in speaker_audio_folder:

        #     audio_files = os.listdir(self.dataset_path+"/"+folders)
        #     # listing of sub directory

        #     for files in audio_files:
        #         path_to_audio =  self.dataset_path+"/"+folders+"/"+files
        #         # path_to_audio =  self.dataset_path+"/"+folders+"/Parkinson/"+files
        #         print('path_to_audio', path_to_audio)
        #         data_frame_row.append([path_to_audio,folders])

        #     data_frame = pd.DataFrame(data_frame_row,columns=['audio_path','actual'])

        return data_frame


def getActualPredictedList():
    '''
    @return pd-frame : list of the actual and predicted list for confusion matrix calculation
    '''
    print("You are in getActualPredictedList")

    data_frame_row = []

    gf = GetFiles(dataset_path="C:\AUDIO-PROCESSED\Validation\ka-ka-ka")
    print("You are in getActualPredictedList ------get test files")
    
    testing_files =  gf.getTestFiles()
    print("You are in getActualPredictedList ------get test files for Control", testing_files)

    # testing_files =  gf.getTestFiles("Parkinson")


    for index, row in testing_files.iterrows():
        audio_path = row["audio_path"]
        print("Audio path for prediction is:_______________", audio_path)
        predicted = pred(word, audio_path) #word, 'test.wav'
        predicted = predicted.rsplit("\\", 1)[-1]

        actual = row["actual"]
        data_frame_row.append([actual, predicted])
    # print('actual_predicted BEFORE sorting', actual_predicted)
    # alphabetic sorting by column 'actual' without affecting the predicted column
    actual_predicted = pd.DataFrame(data_frame_row,columns = ['actual','predicted']).sort_values(by='actual')
    # print('actual_predicted AFTER sorting', actual_predicted)

    return actual_predicted


def showAccuracyPlotAndMeasure():
    actual_pred = getActualPredictedList()
    print("You are in showAccuracyPlotAndMeasure")

    actual = actual_pred["actual"].tolist()
    predicted = actual_pred["predicted"].tolist()
    labels  = sorted(actual_pred["actual"].unique().tolist()) # alphabetic sorting
    print("############actual", actual)
    print("############predicted", predicted)
    print("############labels", labels)
    
    cm = confusion_matrix(actual, predicted, labels=labels) #confusion matrix in matrix form
    
    #convert the table into a confusion matrix display.
    print("###################CM", cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of Recognition Model')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    display_numeric_accuracy(actual, predicted,labels) # displays the precision recall and fscore

def display_numeric_accuracy(actual,predicted,labels):
    '''
    @param list actual : actual label for the speaker's audio
    @param list predicted : predicted label by the GMM classifier
    @param list labels : name of the distinct speaker
    '''
    print("\n")
    print("You are in display_numeric_accuracy-----------------------------")
    print("classification_report################################",classification_report(actual, predicted, target_names=labels))
    	
    # print('Precision: %.3f' % precision_score(actual, predicted))
    # print('Recall: %.3f' % recall_score(actual, predicted))

showAccuracyPlotAndMeasure()