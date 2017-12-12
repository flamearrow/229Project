##################
# Standard imports
from __future__ import print_function
import numpy as np
import glob
import matplotlib.pyplot as plt
import librosa
from mido import MidiFile
import librosa.display


class Record:
    def __init__(self, note, start, stop):
        self.note = note
        self.start = start
        self.stop = stop


def preprocess(filename):
    y, sr = librosa.load(filename+'.wav')
    
    mid = MidiFile(filename+'.mid')
    
    notes_map = list()
    with open(filename+'.txt') as txtfile:
        txtfile.readline()
        for line in txtfile:
            data = line.split('\t')
            notes_map.append(Record(int(data[2]),float(data[0]),float(data[1])))
    
    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))
    C = librosa.cqt(y, sr)
    
    step_size = 0.05
    index = int(notes_map[0].start/step_size)
    init = index * step_size
    index = 1
    while init <= mid.length:
        idx = slice(*librosa.time_to_frames([init,init+step_size], sr=sr))
        plt.figure(figsize=(1, 1), dpi=32)
        librosa.display.specshow(librosa.amplitude_to_db(C[:,idx], ref=np.max),sr=sr)
        temp = filename.split('_')
        plt.savefig('audio/OutputEval/'+ temp[2] + '_' + temp[3] + '_' + str(index) + '.jpg')
        #plt.savefig('audio/OutputEval/'+ temp[3] + '_' + temp[4] + '_' + temp[5] + '_' + str(index) + '.jpg')
        plt.close('all')
        init = init + step_size
        index = index + 1

#filenames = glob.glob('audio/alldata/CH/MAPS_ISOL_CH0.3_F_StbgTGd2.mid')
filenames = glob.glob('audio/alldata/NO/*.mid')
for file in filenames:
    #print(file[0:-4])
    preprocess(file[0:-4])