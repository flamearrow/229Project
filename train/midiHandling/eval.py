# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:54:27 2017

@author: Andy Jiang
"""

from __future__ import print_function
import numpy as np
import mido
from mido import MidiFile
from mido import MidiTrack
from mido import MetaMessage

filename = 'MAPS_ISOL_CH0.3_F_StbgTGd2'
labels_file= 'output_labels_003.txt'

class Record:
    def __init__(self, note, start, stop):
        self.note = note
        self.start = start
        self.stop = stop


def postprocess(output_label):
    orig_mid = MidiFile('audio/alldata/CH/' + filename+'.mid')
    tempo = 439440
    ticks = orig_mid.ticks_per_beat
    #orig_mid.print_tracks()
    mid = MidiFile(ticks_per_beat = ticks)
    track = MidiTrack()
    mid.tracks.append(track)
    time = mido.second2tick(0.05, ticks, tempo)
    track.append(MetaMessage('set_tempo', tempo=439440))

    current_note = output_label[0]
    current_length = 0
    for i in range(len(output_label)):
        if (current_note == output_label[i]):
            current_length += 1
        else:
            msg = mido.Message('note_on', note=current_note, time=int(time/5))
            track.append(msg)
            msg = mido.Message('note_off', note=current_note, time=int(time*current_length))
            track.append(msg)
            current_note = output_label[i]
            current_length = 1

        
    #mid.print_tracks()
    print(orig_mid.length)
    print(mid.length)
    return mid

def test():
    orig_mid = MidiFile('audio/alldata/CH/' + filename+'.mid')
    tempo = 439440
    ticks = orig_mid.ticks_per_beat
    orig_mid.print_tracks()
    mid = MidiFile(ticks_per_beat = ticks)
    track = MidiTrack()
    mid.tracks.append(track)
    time = mido.second2tick(0.05, ticks, tempo)
    track.append(MetaMessage('set_tempo', tempo=439440))
    
    for i in range(50):
        msg = mido.Message('note_on', note=21+i, time=int(time/5))
        track.append(msg)
        msg = mido.Message('note_off', note=21+i, time=int(time*6))
        track.append(msg)
    #mid.print_tracks()
    return mid

labels = []
with open(labels_file) as txtfile:
    for line in txtfile:
        labels.append(int(line))
        
#mid = test()
#mid.save('audio/test.mid')
mid = postprocess(labels)
mid.save('audio/' + filename+ '_test.mid')
print('write to audio/' + filename+ '_test.mid')