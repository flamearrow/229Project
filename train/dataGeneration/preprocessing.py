##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
from mido import MidiFile
import time
import librosa.display


class Record:
    def __init__(self, note, start):
        self.note = note
        self.start = start
        self.stop = 0

    def stop_note(self, time):
        self.stop = time


# 1. Get the file path to the included audio example
# filename = librosa.util.example_audio_file()
# filename = 'audio/Piano-midi.de/mp3/alb_esp1.mp3'
filename = 'audio/Piano-midi.de/mp3/c4-c7.mp3'
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)
# print(y)

mid = MidiFile('audio/Piano-midi.de/mp3/c4-c7.mid')
print(mid)

notes_map = list()
start = time.time()
for msg in mid.play():
    now = time.time() - start
    if msg.type == 'note_on':
        # print(str(now) + ':' + str(msg))
        # if the note is turning on
        if msg.velocity != 0:
            # add a new note in the map
            notes_map.append(Record(msg.note, now))
        # if the note is turning off
        else:
            for note in notes_map:
                if note.note == msg.note and note.start < now and note.stop == 0:
                    note.stop_note(now)
                    break
    if msg.type == 'note_off':
        # print(str(now) + ':' + str(msg))
        for note in notes_map:
            if note.note == msg.note and note.start < now and note.stop == 0:
                note.stop_note(now)
                break

step_size = 0.1
output = np.zeros((int(mid.length / step_size), 1))

for note in notes_map:
    print("note:" + str(note.note) + " start:" + str(note.start) + " stop:" + str(note.stop))
    index = list(range(int(note.start / step_size), int(note.stop / step_size)))
    output[index] = note.note

with open('output.csv', 'w') as f:
    for i in range(output.size):
        print(str(int(output[i][0])), file=f)

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))
C = librosa.cqt(y, sr=sr)

init = 0
index = 1
while init <= mid.length:
    idx = slice(*librosa.time_to_frames([init, init + step_size], sr=sr))
    plt.figure(figsize=(1, 1), dpi=32)
    librosa.display.specshow(librosa.amplitude_to_db(C[:, idx], ref=np.max), sr=sr)
    plt.savefig(filename[0:-4] + '_' + str(index) + '.jpg')
    init = init + step_size
    index = index + 1