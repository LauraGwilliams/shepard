import pandas as pd
import csv, os
import math
from random import shuffle

listdir = "/Users/ellieabrams/Desktop/Projects/Shepard/shepard/pretests/16bit-pitch"

semitones = []
all_tones = []
all_freqs = []

for file in os.listdir(listdir):
    wav = os.path.basename(file)
    wav_split = wav.split('_')
    freq = '%s.%s'%(wav_split[0],wav_split[1].split('-')[0])
    all_tones.append(wav)
    all_freqs.append(freq)
    if wav[-5] == '0':
        semitones.append(wav)
    else:
        continue

csv_out = []

# for i in range(len(all_tones)):
for i in range(len(all_tones)):
    if all_tones[i] in semitones:
        for r in range(2):
            row = dict()
            row['wav_file1'] = all_tones[i]
            row['wav_file2'] = all_tones[i]
            row['freq1'] = all_freqs[i]
            row['freq2'] = all_freqs[i]
            row['distance'] = 0
            row['abs_dist'] = 0
            csv_out.append(row)
        for ii in range(-4,5):
            if (ii == 4) & (all_tones[i] != semitones[len(semitones) - 1]):
                continue
            row = dict()
            row['wav_file1'] = all_tones[i]
            row['wav_file2'] = all_tones[i+ii]
            row['freq1'] = all_freqs[i]
            row['freq2'] = all_freqs[i+ii]
            row['distance'] = abs(ii*25)
            row['abs_dist'] = ii*25
            csv_out.append(row)
            row = dict()
            row['wav_file1'] = all_tones[i+ii]
            row['wav_file2'] = all_tones[i]
            row['freq1'] = all_freqs[i+ii]
            row['freq2'] = all_freqs[i]
            row['distance'] = abs(ii*25)
            row['abs_dist'] = ii*25
            csv_out.append(row)
    else:
        continue

fieldnames = ['wav_file1', 'wav_file2','freq1','freq2','distance','abs_dist']
with open('pitch_discrim.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_out)
