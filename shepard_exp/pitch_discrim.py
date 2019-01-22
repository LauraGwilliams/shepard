import pandas as pd
import csv, os
import math
from random import shuffle

listdir = "/Users/ellieabrams/Desktop/NeLLab/shepard/stimuli/pretests/pitch"

semitones = []
all_tones_temp = []
all_tones = []
all_freqs_temp = []
all_freqs = []


for file in os.listdir(listdir):
    wav = os.path.basename(file)
    wav_split = wav.split('-')
    freq = wav_split[0]
    all_tones.append(wav)
    all_freqs.append(freq)
    if (wav[-5] == '0') & (wav[0:3] != '208') & (wav[0:3] != '659'):
        semitones.append(wav)
    else:
        continue

# for i in range(len(all_tones_temp)):
#     if i == 0:
#         all_tones.append(all_tones_temp[len(all_tones_temp)-1])
#     else:
#         all_tones.append(all_tones_temp[i-1])
#
# for i in range(len(all_freqs_temp)):
#     if i == 0:
#         all_freqs.append(all_freqs_temp[len(all_freqs_temp)-1])
#     else:
#         all_freqs.append(all_freqs_temp[i-1])

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
            row['abs_dist'] = 0
            row['dist_semi'] = 0
            row['direction'] = 0
            csv_out.append(row)
        for ii in range(-4,5):
            if (ii == 4) & (all_tones[i] != semitones[len(semitones) - 1]):
                continue
            row = dict()
            row['wav_file1'] = all_tones[i]
            row['wav_file2'] = all_tones[i+ii]
            row['freq1'] = all_freqs[i]
            row['freq2'] = all_freqs[i+ii]
            row['abs_dist'] = abs(ii*25)
            row['dist_semi'] = ii*25
            if row['freq1'] > row['freq2']:
                row['direction'] = 1
            else:
                row['direction'] = 2
            csv_out.append(row)
            row = dict()
            row['wav_file1'] = all_tones[i+ii]
            row['wav_file2'] = all_tones[i]
            row['freq1'] = all_freqs[i+ii]
            row['freq2'] = all_freqs[i]
            row['abs_dist'] = abs(ii*25)
            row['dist_semi'] = ii*25
            if row['freq1'] > row['freq2']:
                row['direction'] = 1
            else:
                row['direction'] = 2
            csv_out.append(row)
    else:
        continue

fieldnames = ['wav_file1', 'wav_file2','freq1','freq2','abs_dist','dist_semi','direction']
with open('pitch_discrim.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_out)
