import pandas as pd
import csv, os
import math
import re
from random import shuffle

listdir = "/Users/ellieabrams/Desktop/Projects/Shepard/shepard/pretests/16bit-amp"

csv_out = []

for file in os.listdir(listdir):
    wav = os.path.basename(file)
    wav_split = re.split(r'[_.]', wav)
    freq = wav_split[0]
    amp_level = int(math.ceil(float('.%s'%((wav_split[2])))/.05))
    row = dict()
    row['wav_file'] = wav
    row['freq'] = freq
    row['amp_level'] = amp_level
    csv_out.append(row)


fieldnames = ['wav_file','freq','amp_level']
with open('psycho.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_out)
