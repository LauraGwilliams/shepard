# ea84@nyu.edu
# create evokeds/stcs for each tone, collapsed across scale/circular/random

import os
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse
from mne import combine_evoked
from meg_preprocessing import (raw, epochs, evoked, inverse_operator,
                                inverse_operator_signed)

# order low to high: lowEb, lowA, lowC, highEb, highA, highC
# 156, 175, 196, 208, 220, 233, 247, 262, 277, 294, 312, 330, 349, 370,
# 392, 415, 440, 494, 524

# Eb notes: 156, 175, 196, 208, 233, 262, 294, 312
# A notes: 220, 247, 277, 294, 330, 370, 415, 440
# C notes: 262, 294, 330, 349, 392, 440, 494, 524

meg_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/'

keys = ['A', 'C', 'Dsharp']

A_tones = ['A_220', 'A_247', 'A_277', 'A_294', 'A_330', 'A_370', 'A_415', 'A_440']
C_tones = ['C_262', 'C_294', 'C_330', 'C_349', 'C_392', 'C_440', 'C_494', 'C_524']
Eb_tones = ['D_156', 'D_175', 'D_196', 'D_208', 'D_233', 'D_262', 'D_294', 'D_312']

all_tones = [A_tones, C_tones, Eb_tones]

A_freqs = ['220', '247', '277', '294', '330', '370', '415', '440']
C_freqs = ['262', '294', '330', '349', '392', '440', '494', '524']
Eb_freqs = ['156', '175', '196', '208', '233', '262', '294', '312']

all_freqs = [A_freqs, C_freqs, Eb_freqs]

freqs = ['156', '175', '196', '208', '220', '233', '247', '262', '277', '294', '312', '330', '349', '370',
'392', '415', '440', '494', '524']

wav = '_300ms.wav'

par = 'partial-'
pure = 'pure-'
shep = 'shepard-'

# conditions = []
# for x in range(8):
#     conditions.extend((pure+A_tones[x],par+A_tones[x],pure+C_tones[x],
#                         par+C_tones[x],pure+Eb_tones[x],par+Eb_tones[x]))
#
# for x in range(len(conditions)):
#      os.mkdir(meg_dir + 'stc/%s'%(conditions[x]))

for x in range(len(freqs)):
    os.mkdir(meg_dir + 'stc/%sHz'%(freqs[x]))
    os.mkdir(meg_dir + 'stc/%sHzPure'%(freqs[x]))
    os.mkdir(meg_dir + 'stc/%sHzPartials'%(freqs[x]))

for x in range(8):
    os.mkdir(meg_dir + 'stc/%sHz'%(A_tones[x]))
    os.mkdir(meg_dir + 'stc/%sHz'%(C_tones[x]))
    os.mkdir(meg_dir + 'stc/%sHz'%(Eb_tones[x]))

snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

#-------------------------------------------------------------------------------
# SAVE EVOKEDS AND STCS

# all 48 conditions
for i in range(len(conditions)):
    temp_evoked = epochs[(epochs.metadata['wav_file'] == conditions[i]+wav)].average()
    temp_evoked.save(meg_dir + 'R1201/evokeds/R1201_shepard_%s-evoked-ave.fif'%(conditions[i]))
    temp_stc = apply_inverse(temp_evoked, inverse_operator, lambda2, method = 'dSPM')
    temp_stc.save(meg_dir + 'stc/%s/R1201_shepard_%s'%(conditions[i],conditions[i]))

# 19 frequencies collapsed over pure and partial
for j in range(len(freqs)):
    temp_evoked = epochs[((epochs.metadata['condition'] == 'pure') | (epochs.metadata['condition']
                            == 'partial')) & (epochs.metadata['freq'] == freqs[j])].average()
    temp_evoked.save(meg_dir + 'R1201/evokeds/R1201_shepard_%s-evoked-ave.fif'%(freqs[j]))
    temp_stc = apply_inverse(temp_evoked, inverse_operator, lambda2, method = 'dSPM')
    temp_stc.save(meg_dir + 'stc/%sHz/R1201_shepard_%s'%(freqs[j],freqs[j]))

# 19 pure frequencies
for k in range(len(freqs)):
    temp_evoked = epochs[(epochs.metadata['condition'] == 'pure') &
                    (epochs.metadata['freq'] == freqs[k])].average()
    temp_evoked.save(meg_dir + 'R1201/evokeds/R1201_shepard_%spure-evoked-ave.fif'%(freqs[k]))
    temp_stc = apply_inverse(temp_evoked, inverse_operator, lambda2, method = 'dSPM')
    temp_stc.save(meg_dir + 'stc/%sHzPure/R1201_shepard_%spure'%(freqs[k],freqs[k]))

# 19 partials frequencies
for l in range(len(freqs)):
    temp_evoked = epochs[(epochs.metadata['condition'] == 'partial') &
                    (epochs.metadata['freq'] == freqs[l])].average()
    temp_evoked.save(meg_dir + 'R1201/evokeds/R1201_shepard_%spar-evoked-ave.fif'%(freqs[l]))
    temp_stc = apply_inverse(temp_evoked, inverse_operator, lambda2, method = 'dSPM')
    temp_stc.save(meg_dir + 'stc/%sHzPartials/R1201_shepard_%spar'%(freqs[l],freqs[l]))

# 8 frequencies in each key collapsed pure/partial
for m in range(len(keys)):
    for n in range(8):
        temp_evoked = epochs[(epochs.metadata['key'] == keys[m]) &
                        (epochs.metadata['freq'] == all_freqs[m][n])].average()
        temp_evoked.save(meg_dir + 'R1201/evokeds/R1201_shepard_%s-evoked-ave.fif'%(all_tones[m][n]))
        temp_stc = apply_inverse(temp_evoked, inverse_operator, lambda2, method = 'dSPM')
        temp_stc.save(meg_dir + 'stc/%sHz/R1201_shepard_%s'%(all_tones[m][n],all_tones[m][n]))
