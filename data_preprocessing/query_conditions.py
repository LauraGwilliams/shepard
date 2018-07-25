import os
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse
from mne import combine_evoked
from meg_preprocessing import (raw, epochs, evoked, inverse_operator,
                                inverse_operator_signed)

# stim shortcuts (for pilot subject only, in future use Eb)
lowA = '-A_220_300ms.wav'
highA = '-A_440_300ms.wav'
lowC = '-C_262_300ms.wav'
highC = '-C_524_300ms.wav'
lowEb = '-D_156_300ms.wav'
highEb = '-D_312_300ms.wav'

#order low to high: lowEb, lowA, lowC, highEb, highA, highC

par = 'partial'
pure = 'pure'
shep = 'shepard'

snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

# high and low notes of each scale, collapse across pure/partial/scale/random
high_A = epochs[(epochs.metadata['wav_file'] == par+highA) |
                (epochs.metadata['wav_file'] == pure+highA)]
low_A = epochs[(epochs.metadata['wav_file'] == par+lowA) |
                (epochs.metadata['wav_file'] == pure+lowA)]
high_C = epochs[(epochs.metadata['wav_file'] == par+highC) |
                (epochs.metadata['wav_file'] == pure+highC)]
low_C = epochs[(epochs.metadata['wav_file'] == par+lowC) |
                (epochs.metadata['wav_file'] == pure+lowC)]
high_Eb = epochs[(epochs.metadata['wav_file'] == par+highEb) |
                (epochs.metadata['wav_file'] == pure+highEb)]
low_Eb = epochs[(epochs.metadata['wav_file'] == par+lowEb) |
                (epochs.metadata['wav_file'] == pure+lowEb)]

#shepard tones collapsed
shep_A = epochs[epochs.metadata['wav_file'] == shep+lowA]
shep_C = epochs[epochs.metadata['wav_file'] == shep+lowC]
shep_Eb = epochs[epochs.metadata['wav_file'] == shep+lowEb]

#shepard tones in circular
shep_A_circ = epochs[(epochs.metadata['wav_file'] == shep+lowA) &
                    (epochs.metadata['circscale'] == 'circular')]
shep_C_circ = epochs[(epochs.metadata['wav_file'] == shep+lowC) &
                    (epochs.metadata['circscale'] == 'circular')]
shep_Eb_circ = epochs[(epochs.metadata['wav_file'] == shep+lowEb) &
                    (epochs.metadata['circscale'] == 'circular')]

partials = epochs[epochs.metadata['condition'] == 'partial']
partials_evoked = partials.average()

pure = epochs[epochs.metadata['condition'] == 'pure']
pure_evoked = pure.average()

high_A_evoked = high_A.average()
low_A_evoked = low_A.average()
# subtraction high - low
difference_evoked_A = combine_evoked([high_A_evoked,low_A_evoked],[1,-1])

high_C_evoked = high_C.average()
low_C_evoked = low_C.average()
difference_evoked_C = combine_evoked([high_C_evoked,low_C_evoked],[1,-1])

high_Eb_evoked = high_Eb.average()
low_Eb_evoked = low_Eb.average()
difference_evoked_Eb = combine_evoked([high_Eb_evoked,low_Eb_evoked],[1,-1])

difference_evoked_hi_lo = combine_evoked([high_C_evoked, low_Eb_evoked], [1,-1])

# unsigned data with original inverse_operator
high_A_evoked_stc = apply_inverse(high_A_evoked, inverse_operator, lambda2,
                                method='dSPM')
low_A_evoked_stc = apply_inverse(low_A_evoked, inverse_operator, lambda2,
                                method='dSPM')
high_C_evoked_stc = apply_inverse(high_C_evoked, inverse_operator, lambda2,
                                method='dSPM')
low_C_evoked_stc = apply_inverse(low_C_evoked, inverse_operator, lambda2,
                                method='dSPM')
high_Eb_evoked_stc = apply_inverse(high_Eb_evoked, inverse_operator, lambda2,
                                method='dSPM')
low_Eb_evoked_stc = apply_inverse(low_Eb_evoked, inverse_operator, lambda2,
                                method='dSPM')

# difference stcs with fixed orientation
difference_stc_A = apply_inverse(difference_evoked_A, inverse_operator_signed,
                                lambda2, method='dSPM')
difference_stc_C = apply_inverse(difference_evoked_C, inverse_operator_signed,
                                lambda2, method='dSPM')
difference_stc_Eb = apply_inverse(difference_evoked_Eb, inverse_operator_signed,
                                lambda2, method='dSPM')

tones = ['high_A','low_A','high_C','low_C','high_Eb','low_Eb']
stcs = [high_A_evoked_stc, low_A_evoked_stc, high_C_evoked_stc,low_C_evoked_stc,
        high_Eb_evoked_stc,low_Eb_evoked_stc]

for i in range(6):

    stcs[i].save('/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/stc/%s/R1201_shepard_%s'%(tones[i],tones[i]))

diff_folders = ['A_diff', 'C_diff', 'Eb_diff']
diff_stcs = [difference_stc_A, difference_stc_C, difference_stc_Eb]

for i in range(3):
    diff_stcs[i].save('/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/stc/%s/R1201_shepard_%s'%(diff_folders[i],diff_folders[i]))
