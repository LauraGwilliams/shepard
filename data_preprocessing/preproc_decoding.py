import numpy as np
import os
import os.path as op
from mne.io import read_raw_fif
from mne import (find_events, Epochs)
import pandas as pd

# params
subject = 'A0305'
meg_dir = '/Users/meglab/Desktop/shep_fifs/%s/'%(subject)
filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 0.6

# file names
raw_fname = meg_dir + subject+ '_shepard-raw.fif'
epochs_fname = meg_dir + subject+ '_shepard-epo.fif'

# read, filter
raw = read_raw_fif(raw_fname, preload=True)
raw = raw.filter(filt_l, filt_h)

# check events, create epochs
events = find_events(raw)  # the output of this is a 3 x n_trial np array
epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=None)

# set metadata, save epochs
trial_info = pd.read_csv(meg_dir + '%s_shepard_trialinfo.csv')
epochs.metadata = trial_info
epochs.save(epochs_fname)
