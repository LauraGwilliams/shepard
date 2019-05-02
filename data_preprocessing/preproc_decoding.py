import mne
import os.path as op
import numpy as np
from mne.io import read_raw_fif
from mne import (find_events, Epochs)
import pandas as pd
import glob

# params
subjects = ['A0364']

for subject in subjects:
    meg_dir = '/Users/ea84/Dropbox/shepard_decoding/%s/'%(subject)
    filt_l = 1  # same as aquisition
    filt_h = 60
    tmin = -0.2
    tmax = 0.6

    # file names
    raw_fname = meg_dir + '%s_shepard-raw.fif'%(subject)
    epochs_fname = meg_dir + '%s_shepard-epo.fif'%(subject)

    server_dir = '/Users/ea84/Dropbox/shepard_preproc/'

    # read, filter
    if op.isfile(raw_fname):
        raw = read_raw_fif(raw_fname, preload=True)
    else:
        con_fname = server_dir + '%s/%s_shep1_NR.con'%(subject,subject)
        raw1 = mne.io.read_raw_kit(con_fname, preload=True, slope='+')
        con_fname = server_dir + '%s/%s_shep2_NR.con'%(subject,subject)
        raw2 = mne.io.read_raw_kit(con_fname, preload=True, slope='+')
        if len(glob.glob(server_dir + '%s/*.con'%(subject))) > 2:
            con_fname = server_dir + '%s/%s_shep3_NR.con'%(subject,subject)
            raw3 = mne.io.read_raw_kit(con_fname, preload=True, slope='+')
            raw = mne.concatenate_raws([raw1, raw2, raw3])
        else:
            raw = mne.concatenate_raws([raw1, raw2])

        raw.save(raw_fname)

    raw = raw.filter(filt_l, filt_h)

    # check events, create epochs
    events = find_events(raw)  # the output of this is a 3 x n_trial np array
    epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=None)

    # set metadata, save epochs
    trial_info = pd.read_csv(meg_dir + '%s_shepard_trialinfo.csv'%(subject))
    epochs.metadata = trial_info
    epochs.save(epochs_fname)
