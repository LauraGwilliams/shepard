#!/usr/bin/env python
# -*- coding: utf-8 -*

# ea84@nyu.edu
# univariate analysis of shepard data
# goal: load epochs and trial_info with ica applied and epochs rejected

import numpy as np
import os
import eelbrain
import os.path as op
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution, convert_forward_solution)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse
from mne.preprocessing import ICA
from sklearn.decomposition import FastICA
import pandas as pd


subjects = ['A0216','A0314', 'A0306','A0270','A0280','A0305','A0307',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

# params
subject = 'A0365'
tmin = -0.2
tmax = 0.6

print (subject)
meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
raw_fname = meg_dir + subject+ '_shepard-raw.fif'
ica_fname = meg_dir + subject+ '_shepard_ica1-ica.fif'
ica_raw_fname = meg_dir + subject+ '_ica_shepard-raw.fif' # applied ica to raw
ica_rej_fname = meg_dir + subject+ '_shepard_rejfile.pickled' # rejected epochs after ica
epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
evoked_fname = meg_dir + subject+ '_shepard-evoked-ave.fif'

info = meg_dir + subject + '_shepard_trialinfo.csv'
rej_info = meg_dir + subject + '_shepard_rej_trialinfo.csv'
trial_info = pd.read_csv(info)

print ("Reading raw with ICA applied...")
raw = read_raw_fif(ica_raw_fname, preload=True)

print ("Finding events...")
events = find_events(raw)  # the output of this is a 3 x n_trial np array

print ("Epoching data...")
epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=(-0.2,0.6))

print ("Rejecting epochs...")
if op.isfile(ica_rej_fname):
    rejfile = eelbrain.load.unpickle(ica_rej_fname)
else:
    eelbrain.gui.select_epochs(epochs, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_shepard_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
    rejfile = eelbrain.load.unpickle(ica_rej_fname)

rejs = rejfile['accept'].x

# mask epochs and info
epochs = epochs[rejs]
trial_info = trial_info[rejs]
assert(len(epochs.events) == len(trial_info))

# save new epochs metadata, epochs
epochs.metadata = trial_info
epochs.metadata.to_csv(meg_dir+'/stcs/%s_shepard_rej_trialinfo.csv'%(subject))
epochs.save(epochs_fname)

# save evoked
evoked = epochs.average()
evoked.save(evoked_fname)
