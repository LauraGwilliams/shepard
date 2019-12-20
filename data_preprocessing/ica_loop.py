#!/usr/bin/env python
# -*- coding: utf-8 -*

# ea84@nyu.edu
# univariate analysis of shepard data
# goal: loop to create ICA for data with bad channels already removed

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
from mne.preprocessing import (ICA, read_ica)
from sklearn.decomposition import FastICA
import pandas as pd


subjects = ['A0216','A0314','A0306','A0270','A0280','A0305','A0307',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']
# params
filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 0.6

for subject in subjects:
    print (subject)
    meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
    raw_fname = meg_dir + subject+ '_shepard-raw.fif'
    ica_fname = meg_dir + subject+ '_shepard_ica1-ica.fif'
    ica_raw_fname = meg_dir + subject+ '_ica_shepard-raw.fif' # applied ica to raw

    print ("Reading raw file...")
    raw = read_raw_fif(raw_fname, preload=True)
    print ("Filtering data...")
    raw = raw.filter(filt_l, filt_h)

    print ("Finding events...")
    events = find_events(raw)  # the output of this is a 3 x n_trial np array

    print ("Epoching data...")
    epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=None)

    print ("Creating ICA object...")
    # apply ICA to the conjoint data
    picks = pick_types(raw.info, meg=True, exclude='bads')
    ica = ICA(n_components=0.9,method='fastica',max_iter=300)

    print ("Fitting epochs...")
    # get ica components
    ica.exclude = []
    ica.fit(epochs, picks=picks)

    print ("Saving ICA solution...")
    ica.save(ica_fname)  # save solution

#__________________________________________________________
subject = 'A0307'

meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
raw_fname = meg_dir + subject+ '_shepard-raw.fif'
ica_fname = meg_dir + subject+ '_shepard_ica1-ica.fif'
ica_raw_fname = meg_dir + subject+ '_ica_shepard-raw.fif' # applied ica to raw

# params
filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 0.6

print ("Reading raw file...")
raw = read_raw_fif(raw_fname, preload=True)

print ("Filtering data...")
raw = raw.filter(filt_l, filt_h)

print ("Reading ICA...")
ica = read_ica(ica_fname)
ica.plot_sources(raw)

ica.apply(raw)
raw.save(ica_raw_fname, overwrite=True)
