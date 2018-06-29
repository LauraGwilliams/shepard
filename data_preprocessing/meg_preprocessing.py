#!/usr/bin/env python
# -*- coding: utf-8 -*

# leg5@nyu.edu
# univariate analysis of shepard data
# goal: convert raw files into epochs, and then apply source localisation

import numpy as np
import os.path as op
from mne.io import read_raw_fif
from mne import (pick_types, find_events, make_epochs, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.preprocessing import ICA

# params
subject = 'R1201'
filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 0.6

# file names
raw_fname = ''
ica_fname = ''
ica_raw_fname = ''
cov_fname = ''
mri_dir = ''
fwd_fname = ''
bem_fname = ''
src_fname = ''
trans_fname = ''
stc_epochs = ''

# if the ica-clean raw exists, load it
if op.isfile(ica_raw_fname):
    raw = read_raw_fif(ica_raw_fname, preload=True)

# else, make it
else:

    # step 1- concatenate data for each block
    raw = read_raw_fif(raw_fname, preload=True)

    # step 2- remove bad channels
    print raw.info['bads']  # check if any bad channels have been specified already
    raw.plot()  # visualise bad channels
    raw.info['bads'] = ['list_of_bad_channels']
    raw.save(raw_fname, overwrite=True)  # overwrite w/ bad channel info

    # step 3- apply ICA to the conjoint data
    picks = pick_types(raw.info, meg=True, exclude='bads')
    ica = ICA(n_components=0.95, method='fastica')

    # get ica components
    ica.exclude = []
    ica.fit(raw, picks=picks)
    ica.save(ica_fname, overwrite=True)  # save solution

    # view components and make rejections
    ica.plot_sources(raw)

    # apply ica to raw and save resulting clean raw file
    ica.apply(raw, copy=False)
    raw.save(ica_raw_fname)

# step 4- filter
raw = raw.filter(filt_l, filt_h)

# step 5- make epochs
events = find_events(raw)
epochs = make_epochs(raw, events, tmin=tmin, tmax=tmax, baseline=(-0.1, None))

# step 6- make noise covariance matrix
if not op.isfile(cov_fname):
    noise_cov = compute_covariance(epochs, tmax=0., method=['shrunk'])
    write_cov(cov_fname, noise_cov)
else:
    noise_cov = read_cov(cov_fname)

# step 6- make forward solution
if not op.isfile(fwd_fname):
    if not op.isfile(src_fname):
        src = setup_source_space(subject, fname=True, spacing='ico4',
                                 subjects_dir=mri_dir)
    else:
        src = src_fname
    fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                fname=fwd_fname, meg=True, ignore_ref=True)
else:
    fwd = read_forward_solution(fwd_fname)

# step 7- make inverse solution
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)


# step 8- make source estimates
snr = 2.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

# apply inverse
stc_epochs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                  method='dSPM')
np.save(file=stc_fname, arr=stc_epochs)
