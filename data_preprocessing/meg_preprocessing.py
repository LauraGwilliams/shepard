#!/usr/bin/env python
# -*- coding: utf-8 -*

# leg5@nyu.edu
# univariate analysis of shepard data
# goal: convert raw files into epochs, and then apply source localisation

import numpy as np
import os
import shutil
import eelbrain
import os.path as op
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution, convert_forward_solution, read_epochs, read_evokeds,
                 make_bem_model, make_bem_solution, write_bem_solution, write_forward_solution)
from mne.minimum_norm import (make_inverse_operator, apply_inverse_epochs, apply_inverse,
                            read_inverse_operator)
from mne.preprocessing import (ICA, read_ica)
from sklearn.decomposition import FastICA
import pandas as pd


subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

# params
subject = 'A0270'
meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
raw_dir = '/Users/ea84/Dropbox/shepard_decoding/%s/'%(subject)

trans_dir = '/Volumes/APOLLO/Shepard/meg/%s/'%(subject)

filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 0.6

# for plotting
os.environ["SUBJECTS_DIR"] = '/Volumes/APOLLO/Shepard/mri/%s'%(subject)

# file names
info = meg_dir + subject + '_shepard_trialinfo.csv'
raw_fname = meg_dir + subject+ '_shepard-raw.fif'
ica_fname = meg_dir + subject+ '_shepard_ica1-ica.fif'
ica_raw_fname = meg_dir + subject+ '_ica_shepard-raw.fif' # applied ica to raw
ica_rej_fname = meg_dir + subject+ '_shepard_rejfile.pickled' # rejected epochs after ica
epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
evoked_fname = meg_dir + subject+ '_shepard-evoked-ave.fif'
cov_fname = meg_dir + subject+ '_shepard-cov.fif'
mri_dir = '/Volumes/APOLLO/Shepard/mri/'
fwd_fname = meg_dir+ subject+ '_shepard-fwd.fif'
inv_fname = meg_dir+subject+'_shepard-inv.fif'
src_fname = meg_dir+ subject+ '-ico-4-src.fif'
trans_fname = trans_dir + subject+ '-trans.fif'
bem_fname = mri_dir+ '%s/bem/%s-inner_skull-bem-sol.fif'%(subject,subject)
stc_fname = meg_dir + subject+ '_shepard.stc.npy'

# if the ica-clean raw exists, load it
if op.isfile(ica_raw_fname):
    raw = read_raw_fif(ica_raw_fname, preload=True)

# else, make it
else:
    # step 1- concatenate data for each block
    raw = read_raw_fif(raw_fname, preload=True)

    # step 2- remove bad channels
    print (raw.info['bads'])  # check if any bad channels have been specified already
    raw.plot()  # visualise bad channels
    raw.info['bads'] = ['list_of_bad_channels']
    # interpolate bads and reset so that we have same number of channels for all blocks/subjects
    raw.interpolate_bads(reset_bads=True)
    raw.save(raw_fname, overwrite=True)  # overwrite w/ bad channel info/interpolated bads

    # step 3- filter
    raw = raw.filter(filt_l, filt_h)

    # step 4- apply ICA to the conjoint data
    picks = pick_types(raw.info, meg=True, exclude='bads')
    ica = ICA(n_components=0.9,method='fastica')

    # get ica components
    ica.exclude = []
    ica.fit(raw, picks=picks)
    ica.save(ica_fname)  # save solution

    # view components and make rejections
    ica.plot_sources(raw)

    # apply ica to raw and save resulting clean raw file
    ica.apply(raw)
    raw.save(ica_raw_fname, overwrite=True)

# load logfile
trial_info = pd.read_csv(info)

# step 5- make epochs
events = find_events(raw)  # the output of this is a 3 x n_trial np array

# note: you may want to add some decimation here
epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=None)

# step 6- reject epochs based on threshold
# opens the gui, "mark" is to mark in red the channel closest to the eyes
if op.isfile(ica_rej_fname):
    rejfile = eelbrain.load.unpickle(ica_rej_fname)
else:
    eelbrain.gui.select_epochs(epochs, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_shepard_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
    rejfile = eelbrain.load.unpickle(ica_rej_fname)

# create a mask to reject the bad epochs
rejs = rejfile['accept'].x

# mask epochs and info
epochs = epochs[rejs]
trial_info = trial_info[rejs]

# if evoked is made, load
# else make evoked to check for auditory response
if op.isfile(evoked_fname):
    evoked = Evoked(evoked_fname)
else:
    evoked = epochs.average()
    evoked.save(evoked_fname)
    # check for M100
    evoked.plot()

# set metadata: allows you to specify more complex info about events,
# can use pandas-style queries to access subsets of data
epochs.metadata = trial_info
epochs.save(epochs_fname)

# SANITY CHECK!!:
assert(len(epochs.events) == len(trial_info))

#-------------------------------------------------------------------------------

# step 7- make noise covariance matrix
if not op.isfile(cov_fname):
    noise_cov = compute_covariance(epochs, tmax=0., method=['shrunk'])
    write_cov(cov_fname, noise_cov)
else:
    noise_cov = read_cov(cov_fname)

# if using native MRI, need to make_bem_model
if not op.isfile(bem_fname):
   surfaces = make_bem_model(subject, ico=4, conductivity=(0.3,), subjects_dir=mri_dir, verbose=None)
   bem = make_bem_solution(surfaces)
   write_bem_solution(bem_fname, bem)

# step 8- make forward solution
if not op.isfile(fwd_fname):
    if not op.isfile(src_fname):
        src = setup_source_space(subject, spacing='ico4',
                                 subjects_dir=mri_dir)
        fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                 meg=True, ignore_ref=True)
        write_forward_solution(fwd_fname, fwd)
    else:
        src = src_fname
        fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                fname=fwd_fname, meg=True, ignore_ref=True)
        write_forward_solution(fwd_fname, fwd)
else:
    fwd = read_forward_solution(fwd_fname)

# create forward solution with free-orientation and in surface orientation
fwd_fixed = convert_forward_solution(fwd, force_fixed=False,
                                                surf_ori=True)

# step 9- make inverse solution for epochs
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

inverse_operator_signed = make_inverse_operator(epochs.info, fwd_fixed, noise_cov,
                                                depth=0.8, fixed=True)

# step 10- make source estimates
snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

# apply inverse to epochs
stc_epochs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                  method='dSPM')
# save as numpy array
np.save(file=stc_fname, arr=stc_epochs)

# apply inverse to evoked
stc_evoked = apply_inverse(evoked, inverse_operator, lambda2,
                                method='dSPM')

# visualise and move along time course, confirm auditory response
stc_evoked.plot(hemi = 'split', time_viewer=True)
#close the time_viewer window before the brain views to avoid crashing in terminal

# if weirdness happens with plotting
import wx
app = wx.App()
frame = wx.Frame(None, -1, 'simple.py')
frame.Show()
app.MainLoop()

#__________________________________________________________
# QUICK LOAD stcs

# grab trans.fifs
for subject in subjects:
    trans_fname = '/Volumes/APOLLO/Shepard/meg/%s/%s-trans.fif'%(subject,subject)
    meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
    shutil.copy(trans_fname,meg_dir)

import os
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution, convert_forward_solution, read_epochs,
                 read_evokeds)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse

# params
subject = 'A0354'
meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
evoked_fname = meg_dir + subject+ '_shepard-evoked-ave.fif'
cov_fname = meg_dir + subject+ '_shepard-cov.fif'
mri_dir = '/Volumes/APOLLO/Shepard/mri/'
fwd_fname = meg_dir+ subject+ '_shepard-fwd.fif'
inv_fname = meg_dir+subject+'_shepard-inv.fif'
src_fname = meg_dir+ subject+ '-ico-4-src.fif'
trans_fname = trans_dir + subject+ '-trans.fif'
bem_fname = mri_dir+ '%s/bem/%s-inner_skull-bem-sol.fif'%(subject,subject)
stc_fname = meg_dir + subject+ '_shepard.stc.npy'
os.environ["SUBJECTS_DIR"] = '/Users/ea84/Dropbox/shepard_mri'

# load
epochs = read_epochs(epochs_fname)
evoked = epochs.average()

# step 7- make noise covariance matrix
if not op.isfile(cov_fname):
    noise_cov = compute_covariance(epochs, tmax=0., method=['shrunk'])
    write_cov(cov_fname, noise_cov)
else:
    noise_cov = read_cov(cov_fname)

# if using native MRI, need to make_bem_model
if not op.isfile(bem_fname):
   surfaces = make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=mri_dir, verbose=None)
   bem = make_bem_solution(surfaces)
   write_bem_solution(bem_fname, bem)

# step 8- make forward solution
if not op.isfile(fwd_fname):
    if not op.isfile(src_fname):
        src = setup_source_space(subject, spacing='ico4',
                                 subjects_dir=mri_dir)
        fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                 meg=True, ignore_ref=True)
        write_forward_solution(fwd_fname, fwd)
    else:
        src = src_fname
        fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                fname=fwd_fname, meg=True, ignore_ref=True)
        write_forward_solution(fwd_fname, fwd)
else:
    fwd = read_forward_solution(fwd_fname)

# create forward solution with free-orientation and in surface orientation
fwd_fixed = convert_forward_solution(fwd, force_fixed=False,
                                                surf_ori=True)

# step 9- make inverse solution for epochs
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

inverse_operator_signed = make_inverse_operator(epochs.info, fwd_fixed, noise_cov,
                                                depth=0.8, fixed=True)

# inverse and apply
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

stc_epochs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                  method='dSPM')

stc_evoked = apply_inverse(evoked, inverse_operator, lambda2,
                                method='dSPM')

# visualise and move along time course, confirm auditory response
stc_evoked.plot(hemi = 'split', time_viewer=True)

#__________________________________________________________
# RANDOM

# make subject directories
for subject in subjects:
    if not os.path.exists('/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)):
        os.makedirs('/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject))

orig = '/Users/ea84/Dropbox/shepard_decoding/'
dest = '/Users/ea84/Dropbox/shepard_sourceloc/'
for subject in subjects:
    shutil.copy(orig+'%s/%s_shepard_trialinfo.csv'%(subject,subject),dest+'%s'%(subject))

# concatenate fifs
raw_fname = meg_dir + subject + 'shepard-raw.fif'
raw1 = mne.io.read_raw_kit(raw_fname, preload=True, slope='+')
raw_fname = '/Volumes/MEG/NYUAD-Lab-Server/MEGPC/A0301-A0350/A0305/A0305_ShepardUpdated/A0305_shep2_NR.con'
raw2 = mne.io.read_raw_kit(raw_fname, preload=True, slope='+')
raw = mne.concatenate_raws([raw1, raw2])

# quick bad channels check
subject = 'P022'
meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
raw_fname = meg_dir + subject+ '_shepard-raw.fif'
raw = read_raw_fif(raw_fname, preload=True)

print (raw.info['bads'])  # check if any bad channels have been specified already
raw.plot()  # visualise bad channels

if raw.info['bads'] != []:
    raw.interpolate_bads(reset_bads=True)
    raw.save(raw_fname, overwrite=True)  # overwrite w/ bad channel info/interpolated bads
