import numpy as np
import pandas as pd
import os
import eelbrain
import os.path as op
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution, convert_forward_solution, compute_source_morph)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse
from mne.preprocessing import ICA

# params
subject = 'P011'
meg_dir = '/Users/ea84/Dropbox/shepard_preproc/%s/'%(subject) # change to local meg folder
filt_l = 0
filt_h = 60
tmin = -0.2
tmax = 0.6

# for plotting
os.environ["SUBJECTS_DIR"] = '/Users/ea84/Desktop/Projects/Shepard/analysis/mri' # change to local mri folder

# file names
trial_info_fname = meg_dir + subject + '_shepard_trialinfo.csv'
raw_fname = meg_dir + subject+ '_shepard-raw.fif'
ica_fname = meg_dir + subject+ '_shepard_ica1-ica.fif'
ica_raw_fname = meg_dir + subject+ '_ica_shepard-raw.fif' # applied ica to raw
ica_rej_fname = meg_dir + subject+ '_shepard_rejfile.pickled' # rejected epochs after ica
epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
evoked_fname = meg_dir + subject+ '_shepard-evoked-ave.fif'
cov_fname = meg_dir + subject+ '_shepard-cov.fif'
mri_dir = '/Users/ea84/Desktop/Projects/Shepard/analysis/mri/%s/bem/'%(subject) # subject's bem folder
fwd_fname = mri_dir+ subject+ '_shepard-fwd.fif'
inv_fname = mri_dir+ subject+ '_shepard-inv.fif'
bem_fname = mri_dir+ subject+ '-inner_skull-bem-sol.fif'
src_fname = mri_dir+ subject+ '-ico-4-src.fif'
trans_fname = mri_dir+ subject+ '-trans.fif'
stc_fname = meg_dir + subject+ '_shepard.stc.npy'
evoked_stc_fname = meg_dir + subject + '_evoked.stc'

def morph_stcs(stcs, subject):

    # reshape epochs into time points
    dummy_stc = stcs[0].copy()
    stc_data = np.array([s._data for s in stcs])
    stc_data = np.transpose(stc_data, (1, 0, 2))
    original_shape = stc_data.shape
    reshaped_stc = stc_data.reshape([original_shape[0], original_shape[1]*original_shape[2]])
    dummy_stc._data = reshaped_stc

    # morph in one go
    morphed_stc = compute_source_morph(dummy_stc, subject_from=subject,
                                           subject_to='fsaverage', spacing=4,
                                           subjects_dir=mri_dir).apply(dummy_stc)
    X = morphed_stc._data.reshape(original_shape)
    return np.array(X)


#--------------------------------------------
# START PREPROC

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

    # step 3- apply ICA to the conjoint data
    picks = pick_types(raw.info, meg=True, exclude='bads')
    ica = ICA(n_components=0.95, method='fastica')

    # get ica components
    ica.exclude = []
    ica.fit(raw, picks=picks)
    ica.save(ica_fname)  # save solution

    # view components and make rejections
    ica.plot_sources(raw)

    # apply ica to raw and save resulting clean raw file
    ica.apply(raw)
    raw.save(ica_raw_fname, overwrite=True)


# step 4- filter
raw = raw.filter(filt_l, filt_h)

# step 5- make epochs
events = find_events(raw)  # the output of this is a 3 x n_trial np array

# note: you may want to add some decimation here
epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=None)
trial_info = pd.read_csv(trial_info_fname)

# step 6- reject epochs based on threshold (2e-12)
# opens the gui, "mark" is to mark in red the channel closest to the eyes
if op.isfile(ica_rej_fname):
    rejfile = eelbrain.load.unpickle(ica_rej_fname)
else:
    eelbrain.gui.select_epochs(epochs, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
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


# step 8- make forward solution
if not op.isfile(fwd_fname):
    if not op.isfile(src_fname):
        src = setup_source_space(subject, spacing='ico4',
                                 subjects_dir=mri_dir)
        fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                 meg=True, ignore_ref=True)
    else:
        src = src_fname
        fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                fname=fwd_fname, meg=True, ignore_ref=True)
else:
    fwd = read_forward_solution(fwd_fname)

# step 9- make inverse solution for epochs
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

# step 10- make source estimates
snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

# apply inverse to epochs
stc_epochs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                  method='dSPM')

# morph the stc to the fsaverage
X = morph_stcs(stc_epochs, subject)

# check X is same length as trials
assert(X.shape[0] == len(trial_info))

# save
np.save(file=stc_fname, X)

# save the inverse operator
inverse_operator.save(inv_fname)

# apply inverse to evoked, this is not morphed
stc_evoked = apply_inverse(evoked, inverse_operator, lambda2,
                                method='dSPM')

stc_evoked.save(evoked_stc_fname)

# if you want to visualise and move along time course, confirm auditory response
# stc_evoked.plot(hemi = 'split', time_viewer=True)
# close the time_viewer window before the brain views to avoid crashing in terminal

# if weirdness happens with plotting
# import wx
# app = wx.App()
# frame = wx.Frame(None, -1, 'simple.py')
# frame.Show()
# app.MainLoop()
