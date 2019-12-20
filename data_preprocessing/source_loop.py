import numpy as np
import eelbrain
import pandas as pd
import os
import os.path as op
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
from mne.preprocessing import ICA
from sklearn.decomposition import FastICA
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution, convert_forward_solution, read_epochs,
                 read_evokeds, write_forward_solution, make_bem_model, make_bem_solution,
                 write_bem_solution, extract_label_time_course, read_labels_from_annot)
from mne.minimum_norm import (make_inverse_operator, apply_inverse_epochs, apply_inverse,
                            write_inverse_operator, read_inverse_operator)
from surfer import Brain
from surfer.io import read_stc

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307',
            'A0314','A0323','A0326','A0344','A0345','A0353',
            'A0354','A0355','A0358','A0362','A0364','A0365',
            'A0367','A0368','A0369','A0370','P010','P011',
            'P014','P015','P022']

###############################################################################
####################### MAKE ICA SOLUTION and save ############################
###############################################################################

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

###############################################################################
################# PLOT ICA COMPONENTS FOR SINGLE SUBJECT ######################
###############################################################################

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

###############################################################################
######## MAKE EPOCHS, EPOCH REJECTION, SAVE NEW TRIAL INFO AND EVOKED #########
###############################################################################

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

###############################################################################
################# MAKE COV, BEM, FORWARD, INVERSE SOLUTION ####################
###############################################################################

for subject in subjects:

    print (subject)

    meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
    mri_dir = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri/'

    epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
    evoked_fname = meg_dir + subject+ '_shepard-evoked-ave.fif'
    cov_fname = meg_dir + subject+ '_shepard-cov.fif'
    fwd_fname = meg_dir+ subject+ '_shepard-fwd.fif'
    inv_fname = meg_dir+subject+'_shepard-inv.fif'
    src_fname = meg_dir+ subject+ '-ico-4-src.fif'
    trans_fname = meg_dir + subject+ '-trans.fif'
    bem_fname = mri_dir+ '%s/bem/%s-inner_skull-bem-sol.fif'%(subject,subject)
    stc_fname = meg_dir + subject+ '_shepard.stc.npy'
    os.environ["SUBJECTS_DIR"] = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri/'

    # load
    epochs = read_epochs(epochs_fname)

    print ("Making covariance matrix...")
    noise_cov = compute_covariance(epochs, tmax=0., method=['shrunk'])
    write_cov(cov_fname, noise_cov)

    noise_cov = read_cov(cov_fname)

    print ("Making BEM model...")
    surfaces = make_bem_model(subject, ico=4, conductivity=(0.3,), subjects_dir=mri_dir, verbose=None)
    bem = make_bem_solution(surfaces)
    write_bem_solution(bem_fname, bem)

    # if not op.isfile(fwd_fname):
    print ("Making forward solution...")
    src = setup_source_space(subject, spacing='ico4',
                                 subjects_dir=mri_dir)
    fwd = make_forward_solution(epochs.info, trans_fname, src, bem_fname,
                                 meg=True, ignore_ref=True)
    print ("Converting forward solution...")
    fwd_fixed = convert_forward_solution(fwd, force_fixed=False,
                                                surf_ori=True)
    write_forward_solution(fwd_fname, fwd_fixed, overwrite=True)
    # else:
    #     fwd = read_forward_solution(fwd_fname)


    print ("Making inverse operator...")
    inverse_operator = make_inverse_operator(epochs.info, fwd_fixed, noise_cov, fixed=True,
                                        loose = 0.0)
    write_inverse_operator(inv_fname, inverse_operator)

    print ("Done with %s!"%(subject))

###############################################################################
#################  MAKE EPOCHS STCS, SAVE AVG OF 4 LABELS #####################
###############################################################################

snr = 2.0  # Standard assumption for single trial
lambda2 = 1.0 / snr ** 2

mri_dir = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri'

# params
for subject in subjects:

    print (subject)
    meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
    stc_fname = meg_dir+'stcs/%s_shepard_labels.npy'%(subject)

        # paths
    epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
    inv_fname = meg_dir+subject+'_shepard-inv.fif'

    print ("Loading variables...")
    epochs = read_epochs(epochs_fname)
    inverse_operator = read_inverse_operator(inv_fname)
    src = inverse_operator['src']

    # apply inverse to epochs
    print ("Creating stcs...")
    stc_epochs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                      method='dSPM')
    labels = read_labels_from_annot(subject, parc='aparc', subjects_dir=mri_dir)

    # HG and STC labels
    rois = ['transversetemporal','superiortemporal']
    hemis = ['lh','rh']

    lbs = []
    for roi in rois:
        for hemi in hemis:
            lbs.append([label for label in labels if label.name == '%s-%s'%(roi,hemi)][0])

    print ("Extracting labels...")
    stcs = extract_label_time_course(stc_epochs, lbs, src)

    stc_arr = np.transpose(np.array(stcs), (1,0,2))

    print ("Saving array!")
    # save as numpy array
    np.save(file=stc_fname, arr=stc_arr)

    print ("Next subject!")

###############################################################################
##############  MAKE GRAND AVERAGE MOVIE OF SINGLE SUBJECT EVOKED #############
###############################################################################
snr = 3.0  # Standard assumption for average data
lambda2 = 1.0 / snr ** 2

def time_label(t):
    return "time=%0.2f ms" % (1e3 * t)

for subject in subjects:
    print (subject)
    meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/%s/'%(subject)
    os.environ["SUBJECTS_DIR"] = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri/'
    evoked_fname = meg_dir + subject+ '_shepard-evoked-ave.fif'
    inv_fname = meg_dir+subject+'_shepard-inv.fif'
    stc_fname = meg_dir+subject+'_shepard_evoked'

    evoked = Evoked(evoked_fname)
    inverse_operator = read_inverse_operator(inv_fname)
    stc_evoked = apply_inverse(evoked, inverse_operator, lambda2,
                                    method='dSPM')
    stc_evoked.save(stc_fname)

    brain = Brain(subject, 'split', 'partially_inflated', size=(800, 400))
    for hemi in ['lh', 'rh']:
        stc = read_stc(stc_fname+'-%s.stc' % hemi)
        data = stc['data']
        times = np.arange(data.shape[1]) * stc['tstep'] + stc['tmin']
        brain.add_data(data, colormap='RdBu', vertices=stc['vertices'],
                       smoothing_steps=10, time=times, time_label=time_label,
                       initial_time=-0.1,hemi=hemi)

    abs_max = (np.abs(data)).max()
    brain.scale_data_colormap(fmin=0, fmid=abs_max/3, fmax=abs_max, center=0, transparent=True)

    brain.save_movie(meg_dir+subject+'_stc_evoked.mov', tmin=-0.1,tmax=0.5,time_dilation=10)
    brain.close()
