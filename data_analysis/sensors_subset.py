import mne
import os.path as op
import numpy as np
from mne.io import read_raw_fif

def grab_hemi_sensors(epochs,exclude_center=False):

    # grab info and channels from epochs
    info = epochs.info
    chs = info['chs'] # this is a dictionary of channels and their info

    # divide RH and LH
    if exclude_center:
        # exclude center sensors to avoid correlation between lh and rh
        rh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] > 0.03]]
        lh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] < -0.03]]
    else:
        # split down the middle
        rh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] > 0]]
        lh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] < 0]]

    rh_picks = mne.pick_types(epochs.info,selection=rh_ch_names)
    lh_picks = mne.pick_types(epochs.info,selection=lh_ch_names)

    return rh_picks, lh_picks

#-----------------------

subject = 'A0344'
meg_dir = '/Users/ea84/Dropbox/shepard_decoding/%s/'%(subject)

allepochs = meg_dir + '%s_shepard-epo.fif'%(subject)
epochs = mne.read_epochs(allepochs)

rh_picks, lh_picks = grab_hemi_sensors(epochs)

# plot for sanity
evoked = epochs.average()
evoked.plot(picks=rh_picks,spatial_colors=True)
evoked.plot(picks=lh_picks,spatial_colors=True)
