import mne
import os.path as op
import numpy as np
from mne.io import read_raw_fif
from mne import (find_events, Epochs)
import pandas as pd
import glob

def add_tone_properties(epochs, trial_info):

    # musical note position dictionary
    note_pos_dict = {'A220': 1,'A247': 2,'A277': 3,'A294': 4,'A330': 5,'A370': 6,
                     'A415': 7,'A440': 8,'C262': 1,'C294': 2,'C330': 3,'C349': 4,
                     'C392': 5,'C440': 6,'C494': 7,'C524': 8,'Eb312': 1,'Eb349': 2,
                     'Eb392': 3,'Eb415': 4,'Eb466': 5,'Eb523': 6,'Eb524': 6, 'Eb587': 7,'Eb624': 8}

    # add note position to the df
    trial_info['freq_key'] = ['%s%s' % (k, f) for k, f in trial_info[['key', 'freq']].values]
    trial_info['note_position'] = np.array([note_pos_dict[k] for k in trial_info['freq_key']])
    trial_info['time_ms'] = epochs.events[:, 0]
    trial_info['ISI'] = trial_info['time_ms'] - np.roll(trial_info['time_ms'], 1)

    # add tone position
    position = 1
    trial_positions = list()
    for isi in trial_info['ISI'].values:
        # if the time from the previous trial is less than 750, then it is
        # not the first note in the sequence
        if isi < 750:
            trial_positions.append(position)
            position += 1
        else:
            # if the ISI is bigger than 750, it is the first note in the
            # sequence
            trial_positions.append(1)
            position = 2
    trial_info['trial_position'] = np.array(trial_positions)

    # fix the note position for the ambiguous tones in "scale"
    # specifies the perceived tone
    trial_info['note_position'] = (trial_info['note_position'] +
                                   np.logical_and(np.logical_and(trial_info['updown'] == 'down', trial_info['circscale'] == 'scale'),
                                                  trial_info['note_position'] == trial_info['trial_position'])*7)

    trial_info['note_position'] = (trial_info['note_position'] +
                                   np.logical_and(np.logical_and(trial_info['updown'] == 'up', trial_info['circscale'] == 'scale'),
                                                  trial_info['note_position'] != trial_info['trial_position'])*7)

    return trial_info

# params
subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']
for subject in subjects:
    print (subject)
    meg_dir = '/Users/ea84/Dropbox/shepard_decoding/%s/'%(subject)
    filt_l = 1  # same as aquisition
    filt_h = 60
    tmin = -0.2
    tmax = 0.6

    # file names
    raw_fname = meg_dir + '%s_shepard-raw.fif'%(subject)
    epochs_fname = meg_dir + '%s_shepard-epo.fif'%(subject)
    #
    # server_dir = '/Users/ea84/Dropbox/shepard_preproc/'
    #
    # # read, filter
    # if op.isfile(raw_fname):
    #     raw = read_raw_fif(raw_fname, preload=True)
    # else:
    #     con_fname = server_dir + '%s/%s_shep1_NR.con'%(subject,subject)
    #     raw1 = mne.io.read_raw_kit(con_fname, preload=True, slope='+')
    #     con_fname = server_dir + '%s/%s_shep2_NR.con'%(subject,subject)
    #     raw2 = mne.io.read_raw_kit(con_fname, preload=True, slope='+')
    #     if len(glob.glob(server_dir + '%s/*.con'%(subject))) > 2:
    #         con_fname = server_dir + '%s/%s_shep3_NR.con'%(subject,subject)
    #         raw3 = mne.io.read_raw_kit(con_fname, preload=True, slope='+')
    #         raw = mne.concatenate_raws([raw1, raw2, raw3])
    #     else:
    #         raw = mne.concatenate_raws([raw1, raw2])
    #
    #     raw.save(raw_fname)
    #
    # raw = raw.filter(filt_l, filt_h)
    #
    # # check events, create epochs
    # events = find_events(raw)  # the output of this is a 3 x n_trial np array
    # epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, decim = 5, baseline=None)
    epochs = mne.read_epochs(epochs_fname)

    # set metadata, save epochs
    trial_info = pd.read_csv(meg_dir + '%s_shepard_trialinfo.csv'%(subject))
    trial_info = add_tone_properties(epochs,trial_info)

    epochs.metadata = trial_info
    epochs.save(epochs_fname)

    trial_info.to_csv(meg_dir + '%s_shepard_trialinfo.csv'%(subject))
