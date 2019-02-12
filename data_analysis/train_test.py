import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne import read_epochs, read_evokeds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, cross_val_score, cross_val_predict,
                                    StratifiedKFold)
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, get_coef,
                        LinearModel, cross_val_multiscore)
import mne.decoding
from jr import scorer_spearman
from sklearn.metrics import make_scorer, get_scorer


# load all data
subj = 'P010'
meg_dir = '/Users/ea84/Dropbox/shepard_decoding/%s/'%(subj)

allepochs = meg_dir + '%s_shepard-epo.fif'%(subj)
epochs = mne.read_epochs(allepochs)

# params: epochs to use, regressor, how to decode, subsetting
column = ['condition']
train_on = [['partial']]
test_on = ['pure']
regressor = 'freq' #column name
score = 'Spearman R'

if subj[0] == 'A':
    ch = 208
else:
    ch = 157

# get the data
if len(column) > 1:
    train_epochs = epochs[epochs.metadata[column[0]].isin(train_on[0]) &
                        epochs.metadata[column[1]].isin(train_on[1])]
else:
    train_epochs = epochs[epochs.metadata[column[0]].isin(train_on[0])]
train_info = train_epochs.metadata

X_train = train_epochs._data[:, 0:ch, :] # just meg channels
y_train = train_info[regressor].values.astype(float)

test_epochs = epochs[epochs.metadata[column[0]].isin(test_on)]
test_info = test_epochs.metadata
X_test = test_epochs._data[:, 0:ch,]
y_test = test_info[regressor].values.astype(float)

def add_tone_properties(epochs, trial_info):

    # musical note position dictionary
    note_pos_dict = {'A220': 1,'A247': 2,'A277': 3,'A294': 4,'A330': 5,'A370': 6,
                     'A415': 7,'A440': 8,'C262': 1,'C294': 2,'C330': 3,'C349': 4,
                     'C392': 5,'C440': 6,'C494': 7,'C523': 8,'Eb312': 1,'Eb349': 2,
                     'Eb392': 3,'Eb415': 4,'Eb466': 5,'Eb523': 6,'Eb587': 7,'Eb624': 8}

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

# # top and bottom of scale condition
# test_epochs = epochs[epochs.metadata[column[0]].isin(test_on)]
# test_info = test_epochs.metadata
# test_info = add_tone_properties(test_epochs, test_info).reset_index()
# test_info = test_info.query("circscale == 'scale' and note_position in [1, 8]")
# test_epochs = test_epochs[test_info.index]

# # ambigous tone in circular condition

# test_info = add_tone_properties(test_epochs, test_info).reset_index()
# test_info = test_info.query("circscale == 'circular' and note_position in [1]")
# test_epochs = test_epochs[test_info.index]

# train on one subset
n_times = X_train.shape[-1]
scores = list()
y_preds = list()
for tt in range(n_times):
    clf = make_pipeline(StandardScaler(), Ridge())
    clf.fit(X_train[..., tt], y_train)
    y_pred = clf.predict(X_test[..., tt])
    y_preds.append(y_pred)
    scores.append(scorer_spearman(y_test, y_pred))
    print(tt)

# split preds into perceived high and low tones
# y_preds = np.array(y_preds)

# look at ambiguous tone at top and bottom of scale
# for key in test_info['key'].unique():
#     for note_pos in [1, 8]:
#         tone_idx = test_info['note_position'] == note_pos
#         key_idx = test_info['key'] == key
#         idx = np.logical_and(tone_idx, key_idx)
#         plt.plot(test_epochs.times, y_preds[:, idx].mean(1), label='%s %s' % (key, note_pos))
#     plt.legend()
#     plt.show()
#
# # look at ambiguous tone in up vs down circular position
# for key in test_info['key'].unique():
#     for dir in ['up','down']:
#         tone_idx = test_info['updown'] == dir
#         key_idx = test_info['key'] == key
#         idx = np.logical_and(tone_idx, key_idx)
#         plt.plot(test_epochs.times, y_preds[:, idx].mean(1), label='%s %s' % (key, dir))
#     plt.legend()
#     plt.show()

fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.0, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('%s'%(score))
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')
plt.show()
