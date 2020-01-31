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

#funcs
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

def grab_hemi_sensors(epochs,exclude_center=False):

    # grab info and channels from epochs
    info = epochs.info
    chs = info['chs'] # this is a dictionary of channels and their info

    if exclude_center:
        thresh = 0.03
    else:
        thresh = 0

    rh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] > thresh]]
    lh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] < -thresh]]

    rh_picks = mne.pick_types(epochs.info,selection=rh_ch_names)
    lh_picks = mne.pick_types(epochs.info,selection=lh_ch_names)

    return rh_picks, lh_picks
#__________________________________________________________
# load all data
subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']
meg_dir = '/Users/ea84/Dropbox/shepard_decoding/'

for subject in subjects:
    if not os.path.exists(meg_dir+'_GRP_SCORES/n=%i/indiv/ypreds/%s/'%(len(subjects),subject)):
        os.makedirs(meg_dir+'_GRP_SCORES/n=%i/indiv/ypreds/%s/'%(len(subjects),subject))

# params: epochs to use, regressor, how to decode, subsetting
column = ['condition']
train_on = [['pure']]
test_on = ['partial']
regressor = 'freq' #column name
score = 'Spearman R'
sensors = 'lh'

grp_scores = []
grp_ypreds = []
for subject in subjects:

    allepochs = meg_dir + '%s/%s_shepard-epo.fif'%(subject,subject)
    epochs = mne.read_epochs(allepochs)

    if subject[0] == 'A':
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

    rh_picks, lh_picks = grab_hemi_sensors(train_epochs,exclude_center=True)
    if sensors == 'rh':
        X_train = train_epochs._data[:, rh_picks, :]
    elif sensors == 'lh':
        X_train = train_epochs._data[:, lh_picks, :]
    else:
        # pull out sensor data from meg channels only
        X_train = train_epochs._data[:, 0:ch, :]

    y_train = train_info[regressor].values.astype(float)

    test_epochs = epochs[epochs.metadata[column[0]].isin(test_on)]
    test_info = test_epochs.metadata

    rh_picks, lh_picks = grab_hemi_sensors(train_epochs,exclude_center=True)
    if sensors == 'rh':
        X_test = test_epochs._data[:, rh_picks, :]
    elif sensors == 'lh':
        X_test = test_epochs._data[:, lh_picks, :]
    else:
        # pull out sensor data from meg channels only
        X_test = test_epochs._data[:, 0:ch, :]

    y_test = test_info[regressor].values.astype(float)

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
    # grp_ypreds.append(y_preds)
    ypreds_arr = np.array(y_preds)
    np.save(meg_dir+'_GRP_SCORES/n=%i/indiv/ypreds/%s/%s_%s_train%s_ypreds_%s.npy'%(len(subjects),subject,subject,regressor,''.join(train_on[0]),sensors), ypreds_arr)
    ypreds_arr = np.transpose(ypreds_arr, [1,0])
    ypreds_arr = np.mean(ypreds_arr[:,50:70],axis=1)
    kwargs = {"ypreds_%s"%(''.join(train_on[0])) : ypreds_arr}
    test_info_preds = test_info.assign(**kwargs)
    # test_info_preds = add_tone_properties(test_epochs,test_info_preds)
    test_info_preds.to_csv(meg_dir+'_GRP_SCORES/n=%i/indiv/ypreds/%s/%s_%s_train%s_ypreds_%s.csv'%(len(subjects),subject,subject,regressor,''.join(train_on[0]),sensors))
    grp_scores.append(scores)

scores_arr = np.array(grp_scores)
np.save(meg_dir+'_GRP_SCORES/n=%i/group/group_%s_train%s_test%s_%s.npy'%(len(subjects),regressor,''.join(train_on[0]),test_on[0],sensors), scores_arr)

grp_sem = np.std( np.array(grp_scores), axis=0 ) / np.sqrt(len(grp_scores))
grp_avg = np.mean( np.array(grp_scores), axis=0 )


#____________________________YPREDS______________________________
#
# ypreds_arr = np.array(grp_ypreds)
# np.save(meg_dir+'_GRP_SCORES/n=%i/group/group_%s_train%s_ypreds.npy'%(len(subjects),regressor,''.join(train_on[0])), ypreds_arr)
# grp_ypreds = np.mean(ypreds_arr, axis=0) # mean across subjects
# grp_ypreds = np.transpose(grp_ypreds, [1,0]) # transpose so first dimension is individual trials
# freq_preds_window = np.mean(grp_ypreds[:,50:70],axis=1) # grab 50-150ms and avg over window
# shep_preds = test_info.assign(ypreds=grp_ypreds_window) # add prediction column to metadata of test epochs
#
# indiv_ypreds = []
# for subject in subjects:

# now can compare predictions based on conditions/frequencies!!!!!!!!!
#
# # next step: plot predictions over time for epochs going down vs up?
#
#
fig, ax = plt.subplots()
ax.plot(epochs.times, grp_avg, label='score')
ax.fill_between(epochs.times, grp_avg-grp_sem, grp_avg+grp_sem,
                    alpha=0.2, linewidth=0, color='r')
ax.axhline(.0, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('%s'%(score))
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')
plt.savefig(meg_dir + '_GRP_PLOTS/n=%i/group_%s_train%s_test%s_%s.png'%(len(subjects),regressor,''.join(train_on[0]),test_on[0],sensors))
# plt.show()
