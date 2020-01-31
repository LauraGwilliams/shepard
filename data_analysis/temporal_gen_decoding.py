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
from jr import scorer_spearman # will have to download git repos & import some stuff
from sklearn.metrics import make_scorer, get_scorer

# funcs
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

# # params
subjects = ['A0216','A0314','A0306','A0270','A0280','A0305','A0307',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

# epochs subset to train on
column = ['condition']
subset = [['partial']]
sensor_list = ['all','rh','lh']

# regressor to decode, spatial vs. temporal vs. combined
regressor = 'freq' #column name
decode_using = 'spatial' # spatial (trials x sensors x time)
                        # temporal (trials x time x sensors),
                        # combined (trials x sensors*time)
meg_dir = '/Users/ea84/Dropbox/shepard_decoding/'

#-------------------------------------------------------------------------
# for cond in subset[0]:
for sensors in sensor_list:
    grp_scores = []

    for subject in subjects:
        print (subject)
        # params
        allepochs = meg_dir + '%s/%s_shepard-epo.fif'%(subject,subject)
        epochs = mne.read_epochs(allepochs)

        # subset current_epochs based on parameters
        if len(column) > 1:
            current_epochs = epochs[epochs.metadata[column[0]].isin(subset[0]) &
                                epochs.metadata[column[1]].isin(subset[1])]

        else:
            current_epochs = epochs[epochs.metadata[column[0]].isin(subset[0])]

        trial_info = current_epochs.metadata

        if subject[0] == 'A':
            ch = 208
        else:
            ch = 157

        rh_picks, lh_picks = grab_hemi_sensors(current_epochs,exclude_center=False)
        if sensors == 'rh':
            X = current_epochs._data[:, rh_picks, :]
        elif sensors == 'lh':
            X = current_epochs._data[:, lh_picks, :]
        else:
            # pull out sensor data from meg channels only
            X = current_epochs._data[:, 0:ch, :]

        # pull out regressor of interest
        y = trial_info[regressor].values

        #------------------------------------------------------------------------
        # change X if applicable
        if decode_using == 'temporal':
            # switch channels and times to decode using time course
            X = np.transpose(X, [0, 2, 1])
        if decode_using == 'combined':
            # collapse over spatial and temporal dimensions
            [n_trials, n_sns, n_times] = X.shape
            X = np.reshape(X, [n_trials, n_sns*n_times])

        #------------------------------------------------------------------------
        # scaling funcs

        def my_scaler(x):
            '''
            Scale btw 0-1.
            '''
            x = np.array(x).astype(float)
            return (x - (np.min(x)) / (np.max(x) - np.min(x)))

        def binary_scaler(x):
            '''
            Pure = 1, partial = 0.
            '''
            x = np.array(x)
            x[x==subset[0][0]]=1.
            x[x==subset[0][1]]=0.
            x = np.array(x).astype(float)
            return x

        #------------------------------------------------------------------------

        # set up decoder, use logistic for categorical and Ridge for continuous
        if regressor == 'freq':
            y = my_scaler(y) # scale frequencies to between 0 and 1
            clf = make_pipeline(StandardScaler(), Ridge())
            scorer = make_scorer(get_scorer(scorer_spearman))
            score = 'Spearman R'
            cv = KFold(5)
        if regressor == 'condition':
            y = binary_scaler(y) # set values to 0.0 and 1.0
            clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
            scorer = 'roc_auc'
            score = 'AUC'
            cv = StratifiedKFold(5)

        n_jobs = -1

        print ("Decoding...")
        gen = GeneralizingEstimator(n_jobs=n_jobs,
                                scoring=scorer,
                                base_estimator=clf)
        scores = cross_val_multiscore(gen, X, y,
                                      cv=cv)

        # mean scores across cross-validation splits
        scores = np.mean(scores, axis=0)
        grp_scores.append(scores)

    scores_arr = np.array(grp_scores)

    np.save(meg_dir+'_GRP_SCORES/n=%i/temporal/group_%s_%s_%s.npy'%(len(subjects),regressor,
                                                    ''.join(subset[0]),sensors),
                                                    scores_arr)
    scores_arr = scores_arr.mean(0)

    plt.matshow(scores_arr, cmap=plt.cm.RdBu_r, origin='lower')
    # plt.show()
    plt.savefig(meg_dir + '_GRP_PLOTS/n=%s/temporal/group_%s_%s_%s.png'%(len(subjects),regressor,''.join(subset[0]),sensors))
