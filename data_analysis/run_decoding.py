# leg5@nyu.edu & ea84@nyu.edu
# run decoding analysis on shepard data

# packages
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

subj = 'A0305'
meg_dir = '/Users/meglab/Desktop/shep_fifs/' +subj+ '/'

# load all data
allepochs = meg_dir + subj + '_shepard-epo.fif'
epochs = mne.read_epochs(allepochs)

# params: epochs to use, regressor, how to decode, subsetting
column = 'condition'
subset = ['pure','partial']

regressor = 'freq' #column name
decode_using = 'spatial'
subset_trials = False
# spatial (trials x sensors x time), temporal (trials x time x sensors),
# combined (trials x sensors*time)

# subset current_epochs based on parameters
current_epochs = epochs[epochs.metadata[column].isin(subset)]
trial_info = current_epochs.metadata

X = current_epochs._data[:, 0:208, :] # just meg channels

# pull out regressor of interest
y = trial_info[regressor].values

# sanity
assert(len(X) == len(y))

#-------------------------------------------------------------------------------

# change X if applicable
if decode_using == 'temporal':
    # switch channels and times to decode using time course
    X = np.transpose(X, [0, 2, 1])
if decode_using == 'combined':
    # collapse over spatial and temporal dimensions
    [n_trials, n_sns, n_times] = X.shape
    X = np.reshape(X, [n_trials, n_sns*n_times])

# # add another column specifying whether the freq of current trial
# # is higher or lower than the preceeding trial.
# current_trials = trial_info['freq'].values
# preceding_trial = np.roll(current_trials, 1)
# low_preceeding = (current_trials > preceding_trial)*1
# trial_info['low_preceeding'] = low_preceeding
# # first entry not valid because there is no preceding trial by definition
# trial_info['low_preceeding'][0] = np.nan
# trial_info['updown_switch'] = np.array([''.join([str(ii), str(jj)]) for ii, jj in trial_info[['low_preceeding', 'updown']].values])
# trial_info['purepar_updown'] = np.array([''.join([str(ii), str(jj)]) for ii, jj in trial_info[['condition', 'updown']].values])
# trial_info['freq_key'] = np.array([''.join([str(ii), str(jj)]) for ii, jj in trial_info[['key', 'freq']].values])


# to asses decoding performance on a reduced number of trials
if subset_trials == True:
    y = y[0:2160]
    X = X[0:2160, ...]

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
    x[x=='pure']=1.
    x[x=='partial']=0.
    x = np.array(x).astype(float)
    return x

#----------------------------------------------------------------------------

# NOTE: it is important to make sure that the y array is in float
# format not integer, otherwise it turns it into a binary problem.
# Also, it doesn't make sense to use statified KFold for regression problem.
# because there are no classes to speak of. KFold instead. My bad.

# set up decoder, use logistic for categorical and Ridge for continuous
if regressor == 'freq':
    y = my_scaler(y)
    clf = make_pipeline(StandardScaler(), Ridge())
    scorer = make_scorer(get_scorer(scorer_spearman))
    score = 'Spearman R'
    cv = KFold(5)
if regressor == 'condition':
    y = binary_scaler(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    scorer = 'roc_auc'
    score = 'AUC'
    cv = StratifiedKFold(5)

n_jobs = -1

# set up estimator, get scores
if decode_using == 'spatial':
    gen = GeneralizingEstimator(n_jobs=n_jobs,
                            scoring=scorer,
                            base_estimator=clf)
    scores = cross_val_multiscore(gen, X, y,
                                  cv=cv)
elif decode_using == 'temporal':
    gen = SlidingEstimator(n_jobs=n_jobs,
                        scoring=scorer,
                        base_estimator=clf)
    scores = cross_val_multiscore(gen, X, y,
                                cv=cv)
else:
    # scoring defaults to neg mean squared so set to scorer
    # shuffle must be true when binary values, otherwise fold will only have
        # one value
    scores = cross_val_score(clf, X, y,
                            scoring=scorer, #defaults to neg mean squared
                            cv=KFold(5, shuffle=True))


# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# ---------------------------------------------------------------------------
# PLOTTING

if decode_using == 'spatial':
    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(epochs.times, np.diag(scores), label='score')
    if score == 'AUC':
        ax.axhline(.5, color='k', linestyle='--', label='chance')
    else:
        ax.axhline(.0, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('%s'%(score))
    ax.legend()
    # ax.set_ylim(bottom=-0.035, top=0.16)
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    plt.show()
elif decode_using == 'temporal':
    scores_evoked = mne.read_evokeds(meg_dir+subj+'_shepard-evoked-ave.fif')[0]
    scores_evoked._data[0:157, 0] = scores
    if regressor == 'freq':
        scores_evoked.plot_topomap(times=scores_evoked.times[0])
    if regressor == 'condition':
        # center around 0 for plotting
        scores_evoked._data[0:157, 0] = scores_evoked._data[0:157, 0] - scores_evoked._data[0:157, 0].mean()
        scores_evoked.plot_topomap(times=scores_evoked.times[0])
else:
    print (scores)

# plot matrix
plt.matshow(plt.matshow(scores, cmap=plt.cm.RdBu_r, origin='lower'))
plt.show()
