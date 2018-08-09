# leg5@nyu.edu
# run decoding analysis on shepard data

# packages
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne import read_epochs
#from jr import scorer_spearman
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, get_coef,
                        LinearModel, cross_val_multiscore)
import mne.decoding
from jr import scorer_spearman
from sklearn.metrics import make_scorer, get_scorer

meg_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/R1201/'

# epochs collapsed across pure and partials
epoch_fname = meg_dir + 'R1201_purepar-epo.fif'
info_fname = meg_dir + 'R1201_purepar_trialinfo.csv'

# keys collapsed across pure and partials
epoch_A_fname = meg_dir + 'R1201_A_purepar-epo.fif'
epoch_C_fname = meg_dir + 'R1201_C_purepar-epo.fif'
epoch_Eb_fname = meg_dir + 'R1201_Eb_purepar-epo.fif'
info_A_fname = meg_dir + 'R1201_A_purepar_trialinfo.csv'
info_C_fname = meg_dir + 'R1201_C_purepar_trialinfo.csv'
info_Eb_fname = meg_dir + 'R1201_Eb_purepar_trialinfo.csv'

# pure epochs
epoch_pure_fname = meg_dir + 'R1201_pure-epo.fif'
info_pure_fname = meg_dir + 'R1201_pure_trialinfo.csv'

# partials epochs
epoch_partial_fname = meg_dir + 'R1201_par-epo.fif'
info_partial_fname = meg_dir + 'R1201_partials_trialinfo.csv'

# params: choose subject, regressor, epochs/info, how to decode
subject = 'R1201'
regressor = 'freq' #condition or frequency
current_epochs = epoch_fname
current_info = info_fname
decode_using = 'combined'
# spatial (trials x sensors x time), temporal (trials x time x sensors),
# combined (trials x sensors*time)

# load data
epochs = mne.read_epochs(current_epochs)
X = epochs._data[:, 0:157, :] # just meg channels

# change X if applicable
if decode_using == 'temporal':
    # switch channels and times to decode using time course
    X = np.transpose(X, [0, 2, 1])
if decode_using == 'combined':
    # collapse over spatial and temporal dimensions
    [n_trials, n_sns, n_times] = X.shape
    X = np.reshape(X, [n_trials, n_sns*n_times])

# load trial info
trial_info = pd.read_csv(current_info)
y = trial_info[regressor].values

# sanity
assert(len(X) == len(y))

def my_scaler(x):
    '''
    Scale btw 0-1.
    '''
    x = np.array(map(float, x))
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def binary_scaler(x):
    '''
    Pure = 1, partial = 0.
    '''
    x = np.array(x)
    x[x=='pure']=1.
    x[x=='partial']=0.
    x = np.array(map(float, x))
    return x

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
if regressor == 'condition':
    y = binary_scaler(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    scorer = 'roc_auc'
    score = 'AUC'

n_jobs = -1

# set up estimator, get scores
if decode_using == 'spatial':
    gen = GeneralizingEstimator(n_jobs=n_jobs,
                            scoring=scorer,
                            base_estimator=clf)
    scores = cross_val_multiscore(gen, X, y,
                                  cv=KFold(5, shuffle=True))
elif decode_using == 'temporal':
    gen = SlidingEstimator(n_jobs=n_jobs,
                        scoring=scorer,
                        base_estimator=clf)
    scores = cross_val_multiscore(gen, X, y,
                                cv=KFold(5, shuffle=True))
else:
    scores = cross_val_score(clf, X, y,
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
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    plt.show()
elif decode_using == 'temporal':
    scores_evoked = mne.load_evokeds(meg_dir+'R1201_shepard-evoked-ave.fif')
    scores_evoked._data[0:157, 0] = scores
    if regressor == 'freq':
        scores_evoked.plot_topomap(times=scores_evoked.times[0])
    if regressor == 'condition':
        # center around 0 for plotting
        scores_evoked._data[0:157, 0] = scores_evoked._data[0:157, 0] -
                                        scores_evoked._data[0:157, 0].mean()
        scores_evoked.plot_topomap(times=scores_evoked.times[0])
else:
