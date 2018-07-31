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
#from meg_preprocessing import epochs, trial_info
# from sklearn.metrics import neg_mean_squared_error
from jr import scorer_spearman
from sklearn.metrics import make_scorer, get_scorer

meg_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/R1201/'

# params
subject = 'R1201'
regressor = 'freq'

# subset epochs, trial_info
# epochs_decoding = epochs[(epochs.metadata['condition'] == 'partial') |
#                         (epochs.metadata['condition'] == 'pure')]
# epochs_decoding.save(meg_dir + 'R1201_purepar-epo.fif')
# trial_info = trial_info[(trial_info["condition"] == "partial") |
#                         (trial_info["condition"] == "pure")]
# trial_info.to_csv(meg_dir + 'R1201_purepar_trialinfo.csv')


# epochs collapsed across pure and partials
epoch_fname = meg_dir + 'R1201_purepar-epo.fif'
info_fname = meg_dir + 'R1201_purepar_trialinfo.csv'

# pure epochs
epoch_pure_fname = meg_dir + 'R1201_pure-epo.fif'
info_pure_fname = meg_dir + 'R1201_pure_trialinfo.csv'

# partials epochs
epoch_partial_fname = meg_dir + 'R1201_par-epo.fif'
info_partial_fname = meg_dir + 'R1201_partials_trialinfo.csv'

# load data
#epochs = mne.read_epochs(epoch_fname)
#epochs = mne.read_epochs(epoch_pure_fname)
epochs = mne.read_epochs(epoch_partial_fname)
X = epochs._data[:, 0:156, :] # just meg channels

# load trial info
#trial_info = pd.read_csv(info_fname)
#trial_info = pd.read_csv(info_pure_fname)
trial_info = pd.read_csv(info_partial_fname)
y = trial_info[regressor].values

# sanity
assert(len(X) == len(y))

def my_scaler(x):
    '''
    Scale btw 0-1.
    '''
    x = np.array(map(float, x))
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# NOTE: it is important to make sure that the y array is in float
# format not integer, otherwise it turns it into a binary problemself.
# Also, it doesn't make sense to use statified KFold for regresion probmeself.
# because there are no classes to speak of. KFold instead. My bad.

# set up decoder
y = my_scaler(y)
clf = make_pipeline(StandardScaler(),
                    Ridge())  # use logistic for categorical and Ridge for continuous

# scorer = 'mean_squared_error'
scorer = make_scorer(get_scorer(scorer_spearman))
n_jobs = 2


gen = GeneralizingEstimator(n_jobs=n_jobs,
                            scoring=scorer,
                            base_estimator=clf)

scores = cross_val_multiscore(gen, X, y,
                              cv=5)
