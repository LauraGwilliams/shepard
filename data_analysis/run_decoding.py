# leg5@nyu.edu
# run decoding analysis on shepard data

# packages
import numpy as np
import pandas as pd
import mne
from mne import read_epochs
#from jr import scorer_spearman
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, get_coef,
                        LinearModel, cross_val_multiscore)
import mne.decoding
from meg_preprocessing import epochs, trial_info
from sklearn.metrics import mean_squared_error

meg_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/R1201/'

# params
subject = 'R1201'
regressor = 'freq'

# subset epochs, trial_info
epochs_decoding = epochs[(epochs.metadata['condition'] == 'partial') |
                        (epochs.metadata['condition'] == 'pure')]
epochs_decoding.save(meg_dir + 'R1201_purepar-epo.fif')
trial_info = trial_info[(trial_info["condition"] == "partial") |
                        (trial_info["condition"] == "pure")]
trial_info.to_csv(meg_dir + 'R1201_purepar_trialinfo.csv')


# paths
epoch_fname = meg_dir + 'R1201_purepar-epo.fif'
info_fname = meg_dir + 'R1201_purepar_trialinfo.csv'

# load data
epochs = mne.read_epochs(epoch_fname)
X = epochs._data

# load trial info
trial_info = pd.read_csv(info_fname)
y = trial_info[regressor].values

# sanity
assert(len(X) == len(y))

# set up decoder
clf = make_pipeline(StandardScaler(),
                    Ridge())  # use logistic for categorical and Ridge for continuous

scorer = 'mean_squared_error'
n_jobs = 2


gen = mne.decoding.GeneralizingEstimator(n_jobs=n_jobs,
                            scoring=scorer,
                            base_estimator=clf)

scores = cross_val_multiscore(gen, X, y,
                              cv=StratifiedKFold(5),
                              n_jobs=n_jobs)
