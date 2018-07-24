# leg5@nyu.edu
# run decoding analysis on shepard data

# packages
import numpy as np
import pandas as pd
from mne import read_epochs
from jr import scorer_spearman
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from mne.decoding import GeneralizingEstimator, SlidingEstimator, get_coef, LinearModel, cross_val_multiscore
from sklean.metrics import mean_squared_error

# params
subject = ''
regressor = ''

# paths
epoch_fname = ''
info_fname = ''

# load data
epochs = mne.read_epochs(epoch_path)
X = epochs._data

# load trial info
trial_info = pd.read_csv(info_fname)
y = trial_info[regressor].values

# sanity
assert(len(X) == len(y))

# set up decoder
clf = make_pipeline(StandardScaler(),
                    Ridge())  # use logisitc for categorical and Ridge for continuous

scorer = 'mean_squared_error'

gen = GeneralizingEstimator(n_jobs=n_jobs,
                            scoring=scorer,
                            base_estimator=clf)

scores = cross_val_multiscore(gen, X, y,
                              cv=StratifiedKFold(5),
                              n_jobs=n_jobs)