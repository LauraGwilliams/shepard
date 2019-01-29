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
subj = 'A0305'
meg_dir = '/Users/meglab/Desktop/shep_fifs/%s/'%(subj)

allepochs = meg_dir + '%s_shepard-epo.fif'%(subj)
epochs = mne.read_epochs(allepochs)

# params: epochs to use, regressor, how to decode, subsetting
column = 'condition'
train_on = ['pure']
test_on = ['partial']
regressor = 'freq' #column name
score = 'Spearman R'

# get the data
train_epochs = epochs[epochs.metadata[column].isin(train_on)]
train_info = train_epochs.metadata

X_train = train_epochs._data[:, 0:157, :] # just meg channels
y_train = train_info[regressor].values

test_epochs = epochs[epochs.metadata[column].isin(test_on)]
test_info = test_epochs.metadata

X_test = test_epochs._data[:, 0:157,]
y_test = test_info[regressor].values


# train on one subset
n_times = X_train.shape[-1]
scores = list()
for tt in range(n_times):
    clf = make_pipeline(StandardScaler(), Ridge())
    clf.fit(X_train[..., tt], y_train)
    y_pred = clf.predict(X_test[..., tt])
    scores.append(scorer_spearman(y_test, y_pred))
    print(tt)

fig, ax = plt.subplots()
ax.plot(epochs.times, np.array(scores), label='score')
ax.axhline(.0, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('%s'%(score))
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')
plt.show()
