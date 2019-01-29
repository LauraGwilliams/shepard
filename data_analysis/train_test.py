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


# train on pure, test on partial

# get the data
purepar_fname = '/Users/meglab/Desktop/shep_fifs/A0305/A0305_purepar-epo.fif'
purepar_epochs = mne.read_epochs(purepar_fname)
X_purepar = purepar_epochs._data[:, 0:157, :] # just meg channels
y_purepar = purepar_epochs.metadata['freq'].values

partial_fname = '/Users/meglab/Desktop/shep_fifs/A0305/A0305_par-epo.fif'
partial_epochs = mne.read_epochs(partial_fname)
X_partial = partial_epochs._data[:, 0:157, :] # just meg channels
y_partial = partial_epochs.metadata['freq'].values

pure_fname = '/Users/meglab/Desktop/shep_fifs/A0305/A0305_pure-epo.fif'
pure_epochs = mne.read_epochs(pure_fname)
X_pure = pure_epochs._data[:, 0:157, :] # just meg channels
y_pure = pure_epochs.metadata['freq'].values

shep_fname = '/Users/meglab/Desktop/shep_fifs/A0305/A0305_shep-epo.fif'
shep_epochs = mne.read_epochs(shep_fname)
X_shep = shep_epochs._data[:, 0:157, :] # just meg channels
y_shep = shep_epochs.metadata['freq'].values

# train on one subset
n_times = X_partial.shape[-1]
scores = list()
for tt in range(n_times):
    clf = make_pipeline(StandardScaler(), Ridge())
    clf.fit(X_purepar[..., tt], y_purepar)
    y_pred_shep = clf.predict(X_shep[..., tt])
    scores.append(scorer_spearman(y_shep, y_pred_shep))
    print(tt)


fig, ax = plt.subplots()
ax.plot(partial_epochs.times, np.array(scores), label='score')
ax.axhline(.0, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('%s'%(score))
ax.legend()
# ax.set_ylim(bottom=-0.035, top=0.16)
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')
plt.show()
