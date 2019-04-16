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
subjects = ['A0216','A0280','A0305','A0306','A0314','A0270','P010',
            'P011','A0343','A0344','A0345','A0353','A0323','A0307','A0354']

# params: epochs to use, regressor, how to decode, subsetting
column = ['condition']
train_on = [['pure']]
test_on = ['shepard']
regressor = 'freq' #column name
score = 'Spearman R'

grp_scores = []
for subject in subjects:
    meg_dir = '/Users/ea84/Dropbox/shepard_decoding/'

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

    X_train = train_epochs._data[:, 0:ch, :] # just meg channels
    y_train = train_info[regressor].values.astype(float)

    test_epochs = epochs[epochs.metadata[column[0]].isin(test_on)]
    test_info = test_epochs.metadata
    X_test = test_epochs._data[:, 0:ch,]
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

        grp_scores.append(scores)

scores_arr = np.array(grp_scores)
np.save(meg_dir+'_GRP_SCORES/group_%s_train%s_test%s.png'%(regressor,''.join(train_on[0]),test_on[0]), scores_arr)

grp_sem = np.std( np.array(grp_scores), axis=0 ) / np.sqrt(len(grp_scores))
grp_avg = np.mean( np.array(grp_scores), axis=0 )

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
plt.savefig(meg_dir + '_GRP_PLOTS/group_%s_train%s_test%s.png'%(regressor,''.join(train_on[0]),test_on[0]))
# plt.show()
