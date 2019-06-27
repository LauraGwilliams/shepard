import numpy as np
import pandas as pd
import os.path as op
import mne
from scipy.stats import ttest_ind, spearmanr
from jr import scorer_spearman


# params
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/indiv/ypreds/'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

subject = 'P022'
column = ['condition']
train_on = [['partial']]
test_on = ['shepard']
regressor = 'freq' #column name

condition = 'circular'

y_preds = np.load(scores_dir+'%s/%s_%s_train%s_ypreds.npy'%(subject,subject,regressor,''.join(train_on[0])))
csv = pd.read_csv(scores_dir+'%s/%s_%s_train%s_ypreds.csv'%(subject,subject,regressor,''.join(train_on[0])))

n_times = len(y_preds)

if condition != 'all':

    csv = csv[csv['circscale']==condition]
    mask = csv.index[csv['circscale']==condition].tolist()
    y_preds = np.transpose(y_preds, [1,0])
    y_preds = y_preds[mask]
    y_preds = np.transpose(y_preds, [1,0])

y_test = np.array(csv['freq'])

scores = list()
for tt in range(n_times):
    scores.append(scorer_spearman(y_test, y_preds[tt]))
    print(tt)

mean_score = np.mean(scores)


#csv.index[csv['trial_type'].str.contains('Scale')].tolist()
