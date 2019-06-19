import numpy as np
import pandas as pd
import os.path as op
import mne
from scipy.stats import ttest_ind, spearmanr


# params
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'
info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'

info = pd.read_csv('%s/subj_msi_sync.csv'%(info_dir))

subjects = list(info['subject'])
MSI = list(info['MSI'])
sync = list(info['sync'])

regressor = 'freq'
subsets = ['pure','partial','purepartial']
sensor_list = ['all']

rows_list = []
for i in range(len(subjects)):
    row = dict()
    row['MSI'] = MSI[i]
    row['sync'] = sync[i]
    for subset in subsets:
        for sensors in sensor_list:
        # load scores for subject
            scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))
            sub_scores = scores[i] # grab timepoints 0 - 300 [50:80]
            # row['%s_%s_%s'%(regressor,subset,sensors)] = np.mean(sub_scores,axis=0)
            row['%s_%s_%s'%(regressor,subset,sensors)] = sub_scores
            # row['tval_purevpar'] = 0.0
            # row['pval_purevpar'] = 0.0
            # row['purevpar'] = ''
    rows_list.append(row)

df = pd.DataFrame(rows_list, index = subjects)

tval_list = []
for i in range(len(df)):
    tval_list.append(ttest_ind(df['freq_pure_all'][i],df['freq_partial_all'][i]))

tvals = []
pvals = []
purevpar = []
for i in range(len(df)):
    tvals.append(tval_list[i][0])
    pvals.append(tval_list[i][1])
    if tval_list[i][1] < 0.1:
        if tval_list[i][0] > 0:
            purevpar.append('partial')
        else:
            purevpar.append('pure')
    else:
        purevpar.append('same')

df_ttest = df.assign(tvals=tvals,pvals=pvals,preference=purevpar)

# get highs and lows
highs = pd.DataFrame(df[df['sync'] == 1])
high_subs = list(df.query('sync == 1').index)

lows = pd.DataFrame(df[df['sync'] == 0])
low_subs = list(df.query('sync == 0').index)
