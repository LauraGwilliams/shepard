import numpy as np
import pandas as pd
import os.path as op
import mne
from scipy.stats import ttest_ind, spearmanr

# funcs
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# params
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'
info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'

info = pd.read_csv('%s/MSI_breakdown.csv'%(info_dir))

columns = info.columns.tolist()

regressor = 'freq'
subsets = ['pure']
sensor_list = ['all']

rows_list = []
for i in range(len(subjects)):
    row = dict()
    for col in columns:
        row['%s'%(col)] = info[col][i]
    for subset in subsets:
        for sensors in sensor_list:
        # load scores for subject
            scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))
            for tt in range(161):
                row['data_tt%s'%(tt)] = scores[i][tt]
            row['max_score'] = np.mean(scores[i][50:70])
            # row['tval_purevpar'] = 0.0
            # row['pval_purevpar'] = 0.0
            # row['purevpar'] = ''
    rows_list.append(row)

df = pd.DataFrame(rows_list)

#____________________________________________________________
# plot scores with behavioral measures
X = df['max_score']
Y = df['General_Sophistication']

corr = X.corr(Y)

# solution
a, b = best_fit(X, Y)

plt.scatter(X, Y)
plt.xlabel('PLV')
plt.ylabel('Musical Sophistication')
plt.title('PLV vs. Musical Sophistication')
plt.xlim(0.0,X.max()+0.02)
plt.ylim(Y.min()-10,Y.max()+10)
plt.axvline(x=0.5, color='Gray', linestyle='--',label='0.5 PLV')
yfit = [a + b * xi for xi in X]
plt.plot(X, yfit, color='Black', label='best fit, r = %s'%(corr))
plt.legend(prop={'size': 8})
plt.show()


#____________________________________________________________
#

ttest_ind(df[df['High/Low']==1]['max_score'],df[df['High/Low']==0]['max_score'])

high_scores = []
low_scores = []

for tt in range(161):
    high_scores.append(np.mean(df[df['High/Low']==1]['data_tt%s'%(tt)]))
    low_scores.append(np.mean(df[df['High/Low']==0]['data_tt%s'%(tt)]))

scores = [high_scores,low_scores]

# plot scores high vs low High/Lowhronizers
for score, lab in zip(scores,labels):
     plt.plot(times,score,label=lab)
plt.title('decoding %s high vs low High/Low synchronizers'%(subsets[0]))
plt.legend()
plt.show()


#____________________________________________________________
# get highs and lows
highs = pd.DataFrame(df[df['High/Low'] == 1])
high_subs = list(df.query('High/Low == 1').index)

lows = pd.DataFrame(df[df['High/Low'] == 0])
low_subs = list(df.query('High/Low == 0').index)

tval_list = []
tvals = []
for tt in range(161):
    tval_list.append(ttest_ind(lows['data_tt%s'%(tt)],highs['data_tt%s'%(tt)]))
    tvals.append(tval_list[tt][0])

plt.plot(times,tvals,label='tval')
plt.legend()
plt.title('ttest high vs low High/Low synchronizers %s'%(subsets[0]))
plt.show()

#_________________________
# individual differences in pure vs partial

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
