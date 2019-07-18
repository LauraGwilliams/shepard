import numpy as np
import pandas as pd
import os.path as op
import mne
from scipy.stats import ttest_ind, spearmanr


# funcs
def cluster_test(scores, n_perm=10000, threshold=1.5, n_jobs=1,
				 dimension='time'):
    '''perform 1-sample t-test over time or space to find clusters'''

    # dims
    # n_obs, n_datapoints = scores.shape

    # connectivity map for spatial map
    if dimension == 'space':
    	connectivity = mne.spatial_tris_connectivity(mne.grad_to_tris(4))
    elif dimension == 'time':
    	connectivity = None

    t_obs, clusters, cluster_pv, H0 = perm_1samp(scores,
                                                     n_jobs=n_jobs,
                                                     threshold=threshold,
                                                     n_permutations=n_perm,
                                                     connectivity=connectivity)

    return(t_obs, clusters, cluster_pv, H0)


def plot_clusters(t_obs, clusters, cluster_pv, label, p_thresh=0.05,
                  times=np.linspace(-200, 600, 161), col='red'):
    '''
    Plot clusters on tval timecourse. Assumes that each input is a list --
    one for each variable.
    '''

    n_times = len(t_obs)

    plt.plot(times, t_obs, label=label, color=col, lw=4)
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_pv[i_c] <= p_thresh:

            # get indices of cluster
            start_idx = int(str(c.start))
            stop_idx = int(str(c.stop))-1

            # define params for fill between for the significant highligting
            x = times[start_idx:stop_idx]
            y1 = t_obs[start_idx:stop_idx]
            plt.fill_between(x, y1, y2=0, color=col, alpha=0.3)

            # get ms dims of cluster
            c_start = times[start_idx]
            c_stop = times[stop_idx]

            # print the stats output for paper results
            print(label, c_start, c_stop, float(cluster_pv[i_c]),
                  t_obs[c.start-1:c.stop-1].mean(0))

    plt.legend(loc='upper left')
    plt.ylabel('t Value')

    plt.axvline(0, ls='--', c='k')
    plt.axhline(0, ls='-', c='k')
    plt.axhline(2, ls=':', c='k')

    return plt

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

#____________________________________________________________
#############################################################
################### RUN THIS CHUNK FIRST ####################
#############################################################

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'
info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'

info = pd.read_csv('%s/MSI_breakdown.csv'%(info_dir))
pitch_discrim = np.load('/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/pitchdiscrim/pitch_discrim_res.npy')

columns = info.columns.tolist()

regressor = 'freq'
subsets = ['pure']
sensor_list = ['all']

# create dataframe with musicianship scores, scores at each timepoint
rows_list = []

for i in range(len(subjects)):
	row = dict()
	for col in columns:
		row['%s'%(col)] = info[col][i]
	for subset in subsets:
		for sensors in sensor_list:
			scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))
			for tt in range(161):
				row['data_tt%s'%(tt)] = scores[i][tt]
			row['max_score'] = np.mean(scores[i][40:100])
	subject = subjects[i]
	row['same_diff'] = pitch_discrim.item().get(subject)[0]
	row['up_down'] = pitch_discrim.item().get(subject)[1]
	rows_list.append(row)

df = pd.DataFrame(rows_list)
#____________________________________________________________
# histogram
import matplotlib.mlab as mlab
from scipy.stats import norm

# grab PLV column from info dataframe
PLV = np.array(info['PLV'])

# create histogram
n, bins, patches = plt.hist(PLV,bins=10)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.title('PLV, n=28')
plt.show()

#____________________________________________________________
# correlations between MSI categories


categories = df[['Active_Engagement','Perceptual_Abilities','Musical_Training',
        'Emotions','Singing_Abilities','General_Sophistication','same_diff',
		'up_down','PLV']]

print categories.corr()

#____________________________________________________________
# plot average decoding accuracy 0-300ms with behavioral measures
# scatter plot measures w correlations

X = df['max_score']
Y = df['Perceptual_Abilities']

corr = X.corr(Y)
# r value of logistic regression

# solution
a, b = best_fit(X, Y)
yfit = [a + b * xi for xi in X]

plt.scatter(X, Y)
plt.xlabel(X.name)
plt.ylabel(Y.name)
plt.title('%s vs. %s'%(X.name,Y.name))
plt.xlim(0.0,X.max()+0.01)
plt.ylim(Y.min()-10,Y.max()+10)
plt.axvline(x=0.5, color='Gray', linestyle='--',label='0.5 PLV')
plt.plot(X, yfit, color='Black', label='best fit, r = %s'%(corr))
plt.legend(prop={'size': 8})
plt.show()

#____________________________________________________________
plots_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_PLOTS/n=28/high_vs_low_sync/'

# t-test average scores across entire time window
ttest_ind(df[df['High_Low']==1]['max_score'],df[df['High_Low']==0]['max_score'])

# t-test scores at each timepoint for highs vs lows
high_scores = []
low_scores = []

for tt in range(161):
    high_scores.append(np.mean(df[df['High_Low']==1]['data_tt%s'%(tt)]))
    low_scores.append(np.mean(df[df['High_Low']==0]['data_tt%s'%(tt)]))

scores = [high_scores,low_scores]

# plot ACCURACY SCORES high vs low synchronizers
for score, lab in zip(scores,labels):
     plt.plot(times,score,label=lab)
plt.axhline(y=0,color='Black',linestyle='--')
plt.title('decoding accuracy %s high vs low synchronizers'%(subsets[0]))
plt.legend()
plt.savefig(plots_dir+'highvlow_%s'%(''.join(subsets)))
plt.show()
plt.close()

#____________________________________________________________
# get highs and lows
highs = pd.DataFrame(df[df['High_Low'] == 1])
high_subs = list(df.query('High_Low == 1').index)

lows = pd.DataFrame(df[df['High_Low'] == 0])
low_subs = list(df.query('High_Low == 0').index)

tval_list = []
tvals = []
for tt in range(161):
    tval_list.append(ttest_ind(lows['data_tt%s'%(tt)],highs['data_tt%s'%(tt)]))
    tvals.append(tval_list[tt][0])

# plot TVALUES
plt.plot(times,tvals,label='tval')
plt.legend()
plt.axhline(y=0,color='Black',linestyle='--')
plt.title('ttest high vs low synchronizers %s'%(subsets[0]))
plt.savefig(plots_dir+'ttest_highvlow_%s'%(''.join(subsets)))
plt.show()

pval_list = []
pvals = []
for tt in range(161):
    pval_list.append(ttest_ind(lows['data_tt%s'%(tt)],highs['data_tt%s'%(tt)]))
    pvals.append(pval_list[tt][1])

tvals = np.array(tvals)

# cluster test
res = cluster_test(tvals, n_perm=10000, threshold=1.5, n_jobs=1,
                    dimension='time')

t_obs, clusters, cluster_pv, H0 = res

plot_clusters(t_obs, clusters, cluster_pv, label=regressor)
plt.title('Decoding %s over time'%(regressor))
plt.xlabel('Times')
plt.ylabel('T-value')
plt.show()
# plt.savefig(scores_dir + 'stats/group_stats_%s_%s_%s.png'%(regressor,subset,sensors))
plt.close()

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
