import numpy as np
import pandas as pd
import os.path as op
import mne
from scipy.stats import ttest_ind, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

#____________________________________________________________
# funcs

def cluster_test(scores, n_perm=10000, threshold=1.5, n_jobs=1,
				 dimension='time'):
    '''perform 1-sample t-test over time or space to find clusters'''

    # dims
    n_obs, n_datapoints = scores.shape

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

#____________________________________________________________
# make dirs if needed
import shutil
import os

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group/'
plots_dir =  '/Users/ea84/Dropbox/shepard_decoding/_GRP_PLOTS/n=28/'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

n = len(subjects)

# make new subject dirs
for subject in subjects:
    if not os.path.exists(meg_dir+'_GRP_PLOTS/%i/indiv/%s/'%(n,subject)):
        os.makedirs(meg_dir+'_GRP_PLOTS/n=%i/indiv/%s/'%(n,subject))

# move .pngs to individual plot folders
for subject in subjects:
	source_path = '/Users/ea84/Dropbox/shepard_decoding/%s/'%(subject)
	source_files = os.listdir(source_path)
	dest_path = '/Users/ea84/Dropbox/shepard_decoding/_GRP_PLOTS/n=28/indiv/%s/'%(subject)
	for file in source_files:
		if file.endswith('.png'):
			shutil.move(os.path.join(source_path,file), os.path.join(dest_path,file))


#____________________________________________________________
# hemisphere decoding scores

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group/'

regressor = 'freq'
subset = ['partial']
hemis = ['rh','lh']


lh_scores = np.load(scores_dir+'group_%s_%s_lh.npy'%(regressor,''.join(subset)))
lh_scores_mean = np.mean(lh_scores,axis=0)

rh_scores = np.load(scores_dir+'group_%s_%s_rh.npy'%(regressor,''.join(subset)))
rh_scores_mean = np.mean(rh_scores,axis=0)

ttest_ind(rh_scores_mean,lh_scores_mean)

#____________________________________
# interaction hemi, condition

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group/'
info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'

info = pd.read_csv('%s/subj_msi_sync.csv'%(info_dir))
MSI = list(info['MSI'])
sync = list(info['sync'])

regressor = 'freq'
subset = ['partial','pure']
hemis = ['lh','rh']

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']



rows_list = []

for i in range(len(subjects)):
	for condition in subset:
		for hemi in hemis:
			scores = np.load(scores_dir+'group_%s_%s_%s.npy'%(regressor,condition,hemi))
			row = dict()
			row['subject_number'] = subjects[i]
			row['MSI'] = MSI[i]
			row['sync'] = sync[i]
			row['hemi'] = hemis.index(hemi)
			row['condition'] = subset.index(condition)
			row['interaction'] = row['sync'] * row['condition']
			for tt in range(161):
				row['data_tt%s'%(tt)] = scores[i][tt]
			rows_list.append(row)

df = pd.DataFrame(rows_list)

# hemi % condition
coef_bin = np.zeros([161, 4])
for tt in range(161):
    md = smf.mixedlm('data_tt%s ~ hemi * condition' % (tt), df,
	groups=df['subject_number'], re_formula="~hemi*condition")
    mdf = md.fit()
	# add intercept, and first 3 factor coefs to the bin
    coef_bin[tt, :] = mdf.params[0:4]

var_labels = ['intercept', 'hemi', 'condition', 'interaction']
times = np.linspace(-200, 600, 161)
for coef, lab in zip(coef_bin.T, var_labels):
	 plt.plot(times, coef, label=lab)
plt.legend()
plt.show()

# sync % condition
coef_bin = np.zeros([161, 4])
for tt in range(161):
    md = smf.mixedlm('data_tt%s ~ sync* condition' % (tt), df,
	groups=df['subject_number'], re_formula="~sync")
    mdf = md.fit()
	# add intercept, and first 3 factor coefs to the bin
    coef_bin[tt, :] = mdf.params[0:4]

var_labels = ['intercept', 'sync', 'condition', 'interaction']
times = np.linspace(-200, 600, 161)
for coef, lab in zip(coef_bin.T, var_labels):
	 plt.plot(times, coef, label=lab)
plt.legend()
plt.show()
#____________________________________________________________
# train on x, evaluate on shepard scores

regressor = 'freq'
train_on = ['pure','partial']
test_on = ['shepard']

pure_scores = np.load(scores_dir+'group_%s_train%s_test%s.npy'%(regressor,train_on[0],test_on[0]))
pure_scores_mean = np.mean(pure_scores,axis=0)

partial_scores = np.load(scores_dir+'group_%s_train%s_test%s.npy'%(regressor,train_on[1],test_on[0]))
partial_scores_mean = np.mean(partial_scores,axis=0)

ttest_ind(pure_scores_mean,partial_scores_mean)

#____________________________________________________________
# y-preds compare accuracy

# params
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/indiv/ypreds/'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

subject = 'P015'
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

#____________________________________________________________
# ambiguous tone exploration

# get ypreds for upFirst, upLast, downFirst, downLast ambiguous shepard tones
up_first = csv[(csv['freq']==220) & (csv['key']=='A') & (csv['circscale']=='scale')
				& (csv['updown'] == 'up') & (csv['note_position']==1)]
mask = csv.index[(csv['freq']==220) & (csv['key']=='A') & (csv['circscale']=='scale')
				& (csv['updown'] == 'up') & (csv['note_position']==1)].tolist()
y_preds = np.transpose(y_preds, [1,0])
y_preds = y_preds[mask]
y_preds = np.transpose(y_preds, [1,0])

y_test = np.array(up_first['freq'])

up_last = csv[(csv['freq']==220) & (csv['key']=='A') & (csv['circscale']=='scale')
				& (csv['updown'] == 'up') & (csv['note_position']==8)]
down_first = csv[(csv['freq']==220) & (csv['key']=='A') & (csv['circscale']=='scale')
				& (csv['updown'] == 'down') & (csv['note_position']==1)]
down_last = csv[(csv['freq']==220) & (csv['key']=='A') & (csv['circscale']=='scale')
				& (csv['updown'] == 'down') & (csv['note_position']==8)]

#_______________________________
# plot based on MSI (color scale ranges from lowest to highest MSI)

import matplotlib.cm as cm
import matplotlib

regressor = 'freq'
subset = ['partial']
hemi = 'all'

info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'
info = pd.read_csv('%s/subj_msi_sync.csv'%(info_dir))

subjects = list(info['subject'])
MSI = list(info['MSI'])
sync = list(info['sync'])

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group/'

times = np.linspace(-200,600,161)

grp_scores = np.load(scores_dir+'group_%s_%s_%s.npy'%(regressor,''.join(subset),hemi))

# create color scale
norm = matplotlib.colors.Normalize(vmin=0,vmax=103)
colors = cm.rainbow(np.linspace(0, 1, len(subjects)))

fig, ax = plt.subplots()
for subject,scores,music in zip(subjects,grp_scores,MSI):
    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    color = cm.rainbow(norm(music),bytes=True)
    new_color = tuple(ti/255.0 for ti in color)
    ax.plot(times, scores, color=new_color, label=subject)
    # ax.fill_between(epochs.times, np.diag(grp_avg-grp_sem), np.diag(grp_avg+grp_sem),
    #                     alpha=0.2, linewidth=0, color='r')
if score == 'AUC':
    ax.axhline(.5, color='k', linestyle='--', label='chance')
else:
    ax.axhline(.0, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('Scores')
ax.legend()
# plt.colorbar()
# ax.set_ylim(bottom=-0.035, top=0.16)
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')
    # plt.savefig(meg_dir + '_GRP_PLOTS/group_%s_%s.png'%(regressor,''.join(subset[0])))
plt.show()
