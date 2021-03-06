import numpy as np
import pandas as pd
import os.path as op
import mne
from scipy.stats import ttest_ind, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

# TABLE OF CONTENTS:
# - make dirs
# - plot overall hemisphere decoding differences
# - plot hemis by high vs low syncs
# - plot hemis (separate or together) by high vs low MSI
# - interaction between hemis and condition

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

# func to find best fit line
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
# MAKE DIRS IF NEEDED

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
# HEMISPHERE DECODING

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group/'
plots_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_PLOTS/n=28/hemis/'


regressor = 'freq'
subset = ['partial']
hemis = ['rh','lh']
syncs = ['high','low']

colors = ['Blue','Green']

lh_scores = np.load(scores_dir+'group_%s_%s_lh.npy'%(regressor,''.join(subset)))
lh_scores_mean = np.mean(lh_scores,axis=0)
lh_sem = np.std( np.array(lh_scores), axis=0 ) / np.sqrt(len(lh_scores))

rh_scores = np.load(scores_dir+'group_%s_%s_rh.npy'%(regressor,''.join(subset)))
rh_scores_mean = np.mean(rh_scores,axis=0)
rh_sem = np.std( np.array(rh_scores), axis=0 ) / np.sqrt(len(rh_scores))

scores = [rh_scores_mean,lh_scores_mean]
errors = [rh_scores_mean-rh_sem,lh_scores_mean-lh_sem]
errors2 = [rh_scores_mean+rh_sem,lh_scores_mean+lh_sem]

times = np.linspace(-200,600,161)
for score, error, error2, lab, color in zip(scores,errors,errors2,hemis,colors):
	plt.plot(times,score,label=lab,color=color)
	plt.fill_between(times,error,error2,alpha=0.2, linewidth=0, color=color)
plt.axhline(y=0,color='Black',linestyle='--')
plt.legend()
if subset == ['pure','partial']:
	plt.title('Decoding pure and partial tones in RH vs LH')
else:
	plt.title('Decoding %s tones in RH vs LH '%(''.join(subset)))
plt.savefig(plots_dir+'%s_%s_hemis.png'%(regressor,''.join(subset)))
plt.show()
plt.close()


#__________________________________________________________
# hemisphere differences by low/high synchronizers

regressor = 'freq'
subset = ['pure']
colors = ['Blue','Green','Purple','Red']
labels = ['rh_high','rh_low','lh_high','lh_low']

lh_scores = np.load(scores_dir+'/group_%s_%s_lh.npy'%(regressor,''.join(subset)))
rh_scores = np.load(scores_dir+'/group_%s_%s_rh.npy'%(regressor,''.join(subset)))

highs = df['High_Low'] == 1
lows = df['High_Low'] == 0
highs = np.array(highs)
lows = np.array(lows)

lh_scores_high = lh_scores[highs]
lh_scores_low = lh_scores[lows]
rh_scores_high = rh_scores[highs]
rh_scores_low = rh_scores[lows]

scores = [rh_scores_high,rh_scores_low,lh_scores_high,lh_scores_low]

times = np.linspace(-200,600,161)
for score, lab, color in zip(scores,labels,colors):
	plt.plot(times,score.mean(0),label=lab,color=color)
plt.axhline(y=0,color='Black',linestyle='--')
# plt.ylim(0,0.05)
plt.legend()
if subset == ['pure','partial']:
	plt.title('Decoding pure and partial tones in RH vs LH in high vs low synchronizers')
else:
	plt.title('Decoding %s tones in RH vs LH in high vs low synchronizers'%(''.join(subset)))
plt.savefig(plots_dir+'%s_%s_lowhigh_hemis.png'%(regressor,''.join(subset)))
plt.show()
plt.close()

#__________________________________________________________
# hemisphere differences by low/high MSI

regressor = 'freq'
subset = ['pure']
colors = ['Blue','deepskyblue','Green','lime']
labels = ['rh_high','rh_low','lh_high','lh_low']

rh_colors = ['Red','lightsalmon']
lh_colors = ['Green','lime']
rh_labels = ['rh_high','rh_low']
lh_labels = ['lh_high','lh_low']


lh_scores = np.load(scores_dir+'/group_%s_%s_lh.npy'%(regressor,''.join(subset)))
rh_scores = np.load(scores_dir+'/group_%s_%s_rh.npy'%(regressor,''.join(subset)))

MSI = 'PLV'
median = df[MSI].median()

highs = df[MSI]>0.68
lows = df[MSI]<0.64
highs = np.array(highs)
lows = np.array(lows)

lh_scores_high = lh_scores[highs]
lh_scores_low = lh_scores[lows]
rh_scores_high = rh_scores[highs]
rh_scores_low = rh_scores[lows]

scores = [rh_scores[highs],rh_scores[lows],lh_scores[highs],lh_scores[lows]]
rh_scores = [rh_scores[highs],rh_scores[lows]]
lh_scores = [lh_scores[highs],lh_scores[lows]]

# plot RH for high vs low MSI
times = np.linspace(-200,600,161)
for score, lab, color in zip(rh_scores,rh_labels,rh_colors):
	plt.plot(times,score.mean(0),label=lab,color=color)
plt.axhline(y=0,color='Black',linestyle='--')
if regressor == 'condition':
	plt.ylim(0.5,0.6)
plt.ylim(-0.02,0.10)
plt.legend()
if subset == ['pure','partial']:
	plt.title('Decoding pure and partial tones in RH \nin high vs low %s'%(MSI))
else:
	plt.title('Decoding %s tones in RH \nin high vs low %s'%(''.join(subset),MSI))
plt.savefig(plots_dir+'%s_%s_lowhigh_%s_RH.png'%(regressor,''.join(subset),MSI))
plt.show()
plt.close()

# plot LH for high vs low MSI
times = np.linspace(-200,600,161)
for score, lab, color in zip(lh_scores,lh_labels,lh_colors):
	plt.plot(times,score.mean(0),label=lab,color=color)
plt.axhline(y=0,color='Black',linestyle='--')
if regressor == 'condition':
	plt.ylim(0.5,0.6)
plt.ylim(-0.02,0.10)
plt.legend()
if subset == ['pure','partial']:
	plt.title('Decoding pure and partial tones in LH \nin high vs low %s'%(MSI))
else:
	plt.title('Decoding %s tones in LH \nin high vs low %s'%(''.join(subset),MSI))
plt.savefig(plots_dir+'%s_%s_lowhigh_%s_LH.png'%(regressor,''.join(subset),MSI))
plt.show()
plt.close()

# plot LH vs RH in low vs high MSI
times = np.linspace(-200,600,161)
for score, lab, color in zip(scores,labels,colors):
	plt.plot(times,score.mean(0),label=lab,color=color)
plt.axhline(y=0,color='Black',linestyle='--')
if regressor == 'condition':
	plt.ylim(0.5,0.6)
# plt.ylim(0,0.05)
plt.legend()
if subset == ['pure','partial']:
	plt.title('Decoding pure and partial tones in RH vs LH in high vs low MSI')
else:
	plt.title('Decoding %s tones in RH vs LH in high vs low %s'%(''.join(subset),MSI))
plt.savefig(plots_dir+'%s_%s_lowhigh_%s_hemis.png'%(regressor,''.join(subset),MSI))
plt.show()
plt.close()

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

#____________________________________________________________
# y-preds compare accuracy

# params
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/indiv/ypreds/'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

subject = 'A0354'
column = ['condition']
train_on = [['purepartial']]
test_on = ['shepard']
regressor = 'freq' #column name

condition = 'all'

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

#____________________________________________________________
# other measures of musical sophistication with PLV

info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'
info = pd.read_csv('%s/MSI_breakdown.csv'%(info_dir))

plots_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_PLOTS/n=28/high_vs_low_sync/'

plt.close()

# set X and Y for plotting
X = info['PLV']
Y = info['Active_Engagement']

corr = X.corr(Y)

# solution
a, b = best_fit(X, Y)
yfit = [a + b * xi for xi in X]

# grab high vs low synchronizers
X_high = []
X_low = []
Y_high = []
Y_low = []
# plot points and fit line
for point in range(len(X)):
	if X[point] > 0.5:
		# plt.scatter(X[point],Y[point], color ='g')
		X_high.append(X[point])
		Y_high.append(Y[point])
	else:
		X_low.append(X[point])
		Y_low.append(Y[point])
		# plt.scatter(X[point],Y[point], color ='b')
plt.scatter(X_high,Y_high,color='g',label='high synchronizers')
plt.scatter(X_low,Y_low,color='b',label='low synchronizers')
# plt.scatter(X, Y)
plt.xlabel(X.name)
plt.ylabel(Y.name)
plt.title('%s vs. %s'%(X.name,Y.name))
plt.ylim(Y.min()-10,Y.max()+10)
plt.axvline(x=0.5, color='Gray', linestyle='--',label='0.5 PLV')
plt.plot(X, yfit, color='Black', label='best fit, r = %s'%(corr))
plt.legend(prop={'size': 8})
plt.savefig(plots_dir+'%s_%s_corr.png'%(X.name,Y.name))
plt.show()
