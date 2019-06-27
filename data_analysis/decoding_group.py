import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne import read_epochs, read_evokeds
from scipy.stats import sem as scipy_sem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, cross_val_score, cross_val_predict,
                                    StratifiedKFold)
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, get_coef,
                        LinearModel, cross_val_multiscore)
import mne.decoding
from mne.stats import permutation_cluster_1samp_test as perm_1samp
from jr import scorer_spearman # will have to download git repos & import some stuff
from sklearn.metrics import make_scorer, get_scorer
import matplotlib.cm as cm


#------------------------------------------------------------------------
# scaling funcs

def my_scaler(x):
    '''
    Scale btw 0-1.
    '''
    x = np.array(x).astype(float)
    return (x - (np.min(x)) / (np.max(x) - np.min(x)))

def binary_scaler(x):
    '''
    Pure = 1, partial = 0.
    '''
    x = np.array(x)
    x[x==subset[0][0]]=1.
    x[x==subset[0][1]]=0.
    x = np.array(x).astype(float)
    return x

#------------------------------------------------------------------------
# stats funcs

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
            start_idx = int(str(c.start))-1
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

#------------------------------------------------------------------------
# split sensors

def grab_hemi_sensors(epochs,exclude_center=False):

    # grab info and channels from epochs
    info = epochs.info
    chs = info['chs'] # this is a dictionary of channels and their info

    if exclude_center:
        thresh = 0.03
    else:
        thresh = 0

    rh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] > thresh]]
    lh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] < -thresh]]

    rh_picks = mne.pick_types(epochs.info,selection=rh_ch_names)
    lh_picks = mne.pick_types(epochs.info,selection=lh_ch_names)

    return rh_picks, lh_picks

#-------------------------------------------------------------------------

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

# music_soph = [54.5,75,42.5,0,92,0,74,37,48,86,65.5,44.5,48.5,66,102.5]

# params

# epochs subset to train on
column = ['condition','circscale']
subset = [['shepard'],['circular']]
sensor_list = ['all']

# regressor to decode, spatial vs. temporal vs. combined
regressor = 'note_position' #column name
decode_using = 'spatial' # spatial (trials x sensors x time)
                        # temporal (trials x time x sensors),
                        # combined (trials x sensors*time)

meg_dir = '/Users/ea84/Dropbox/shepard_decoding/'

#-------------------------------------------------------------------------
grp_scores = []

scores_arr = np.array([])

for sensors in sensor_list:
    for subject in subjects:
        print (subject)

        # load all data
        allepochs = meg_dir + '%s/%s_shepard-epo.fif'%(subject,subject)
        epochs = mne.read_epochs(allepochs)

        # subset current_epochs based on parameters
        if len(column) > 1:
            current_epochs = epochs[epochs.metadata[column[0]].isin(subset[0]) &
                                epochs.metadata[column[1]].isin(subset[1])]

        else:
            current_epochs = epochs[epochs.metadata[column[0]].isin(subset[0])]

        trial_info = current_epochs.metadata

        if subject[0] == 'A':
            ch = 208
        else:
            ch = 157

        rh_picks, lh_picks = grab_hemi_sensors(current_epochs,exclude_center=True)
        if sensors == 'rh':
            X = current_epochs._data[:, rh_picks, :]
        elif sensors == 'lh':
            X = current_epochs._data[:, lh_picks, :]
        else:
            # pull out sensor data from meg channels only
            X = current_epochs._data[:, 0:ch, :]

        # pull out regressor of interest
        y = trial_info[regressor].values

        #------------------------------------------------------------------------
        # change X if applicable
        if decode_using == 'temporal':
            # switch channels and times to decode using time course
            X = np.transpose(X, [0, 2, 1])
        if decode_using == 'combined':
            # collapse over spatial and temporal dimensions
            [n_trials, n_sns, n_times] = X.shape
            X = np.reshape(X, [n_trials, n_sns*n_times])

        #------------------------------------------------------------------------

        # NOTE: it is important to make sure that the y array is in float
        # format not integer, otherwise it turns it into a binary problem.
        # Also, it doesn't make sense to use statified KFold for regression problem.
        # because there are no classes to speak of. KFold instead. My bad.

        # set up decoder, use logistic for categorical and Ridge for continuous
        if (regressor == 'freq') | (regressor == 'note_position'):
            y = my_scaler(y) # scale frequencies to between 0 and 1
            clf = make_pipeline(StandardScaler(), Ridge())
            scorer = make_scorer(get_scorer(scorer_spearman))
            score = 'Spearman R'
            cv = KFold(5)
        if regressor == 'condition':
            y = binary_scaler(y) # set values to 0.0 and 1.0
            clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
            scorer = 'roc_auc'
            score = 'AUC'
            cv = StratifiedKFold(5)

        n_jobs = -1

        # set up estimator, get scores
        if decode_using == 'spatial':
            gen = SlidingEstimator(n_jobs=n_jobs,
                                    scoring=scorer,
                                    base_estimator=clf)
            scores = cross_val_multiscore(gen, X, y,
                                          cv=cv)
        elif decode_using == 'temporal':
            gen = SlidingEstimator(n_jobs=n_jobs,
                                scoring=scorer,
                                base_estimator=clf)
            scores = cross_val_multiscore(gen, X, y,
                                        cv=cv)
        else:
            # scoring defaults to neg mean squared so set to scorer
            # shuffle must be true when binary values, otherwise fold will only have one value
            scores = cross_val_score(clf, X, y,
                                    scoring=scorer, #defaults to neg mean squared
                                    cv=KFold(5, shuffle=True))

        # mean scores across cross-validation splits
        scores = np.mean(scores, axis=0)
        grp_scores.append(scores)
        # scores_arr = np.array(grp_scores)
        # np.save(meg_dir+'_GRP_SCORES/n=%i/indiv/%s_%s_%s/%s_%s_%s_%s.npy'%(len(subjects),regressor,''.join(subset[0]),sensors,
        #                                                         subject,regressor,''.join(subset[0]),sensors), scores_arr)


    scores_arr = np.array(grp_scores)
    np.save(meg_dir+'_GRP_SCORES/n=%i/group_%s_%s_%s.npy'%(len(subjects),regressor,
                                                        ''.join(subset[0]),sensors),
                                                        scores_arr)

    grp_sem = np.std( np.array(grp_scores), axis=0 ) / np.sqrt(len(grp_scores))
    grp_avg = np.mean( np.array(grp_scores), axis=0 )

# ---------------------------------------------------------------------------
# PLOTTING

if decode_using == 'spatial':
    fig, ax = plt.subplots()
    ax.plot(epochs.times, grp_avg, label='score')
    # ax.fill_between(epochs.times, np.diag(grp_avg-grp_sem), np.diag(grp_avg+grp_sem),
    #                     alpha=0.2, linewidth=0, color='r')
    ax.fill_between(epochs.times, grp_avg-grp_sem, grp_avg+grp_sem,
                        alpha=0.2, linewidth=0, color='r')
    if score == 'AUC':
        ax.axhline(.5, color='k', linestyle='--', label='chance')
    else:
        ax.axhline(.0, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('%s'%(score))
    ax.legend()
    # ax.set_ylim(bottom=-0.035, top=0.16)
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    plt.savefig(meg_dir + '_GRP_PLOTS/n=%i/group/group_%s_%s_%s.png'%(len(subjects),regressor,''.join(subset[0]),sensors))
    plt.show()
#
# elif decode_using == 'temporal':
#     # load in evoked object to plot on
#     scores_evoked = mne.read_evokeds(meg_dir + '%s/%s_shepard-evoked-ave.fif'%(subject,subject))[0]
#     scores_evoked._data[0:ch, 0] = scores
#     if regressor == 'freq':
#         scores_evoked.plot_topomap(times=scores_evoked.times[0])
#     if regressor == 'condition':
#         # center around 0 for plotting
#         scores_evoked._data[0:ch, 0] = scores_evoked._data[0:ch, 0] - scores_evoked._data[0:ch, 0].mean()
#         scores_evoked.plot_topomap(times=scores_evoked.times[0])
#
#     coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
#     evoked = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
#     joint_kwargs = dict(ts_args=dict(time_unit='s'),
#                         topomap_args=dict(time_unit='s'))
#     evoked.plot_joint(times=np.arange(0., .500, .100), title='patterns',
#                       **joint_kwargs)
# else:
#     # if combined, it just returns an overall score (nothing to plot!)
#     print (scores)
# norm = matplotlib.colors.Normalize(vmin=0,vmax=103)
# # colors = cm.rainbow(np.linspace(0, 1, len(subjects)))

info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'
info = pd.read_csv('%s/subj_msi_sync.csv'%(info_dir))

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group/'

times = np.linspace(-200,600,161)

grp_scores = np.load(scores_dir+'group_%s_%s_%s.npy'%(regressor,subset,hemi))
if decode_using == 'spatial':
    fig, ax = plt.subplots()
    for subject,scores,music in zip(subjects,grp_scores,music_soph):
        # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
        color = cm.rainbow(norm(music),bytes=True)
        new_color = tuple(ti/255.0 for ti in color)
        ax.plot(epochs.times, scores, color=new_color, label=subject)
        # ax.fill_between(epochs.times, np.diag(grp_avg-grp_sem), np.diag(grp_avg+grp_sem),
        #                     alpha=0.2, linewidth=0, color='r')
    if score == 'AUC':
        ax.axhline(.5, color='k', linestyle='--', label='chance')
    else:
        ax.axhline(.0, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('%s'%(score))
    ax.legend()
    # plt.colorbar()
    # ax.set_ylim(bottom=-0.035, top=0.16)
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
        # plt.savefig(meg_dir + '_GRP_PLOTS/group_%s_%s.png'%(regressor,''.join(subset[0])))
    plt.show()
