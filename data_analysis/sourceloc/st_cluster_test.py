# leg5@nyu.edu
# source space regression-based spatio-temporal cluster test
# 1) load epochs and variables
# 2) apply regression within each subject
# 3) perform 1-samp cluster test on the beta values against zero over
#    subjects. access significance with random permutations.

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import eelbrain
import mne
from mne import (find_events, read_epochs, Epochs)
from mne.io import read_raw_fif
from mne.stats import linear_regression
# from mne.stats import permutation_cluster_1samp_test as p1samp
from mne.stats import spatio_temporal_cluster_1samp_test as p1samp

from mne.source_estimate import read_source_estimate
from mne.minimum_norm import (read_inverse_operator, apply_inverse_epochs)
from mne import spatial_tris_connectivity, grade_to_tris

# params
subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

var_names = ['freq', 'condition', 'circscale', 'interaction']  # get rid of shepard and add interaction (multiplication of freq x TT)
data_type = 'beta'

# cluster test params
n_perm = 10000  # 10 is OK if we are just saving the observed t-vals
min_times = 3  # pay attention to decimate here - its in time samples
min_sources = 10  # min number of sources in cluster
min_p = 0.05

# paths
root_dir = '/Users/ea84/Dropbox/shepard_sourceloc'
subjects_dir = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri'
os.environ["SUBJECTS_DIR"] = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri/'

def threshold_clusters(clusters, cluster_pv, min_times, min_sources, min_p):
    """loop through each cluster, and keep the ones that exeed threshold"""

    # init
    cluster_bool = np.zeros([len(clusters)])
    keep_clusters = []

    # loop through each cluster and test spatio-temporal extent
    for ii, cluster in enumerate(clusters):
        print(ii)
        p_obs = cluster_pv
        n_sources = sum(sum(cluster))
        n_times = sum(sum(cluster.T))
        if (n_times > min_times) & (n_sources > min_sources) & np.unique(p_obs <= min_p)[0]:
            keep_clusters.append(cluster)
            cluster_bool[ii] = 1
    print("%s clusters surpassed constraints" % sum(cluster_bool))
    msg = "The most significant surviving cluster is p ="
    print("%s %s" % (msg, np.min(cluster_pv[cluster_bool == 1])))

    # return 0/1 of which clusters survived
    return cluster_bool


all_betas = []
# loop through subjects
for subject in subjects:
    print(subject)
    meg_dir = '%s/%s/'%(root_dir,subject)
    fifs_dir = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard_fifs/ica/'
    epochs_fname = meg_dir + subject+ '_shepard-epo.fif'
    ica_raw_fname = fifs_dir + subject+ '_ica_shepard-raw.fif' # applied ica to raw
    ica_rej_fname = meg_dir + subject+ '_shepard_rejfile.pickled' # rejected epochs after ica
    inv_fname = meg_dir+subject+'_shepard-inv.fif'
    print ("Reading raw with ICA applied...")
    raw = read_raw_fif(ica_raw_fname, preload=True)

    print ("Finding events...")
    events = find_events(raw)  # the output of this is a 3 x n_trial np array

    print ("Epoching data...")
    epochs = Epochs(raw, events, tmin=-0.2, tmax=0.6, decim = 10, baseline=(-0.2,0.6), preload=True)
    epochs = epochs.crop(tmin=-0.1,tmax=0.5)

    del raw

    rejfile = eelbrain.load.unpickle(ica_rej_fname)
    rejs = rejfile['accept'].x
    epochs = epochs[rejs]

        # load vars
    print ("Loading trial info...")
    X_fname = meg_dir+'/stcs/%s_shepard_rej_trialinfo.csv'%(subject)
    events = pd.read_csv(X_fname)

    # GET RID OF SHEPARD
    epochs.metadata = events
    epochs = epochs[epochs.metadata['condition']!='shepard']

    events = events[events['condition'].values!='shepard']

    # load brain data - generate stc
    inverse_operator = read_inverse_operator(inv_fname)
    src = inverse_operator['src']
    snr = 2.0  # Standard assumption for single trial
    lambda2 = 1.0 / snr ** 2
    spacing = 4 ## density of the source reconstruction

    # apply inverse to epochs
    print ("Creating stcs...")
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                      method='dSPM')

    for col,cond in zip(['condition','circscale'],['pure','scale']):
        list = []
        for i in range(len(events)):
            list.append(int(events.iloc[i][col]==cond))
        events[col] = list

    events['interaction'] = events['freq']*events['condition']

    X = events[var_names].values
    X = sm.add_constant(X)  # add intercept

    print ("Fitting...")
    # fit regression
    lm = linear_regression(stcs, X, ['intercept']+var_names)

    betas = []  # will contain betas over regressors per subject
    for regressor in var_names:
        print ('Regressor: %s'%(regressor))

        # make directories if they dont exists
        save_dir = "%s/_RESULTS/%s" % (root_dir, regressor)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)        # put the data into an stc for convenience, and save
        stc_fname = '%s/morphed_stcs/%s_%s_%s_morphed' % (save_dir, subject, regressor,
                                             data_type)
        stc_data = eval("lm['%s'].%s" % (regressor, data_type))

        # morph the stc to average brain
        morph = mne.compute_source_morph(stc_data, subject_from=subject,
                                 subject_to='fsaverage',
                                 subjects_dir=subjects_dir,
                                 spacing=spacing)
        stc_fsaverage = morph.apply(stc_data)
        # stc_fsaverage = read_source_estimate(stc_fname)
        stc_fsaverage.save(stc_fname)
        betas.append(stc_fsaverage._data)

    all_betas.append(betas)

# convert to np
all_betas = np.array(all_betas, float)

# dimension order should be obs x time x source
all_betas = np.transpose(all_betas, [0, 1, 3, 2])

for ii, regressor in enumerate(var_names):

    save_dir = "%s/_RESULTS/%s" % (root_dir, regressor)
    # apply the permutation test
    connectivity = spatial_tris_connectivity(grade_to_tris(spacing))
    t_obs, clusters, cluster_pv, H0 = p1samp(all_betas[:, ii, ...], threshold=4,
                                             n_permutations=n_perm,
                                             connectivity=connectivity,
                                             buffer_size=None,
                                             verbose=True)
    # save result of the cluster test
    np.save('%s/_stats_%s.npy' % (save_dir, regressor), (t_obs, clusters, cluster_pv, H0))

    # remove clusters that are too short or too small or not significant
    # cluster_bool = threshold_clusters(clusters, cluster_pv, min_times,
    #                                   min_sources, min_p)

    # save the t-value map as an stc. note: can extract np data with stc._data
    stc_fname = '%s/tvals_oversubj_%s.npy' % (save_dir, regressor)
    # morph the stc to average brain
    stc_fsaverage._data = t_obs.T
    stc_fsaverage.save(stc_fname)


##### PLOTTING ######
def time_label(t):
    return "time=%0.2f ms" % (1e3 * t)
brain = Brain('fsaverage', 'split', 'partially_inflated', size=(800, 400))
for hemi in ['lh', 'rh']:
    stc = read_stc(stc_fname+'-%s.stc' % hemi)
    data = stc['data']
    times = np.arange(data.shape[1]) * stc['tstep'] + stc['tmin']
    brain.add_data(data, colormap='RdBu', vertices=stc['vertices'],
                   smoothing_steps=10, time=times, time_label=time_label,
                   initial_time=-0.1,hemi=hemi)

abs_max = (np.abs(data)).max()
brain.scale_data_colormap(fmin=0, fmid=abs_max/3, fmax=abs_max, center=0, transparent=True)

brain.save_movie(save_dir+'/%s_%s_betas.mov'%(subject,regressor), tmin=-0.1,tmax=0.5,time_dilation=10)
