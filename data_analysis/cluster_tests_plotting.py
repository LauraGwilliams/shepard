import numpy as np
import pandas as pd
import os.path as op
from scipy.stats import sem as scipy_sem
import matplotlib.pyplot as plt
# from config import subjects_rep
from mne.stats import permutation_cluster_1samp_test as perm_1samp
import glob

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

# params
n = 28
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=%i/'%(n)
regressor = 'condition'
subset = 'purepartial'
sensor_list = ['all','rh','lh']

for sensors in sensor_list:
	# get scores
	scores_fname = scores_dir + 'group/group_%s_%s_%s.npy'%(regressor,subset,sensors)
	scores = np.load(scores_fname)

	if regressor == 'condition':
		chance = 0.5
	else:
		chance = 0

	scores = scores - chance

	# cluster test
	res = cluster_test(scores, n_perm=10000, threshold=1.5, n_jobs=1,
	    				dimension='time')
	t_obs, clusters, cluster_pv, H0 = res

	plot_clusters(t_obs, clusters, cluster_pv, label=regressor)
	plt.title('Decoding %s over time'%(regressor))
	plt.xlabel('Times')
	plt.ylabel('T-value')
	plt.savefig(scores_dir + 'stats/group_stats_%s_%s_%s.png'%(regressor,subset,sensors))
	plt.close()
	# plt.show()
