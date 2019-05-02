import matplotlib.pyplot as plt
import numpy as np

base_dir = '/Users/ea84/Dropbox/shepard_decoding/GRP_SCORES/'

pure = np.load(base_dir + 'group_scores_pure.npy')
partial = np.load(base_dir + 'group_scores_partial.npy')
shepard = np.load(base_dir + 'group_scores_shepard.npy')
shepard_scale = np.load(base_dir + 'group_scores_shepard_scale.npy')

subjects = ['A0216','A0280','A0305','A0270','P010','P011']
n_subjs = len(subjects)
cols = plt.cm.rainbow(np.linspace(0, 1, n_subjs))
times = np.linspace(-200, 600, 161)

window_idx = [np.logical_and(times > 0, times <= 60),
              np.logical_and(times > 50, times <= 110),
              np.logical_and(times > 100, times <= 200)]

fig, axs = plt.subplots(len(window_idx), 1)

for plt_n, idx in enumerate(window_idx):
    for s in range(n_subjs):
        max_val_pure = np.percentile(pure[s, idx], 95)
        max_val_par = np.percentile(partial[s, idx], 95)
        max_val_shep = np.percentile(shepard[s, idx], 95)
        max_val_shep_scale = np.percentile(shepard_scale[s,:], 95)
        axs[plt_n].scatter(max_val_par, max_val_pure,
                    c=cols[s], label=subjects[s],
                    s=200)
        axs[plt_n].set_xlim([0.0, 0.16])
        axs[plt_n].set_ylim([0.0, 0.16])
plt.legend()
plt.show()
