import os
import matplotlib.pyplot as plt
import numpy as np
import mne


# paths
base_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/shepard/data_analysis'
condition_fname = '%s/group_scores_condition_purepartial.npy' % (base_dir)
freq_fname = '%s/group_scores_freq_purepartial.npy' % (base_dir)

# load data
condition_scores = np.load(condition_fname)
freq_scores = np.load(freq_fname)

# params
n_subj, n_times = condition_scores.shape
times = np.linspace(-0.2, 0.6, n_times)

# compute baseline accuracy
baseline_idx = times < 0
condition_scores = condition_scores - condition_scores[:, baseline_idx].mean()
# freq_scores = freq_scores - freq_scores[:, baseline_idx].mean()

# compute mean and sem over subjects
cond_mean = condition_scores.mean(0)
cond_sem = np.std(condition_scores, axis=0) / np.sqrt(n_subj)
freq_mean = freq_scores.mean(0)
freq_sem = np.std(freq_scores, axis=0) / np.sqrt(n_subj)

# plot it
fig, ax = plt.subplots()

# condition, first
ax.plot(times, cond_mean, label='condition', color='Blue')
ax.fill_between(times, cond_mean-cond_sem, cond_mean+cond_sem, alpha=0.2,
                color='Blue')
ax.axhline(0, color='k', linestyle='--')
ax.axvline(0, color='gray', linestyle='-')
ax.axvline(0.3, color='gray', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.set_xlim([-0.2, 0.5])
ax.tick_params(axis='y', labelcolor='Blue')
ax.legend(loc='upper left')

# now add the freq
# ax2 = ax.twinx()
#
# ax2.plot(times, freq_mean, label='pitch', color='Green')
# ax2.fill_between(times, freq_mean-freq_sem, freq_mean+freq_sem, alpha=0.2,
#                  color='Green')
# ax2.set_ylabel('R')
# ax2.tick_params(axis='y', labelcolor='Green')
# ax2.set_xlim([-0.2, 0.5])
# ax2.legend()

plt.show()

#
# ax.axhline(.0, color='k', linestyle='--', label='chance')
#
#
# ax.set_title('Decoding MEG sensors over time')
# plt.savefig(meg_dir + '%s_%s_%s.png'%(subject,regressor,''.join(subset[0])))
# # plt.show()
