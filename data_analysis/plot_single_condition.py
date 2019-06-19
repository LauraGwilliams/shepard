import os
import matplotlib.pyplot as plt
import numpy as np
import mne

# params
train = 'pure'
test = 'partial'
regressor = 'freq'
subset = 'partial'

base_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=20'

# single_fname = '%s/group_%s_train%s_test%s.npy' % (base_dir,regressor,train,test)
single_fname = '%s/group_%s_%s.npy' % (base_dir,regressor,subset)

# load data
single_scores = np.load(single_fname)

# params
n_subj, n_times = single_scores.shape
times = np.linspace(-0.2, 0.6, n_times)

if regressor == 'condition':
    # compute baseline accuracy
    baseline_idx = times < 0
    condition_scores = condition_scores - condition_scores[:, baseline_idx].mean()

# compute mean and sem over subjects
single_mean = single_scores.mean(0)
single_sem = np.std(single_scores, axis=0) / np.sqrt(n_subj)

# plot it
fig, ax = plt.subplots()

# pure, first
ax.plot(times, single_mean,  color='Blue')
ax.fill_between(times, single_mean-single_sem, single_mean+single_sem, alpha=0.2,
                color='Blue')

# ax.plot(times, freq_mean, label='all', color='Black')
# ax.fill_between(times, freq_mean-freq_sem, freq_mean+freq_sem, alpha=0.2,
#                  color='Black')
ax.axhline(0, color='k', linestyle='--')
ax.axvline(0, color='gray', linestyle='-')
ax.axvline(0.3, color='gray', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('R')
ax.set_xlim([-0.2, 0.5])
# ax.tick_params(axis='y', labelcolor='Red')
# ax.legend()
plt.show()

#
# ax.axhline(.0, color='k', linestyle='--', label='chance')
#
#
# ax.set_title('Decoding MEG sensors over time')
# plt.savefig(meg_dir + '%s_%s_%s.png'%(subject,regressor,''.join(subset[0])))
# # plt.show()
