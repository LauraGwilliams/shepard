import os
import matplotlib.pyplot as plt
import numpy as np
import mne

# paths
base_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=15'
condition_fname = '%s/group_condition_purepartial.npy' % (base_dir)
freq_fname = '%s/group_freq_purepartial.npy' % (base_dir)

pure_fname = '%s/group_freq_pure.npy' % (base_dir)
partial_fname = '%s/group_freq_partial.npy' % (base_dir)


# load data
condition_scores = np.load(condition_fname)
freq_scores = np.load(freq_fname)
pure_scores = np.load(pure_fname)
partial_scores = np.load(partial_fname)

# params
n_subj, n_times = pure_scores.shape
times = np.linspace(-0.2, 0.6, n_times)

# compute baseline accuracy
baseline_idx = times < 0
condition_scores = condition_scores - condition_scores[:, baseline_idx].mean()
freq_scores = freq_scores - freq_scores[:, baseline_idx].mean()

# compute mean and sem over subjects
cond_mean = condition_scores.mean(0)
cond_sem = np.std(condition_scores, axis=0) / np.sqrt(n_subj)
freq_mean = freq_scores.mean(0)
freq_sem = np.std(freq_scores, axis=0) / np.sqrt(n_subj)
pure_mean = pure_scores.mean(0)
pure_sem = np.std(pure_scores, axis=0) / np.sqrt(n_subj)
partial_mean = partial_scores.mean(0)
partial_sem = np.std(partial_scores, axis=0) / np.sqrt(n_subj)

# plot it
fig, ax = plt.subplots()

plot = "condfreq"

if plot == "condfreq":
    # condition, first
    ax.plot(times, cond_mean, label='tone-type', color='Gray')
    ax.fill_between(times, cond_mean-cond_sem, cond_mean+cond_sem, alpha=0.2,
                    color='Gray')
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(0, color='gray', linestyle='-')
    ax.axvline(0.3, color='gray', linestyle='-')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.set_xlim([-0.2, 0.5])
    ax.tick_params(axis='y', labelcolor='Gray')
    ax.legend(loc='upper left')

    # now add the freq
    ax2 = ax.twinx()

    ax2.plot(times, freq_mean, label='frequency', color='Purple')
    ax2.fill_between(times, freq_mean-freq_sem, freq_mean+freq_sem, alpha=0.2,
                     color='Purple')
    ax2.set_ylabel('R')
    ax2.tick_params(axis='y', labelcolor='Purple')
    ax2.set_xlim([-0.2, 0.5])
    ax2.legend()
else:
    # pure, first
    ax.plot(times, pure_mean, label='pure', color='Red')
    ax.fill_between(times, pure_mean-pure_sem, pure_mean+pure_sem, alpha=0.2,
                    color='Blue')
    ax.plot(times, partial_mean, label='partials', color='Blue')
    ax.fill_between(times, partial_mean-partial_sem, partial_mean+partial_sem, alpha=0.2,
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
    ax.tick_params(axis='y', labelcolor='Blue')
    ax.legend()
plt.show()

#
# ax.axhline(.0, color='k', linestyle='--', label='chance')
#
#
# ax.set_title('Decoding MEG sensors over time')
# plt.savefig(meg_dir + '%s_%s_%s.png'%(subject,regressor,''.join(subset[0])))
# # plt.show()
