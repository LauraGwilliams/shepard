import os
import matplotlib.pyplot as plt
import numpy as np
import mne

# paths
base_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'
condition_fname = '%s/group_condition_purepartial_all.npy' % (base_dir)
freq_fname = '%s/group_freq_purepartial_all.npy' % (base_dir)

pure_fname = '%s/group_freq_pure_all.npy' % (base_dir)
partial_fname = '%s/group_freq_partial_all.npy' % (base_dir)


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

plot = 'condfreq'
tonetype = 'pure'

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
    plt.title('Decoding tone-type and frequency across time')

    # # now add the freq
    ax2 = ax.twinx()

    ax2.plot(times, freq_mean, label='frequency', color='Purple')
    ax2.fill_between(times, freq_mean-freq_sem, freq_mean+freq_sem, alpha=0.2,
                     color='Purple')
    ax2.set_ylabel('R')
    ax2.tick_params(axis='y', labelcolor='Purple')
    ax2.set_xlim([-0.2, 0.5])
    ax2.set_ylim([-0.05,0.08])
    ax2.legend()
    plt.title('Decoding tone-type and frequency across time')
    fig.savefig(base_dir + '/group_%s.svg'%(plot),format='svg')

else:
    if tonetype == 'pure':
        ax.plot(times, pure_mean, label='pure', color='Red')
        ax.fill_between(times, pure_mean-pure_sem, pure_mean+pure_sem, alpha=0.2,
                        color='Red')
    if tonetype == 'partial':
        ax.plot(times, partial_mean, label='partial', color='Blue')
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
    ax.tick_params(axis='y', labelcolor='Black')
    ax.legend()
    plt.title('Decoding frequency from %s tones'%(tonetype))
    # fig.savefig(base_dir + '/group_%s.svg'%(tonetype),format='svg')
plt.show()

#
# ax.axhline(.0, color='k', linestyle='--', label='chance')
#
#
# # plt.show()
