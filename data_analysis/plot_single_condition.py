import os
import matplotlib.pyplot as plt
import numpy as np
import mne

# params
train = 'partial'
test = 'pure'
regressor = 'freq'
subset = 'pure'
sensors = 'all'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

base_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=%s/group'% len(subjects)

gen_fname = '%s/group_%s_train%s_test%s_%s.npy' % (base_dir,regressor,train,test,sensors)
single_fname = '%s/group_%s_%s_%s.npy' % (base_dir,regressor,subset,sensors)

# load data
single_scores = np.load(single_fname)
gen_scores = np.load(gen_fname)

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

gen_mean = gen_scores.mean(0)
gen_sem = np.std(gen_scores, axis=0) / np.sqrt(n_subj)

# plot it
fig, ax = plt.subplots()

# pure, first
ax.plot(times, single_mean,  color='Red')
ax.fill_between(times, single_mean-single_sem, single_mean+single_sem, alpha=0.2,
                color='Red')
# ax.plot(times, gen_mean,  color='Red')
# ax.fill_between(times, gen_mean-gen_sem, gen_mean+gen_sem, alpha=0.2,
#                 color='Red')

# ax.plot(times, freq_mean, label='all', color='Black')
# ax.fill_between(times, freq_mean-freq_sem, freq_mean+freq_sem, alpha=0.2,
#                  color='Black')
ax.axhline(0, color='k', linestyle='--')
ax.axvline(0, color='gray', linestyle='-')
ax.axvline(0.3, color='gray', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('R')
ax.set_xlim([-0.2, 0.5])
# ax.set_title('Trained on: %s, Tested on: %s, Decoding: %s'%(train,test,regressor))
ax.set_title('Decoding %s from %s tones'%(regressor,subset))

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
