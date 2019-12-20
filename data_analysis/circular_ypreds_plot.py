import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# paths
base_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/indiv/ypreds'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']
n_subjs = len(subjects)
sensors = 'rh'

# fig, axs = plt.subplots(4, 7, figsize=(40, 15))
# fig.subplots_adjust(wspace=0.2,hspace=0.4)
# axs = axs.ravel()

for trained_on in ['pure']:
    condition = 'circular'
    dir = 'up'

    if trained_on == 'pure':
        c = 'r'
    else:
        c = 'b'

    if condition == 'circular':
        num_tones = 7
    else:
        num_tones = 8

    np_preds = list()
    for i in range(num_tones):
        idx_preds = list()
        print i
        for subject in subjects:

            # load ypreds
            info_fname = '%s/%s/%s_freq_train%s_ypreds_%s.csv' % (base_dir, subject, subject,
                                                          trained_on,sensors)
            trial_info = pd.read_csv(info_fname)

            ypred_fname = '%s/%s/%s_freq_train%s_ypreds_%s.npy' % (base_dir, subject, subject,
                                                          trained_on,sensors)
            ypred = np.load(ypred_fname)

            # sanity
            assert(trial_info.shape[0] == ypred.shape[1])

            # dims
            n_times, n_trials = ypred.shape


            # subset just the scale trials
            # idx = np.logical_and(trial_info['circscale'].values == 'scale',
            #                      trial_info['highest_lowest'].values == 1)
            idx = np.logical_and(trial_info['circscale'].values == condition,
                                    trial_info['updown'].values == dir)
            trial_info = trial_info[idx]
            ypred = ypred[:, idx]

            # get ypreds for each note position
            preds = list()
            idx_np = trial_info['note_position'].values == i+1
            preds.append(ypred[:, idx_np].mean(1))
            idx_preds.append(preds)

        # numpyify
        np_preds.append(idx_preds)
    cmap = cm.rainbow(np.linspace(0, 1, num_tones))
    np_preds = np.array(np_preds)
    np_preds = np.squeeze(np_preds)
    #
    # for subj in range(n_subjs):
    #     for i in range(num_tones):
    #         axs[subj].scatter(i+1, np_preds[i, subj][55:65].mean(0), s=100,
    #         label=trained_on, c=c)
            # axs[subj].set_ylim([385, 395])
        # label='%s, %s' % (i+1, trained_on),
                # color=cmap[i])
    for i in range(num_tones):
        # plt.scatter([i+1]*n_subjs, np_preds[i][:, 50:70].mean(1))
        plt.scatter(i+1, np_preds[i].mean(0)[60], c='k',
                    s=100)
        # plt.ylim([385, 395])
plt.legend()
plt.show()
