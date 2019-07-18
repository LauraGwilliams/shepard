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

for trained_on in ['partial']:

    subset = ['scale','random']

    low_ypreds = list()
    high_ypreds = list()
    low_ypreds_d = list()
    high_ypreds_d = list()
    rand_ypreds = list()
    all_betas = list()
    for subject in subjects:
        print(subject)

        # load ypreds
        info_fname = '%s/%s/%s_freq_train%s_ypreds.csv' % (base_dir, subject, subject,
                                                      trained_on)
        trial_info = pd.read_csv(info_fname)
        # add another variable deinfine the first and last note postion
        trial_info['highest_lowest'] = np.logical_or(trial_info['note_position'] == 1,
                                                      trial_info['note_position'] == 8)*1

        ypred_fname = '%s/%s/%s_freq_train%s_ypreds.npy' % (base_dir, subject, subject,
                                                      trained_on)
        ypred = np.load(ypred_fname)

        # sanity
        assert(trial_info.shape[0] == ypred.shape[1])

        # dims
        n_times, n_trials = ypred.shape

        # params
        explanatory_vars = ['freq', 'updown', 'note_position']

        # subset just the scale trials
        # idx = np.logical_and(trial_info['circscale'].values == 'scale',
        #                      trial_info['highest_lowest'].values == 1)
        idx = np.logical_or(trial_info['circscale'].values == subset[0],
                            trial_info['circscale'].values == subset[1])
        trial_info = trial_info[idx]
        ypred = ypred[:, idx]

        conds_1_8 = ['down','down','up','up','random']
        pos = [1,8,1,8,1]

        # extract just the low and the high ypreds
        idx_1 = np.logical_and(trial_info['note_position'].values == pos[0],
                             trial_info['updown'].values == conds_1_8[0])
        low_ypreds.append(ypred[:, idx_1].mean(1))
        idx_8 = np.logical_and(trial_info['note_position'].values ==pos[1],
                             trial_info['updown'].values == conds_1_8[1])
        high_ypreds.append(ypred[:, idx_8].mean(1))
        idx_1_d = np.logical_and(trial_info['note_position'].values == pos[2],
                             trial_info['updown'].values == conds_1_8[2])
        low_ypreds_d.append(ypred[:, idx_1_d].mean(1))
        idx_8_d = np.logical_and(trial_info['note_position'].values ==pos[3],
                             trial_info['updown'].values == conds_1_8[3])
        high_ypreds_d.append(ypred[:, idx_8_d].mean(1))
        idx_1_rand = np.logical_and(trial_info['note_position'].values == pos[4],
                             trial_info['updown'].values == conds_1_8[4])
        rand_ypreds.append(ypred[:, idx_1_rand].mean(1))
        # low_ypreds.append(ypred[:, trial_info['note_position'].values == 1].mean(1))
        # high_ypreds.append(ypred[:, trial_info['note_position'].values == 8].mean(1))

    #     # extract data
    #     trial_info['updown'] = trial_info['updown'] == 'down'*1
    #     X = trial_info[explanatory_vars].values * 1
    #     X = np.array(X, dtype=float)
    #
    #     # define the multiple regression
    #     regr = make_pipeline(StandardScaler(),
    #                          LinearRegression())
    #
    #     betas = list()
    #
    #     # loop over time
    #     for tt in range(n_times):
    #         y = ypred[tt, :]
    #
    #         # fit the model
    #         regr.fit(X, y)
    #
    #         # get beta coeficients of the regression
    #         betas.append(regr.steps[1][1].coef_)
    #     betas = np.array(betas)
    #     all_betas.append(betas)
    # all_betas = np.array(all_betas)

    # numpyify
    low_ypreds = np.array(low_ypreds)
    high_ypreds = np.array(high_ypreds)
    low_ypreds_d = np.array(low_ypreds_d)
    high_ypreds_d = np.array(high_ypreds_d)
    rand_ypreds = np.array(rand_ypreds)
    np_preds = np.array(np_preds)

    plt.plot(times,low_ypreds.mean(0), 'r', label='%s, %s' % (conds_1_8[0],pos[0]))
    plt.plot(times,high_ypreds.mean(0), 'b', label='%s, %s' % (conds_1_8[1],pos[1]))
    plt.plot(times,low_ypreds_d.mean(0), 'g', label='%s, %s' % (conds_1_8[2],pos[2]))
    plt.plot(times,high_ypreds_d.mean(0), 'm', label='%s, %s' % (conds_1_8[3],pos[3]))
    plt.plot(times,rand_ypreds.mean(0), 'k', label='%s, %s' % (conds_1_8[4],pos[4]))
    plt.title('y_preds for ambiguous tone in %s condition trained on %s'%(subset,
                                                                        trained_on))

plt.legend()
plt.show()

# for ii, var in enumerate(explanatory_vars):
#     plt.plot(all_betas.mean(0)[:, ii], label=var)
# plt.legend()
# plt.show()



# plot all notes y_preds

for trained_on in ['purepartial']:
    condition = 'scale'
    dir = 'up'

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
            info_fname = '%s/%s/%s_freq_train%s_ypreds.csv' % (base_dir, subject, subject,
                                                          trained_on)
            trial_info = pd.read_csv(info_fname)

            ypred_fname = '%s/%s/%s_freq_train%s_ypreds.npy' % (base_dir, subject, subject,
                                                          trained_on)
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
    for i in range(num_tones):
        plt.plot(times,np_preds[i].mean(0).mean(0),label='%s, %s' % (i+1, trained_on),
                color=cmap[i])

plt.legend()
plt.show()

#__________________________________________________________

for trained_on in ['partial']:

    subset = 'random'

    low_ypreds = list()
    high_ypreds = list()
    low_ypreds_d = list()
    high_ypreds_d = list()
    all_betas = list()
    for subject in subjects:
        print(subject)

        # load ypreds
        info_fname = '%s/%s/%s_freq_train%s_ypreds.csv' % (base_dir, subject, subject,
                                                      trained_on)
        trial_info = pd.read_csv(info_fname)
        # add another variable deinfine the first and last note postion
        trial_info['highest_lowest'] = np.logical_or(trial_info['note_position'] == 1,
                                                      trial_info['note_position'] == 8)*1

        ypred_fname = '%s/%s/%s_freq_train%s_ypreds.npy' % (base_dir, subject, subject,
                                                      trained_on)
        ypred = np.load(ypred_fname)

        # sanity
        assert(trial_info.shape[0] == ypred.shape[1])

        # dims
        n_times, n_trials = ypred.shape

        # params
        explanatory_vars = ['freq', 'updown', 'note_position']

        # subset just the scale trials
        idx = trial_info['circscale'].values == subset
        trial_info = trial_info[idx]
        ypred = ypred[:, idx]

        conds_1_8 = ['up','down']
        pos = [1,1]

        # extract just the low and the high ypreds
        idx_1 = np.logical_and(trial_info['note_position'].values == pos[0],
                             trial_info['updown'].values == conds_1_8[0])
        low_ypreds.append(ypred[:, idx_1].mean(1))
        idx_8 = np.logical_and(trial_info['note_position'].values ==pos[1],
                             trial_info['updown'].values == conds_1_8[1])
        high_ypreds.append(ypred[:, idx_8].mean(1))

    # numpyify
    low_ypreds = np.array(low_ypreds)
    high_ypreds = np.array(high_ypreds)
    np_preds = np.array(np_preds)

    plt.plot(times,low_ypreds.mean(0), 'r', label='%s, %s' % (conds_1_8[0],pos[0]))
    plt.plot(times,high_ypreds.mean(0), 'b', label='%s, %s' % (conds_1_8[1],pos[1]))
    plt.title('y_preds for ambiguous tone in %s condition trained on %s'%(subset,
                                                                        trained_on))

plt.legend()
plt.show()
