# leg5@nyu.edu
# run decoding analysis on shepard data

# packages
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne import read_epochs, read_evokeds
#from jr import scorer_spearman
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, cross_val_score, cross_val_predict,
                                    StratifiedKFold)
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, get_coef,
                        LinearModel, cross_val_multiscore)
import mne.decoding
from jr import scorer_spearman
from sklearn.metrics import make_scorer, get_scorer

subj = 'R1460'
meg_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/' +subj+ '/'

# epochs collapsed across pure and partials
allepochs = meg_dir + subj + '_shepard-epo.fif'
allepochs_info = meg_dir + subj + '_shepard_trialinfo.csv'

# keys collapsed across pure and partials
keyA = meg_dir + subj + '_A_purepar-epo.fif'
keyC = meg_dir + subj + '_C_purepar-epo.fif'
keyEb = meg_dir + subj + '_Eb_purepar-epo.fif'
keyA_info = meg_dir + subj + '_A_purepar_trialinfo.csv'
keyC_info = meg_dir + subj + '_C_purepar_trialinfo.csv'
keyEb_info = meg_dir + subj + '_Eb_purepar_trialinfo.csv'

# shepard epochs
shep = meg_dir + subj + '_shep-epo.fif'
shep_info = meg_dir + subj + '_shep_trialinfo.csv'

# pure epochs
pure = meg_dir + subj + '_pure-epo.fif'
pure_info = meg_dir + subj + '_pure_trialinfo.csv'

# partials epochs
partials = meg_dir + subj + '_par-epo.fif'
partials_info = meg_dir + subj + '_par_trialinfo.csv'

# random blocks
random = meg_dir + subj + '_random-epo.fif'
random_info = meg_dir + subj + '_random_trialinfo.csv'

# scale (up or down) blocks
scale = meg_dir + subj + '_scale-epo.fif'
scale_info = meg_dir + subj + '_scale_trialinfo.csv'

# scale broken down by key
keyAscale = meg_dir + subj + '_A_scale-epo.fif'
keyCscale = meg_dir + subj + '_C_scale-epo.fif'
keyEbscale = meg_dir + subj + '_Eb_scale-epo.fif'
keyAscale_info = meg_dir + subj + '_A_scale_trialinfo.csv'
keyCscale_info = meg_dir + subj + '_C_scale_trialinfo.csv'
keyEbscale_info = meg_dir + subj + '_Eb_scale_trialinfo.csv'

#-------------------------------------------------------------------------------

# params: choose subject, regressor, epochs/info, how to decode
subject = 'R1460'
regressor = 'condition' #condition or frequency
current_epochs = shep
current_info = shep_info
decode_using = 'spatial'
subset_trials = False
# spatial (trials x sensors x time), temporal (trials x time x sensors),
# combined (trials x sensors*time)

# load data
epochs = mne.read_epochs(current_epochs)
X = epochs._data[:, 0:157, :] # just meg channels

# change X if applicable
if decode_using == 'temporal':
    # switch channels and times to decode using time course
    X = np.transpose(X, [0, 2, 1])
if decode_using == 'combined':
    # collapse over spatial and temporal dimensions
    [n_trials, n_sns, n_times] = X.shape
    X = np.reshape(X, [n_trials, n_sns*n_times])

# load trial info
trial_info = pd.read_csv(current_info)

# # add another column specifying whether the freq of current trial
# # is higher or lower than the preceeding trial.
# current_trials = trial_info['freq'].values
# preceding_trial = np.roll(current_trials, 1)
# low_preceeding = (current_trials > preceding_trial)*1
# trial_info['low_preceeding'] = low_preceeding
# # first entry not valid because there is no preceding trial by definition
# trial_info['low_preceeding'][0] = np.nan
# trial_info['updown_switch'] = np.array([''.join([str(ii), str(jj)]) for ii, jj in trial_info[['low_preceeding', 'updown']].values])
# trial_info['purepar_updown'] = np.array([''.join([str(ii), str(jj)]) for ii, jj in trial_info[['condition', 'updown']].values])
# trial_info['freq_key'] = np.array([''.join([str(ii), str(jj)]) for ii, jj in trial_info[['key', 'freq']].values])

# add a column to mark how many notes into the scale the trial is

# pull out regressor of interest
y = trial_info[regressor].values

# sanity
assert(len(X) == len(y))

# to asses decoding performance on a reduced number of trials
if subset_trials == True:
    y = y[0:2160]
    X = X[0:2160, ...]

def my_scaler(x):
    '''
    Scale btw 0-1.
    '''
    x = np.array(map(float, x))
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def binary_scaler(x):
    '''
    Pure = 1, partial = 0.
    '''
    x = np.array(x)
    x[x=='pure']=1.
    x[x=='partial']=0.
    x[x=='shepard']=2.
    x = np.array(map(float, x))
    return x

# def add_tone_properties(epochs, trial_info):
#
#     # note position dictionary
#     note_pos_dict = {'A220': 1,
#                      'A247': 2,
#                      'A277': 3,
#                      'A294': 4,
#                      'A330': 5,
#                      'A370': 6,
#                      'A415': 7,
#                      'A440': 8,
#                      'C262': 1,
#                      'C294': 2,
#                      'C330': 3,
#                      'C349': 4,
#                      'C392': 5,
#                      'C440': 6,
#                      'C494': 7,
#                      'C524': 8,
#                      'Eb312': 1,
#                      'Eb349': 2,
#                      'Eb392': 3,
#                      'Eb415': 4,
#                      'Eb466': 5,
#                      'Eb524': 6,
#                      'Eb587': 7,
#                      'Eb624': 8}

    # add note position to the df
    # trial_info['note_position'] = np.array([note_pos_dict[k] for k in trial_info['freq_key']])
    # trial_info['time_ms'] = epochs.events[:, 0]
    # trial_info['ISI'] = trial_info['time_ms'] - np.roll(trial_info['time_ms'], 1)
    #
    # # add tone position
    # position = 0
    # trial_positions = list()
    # for isi in trial_info['ISI'].values:
    #     if isi < 750:
    #         trial_positions.append(position)
    #         position += 1
    #     else:
    #         trial_positions.append(0)
    #         position = 1
    # trial_info['trial_position'] = np.array(trial_positions)
    # return trial_info


#----------------------------------------------------------------------------

# NOTE: it is important to make sure that the y array is in float
# format not integer, otherwise it turns it into a binary problem.
# Also, it doesn't make sense to use statified KFold for regression problem.
# because there are no classes to speak of. KFold instead. My bad.

# set up decoder, use logistic for categorical and Ridge for continuous
if regressor == 'freq':
    y = my_scaler(y)
    clf = make_pipeline(StandardScaler(), Ridge())
    scorer = make_scorer(get_scorer(scorer_spearman))
    score = 'Spearman R'
    cv = KFold(5)
if regressor == 'condition':
    y = binary_scaler(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    scorer = 'roc_auc'
    score = 'AUC'
    cv = StratifiedKFold(5)

n_jobs = -1

# set up estimator, get scores
if decode_using == 'spatial':
    gen = GeneralizingEstimator(n_jobs=n_jobs,
                            scoring=scorer,
                            base_estimator=clf)
    scores = cross_val_multiscore(gen, X, y,
                                  cv=cv)
elif decode_using == 'temporal':
    gen = SlidingEstimator(n_jobs=n_jobs,
                        scoring=scorer,
                        base_estimator=clf)
    scores = cross_val_multiscore(gen, X, y,
                                cv=cv)
else:
    # scoring defaults to neg mean squared so set to scorer
    # shuffle must be true when binary values, otherwise fold will only have
        # one value
    scores = cross_val_score(clf, X, y,
                            scoring=scorer, #defaults to neg mean squared
                            cv=KFold(5, shuffle=True))


# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# -------
# train classifier on all trials to get a prediction for each trial.
# then, evaluate accuracy on a subset of trials.
# gen = SlidingEstimator(n_jobs=n_jobs,
#                     scoring=scorer,
#                     base_estimator=clf)
# y_pred = cross_val_predict(gen, X, y,
#                             cv=cv, method='predict')
#
# # for example, evaluate performance of decoding frequency separately
# # for the pure and partial tones
# scores = list()
# only_look_at = 'nan'
# #only_look_at = "key == 'A'"
# subset_using = 'key'  # which column to use to subset
# subset_values = trial_info[subset_using].values
# search_terms = np.unique(subset_values)
# # loop through each level of the factor we are using to subset
# for search_term in search_terms:
#
#     # skip if nan
#     if 'nan' in str(search_term):
#         continue
#
#     if str(search_term) == '0.0up' or str(search_term) == '1.0down':
#         continue
#
#     # init the scores for this level
#     cond_scores = list()
#
#     # find the indices that correspond to this level
#     subset_idx = np.where(subset_values == search_term)[0]
#
#     if str(only_look_at) != 'nan':
#         subset_idx = np.intersect1d(trial_info.query(only_look_at).index,
#                                     subset_idx)
#
#     print(y_pred[subset_idx, 0], subset_values[subset_idx], search_term)
#
#     # loop through time
#     for t in range(y_pred.shape[-1]):
#         # evaluate at each time point, for this level
#         cond_scores.append(scorer_spearman(y[subset_idx], y_pred[subset_idx, t]))
#
#     cond_scores = np.array(cond_scores)
#     plt.plot(epochs.times, cond_scores, label=search_term)
#
#     # add the performance for this time-point to the list
#     scores.append(np.array(cond_scores))
#
# # and finally, add just the overall accuracy when evaluated on all trials
# # loop through time
# avg_scores = list()
# for t in range(y_pred.shape[-1]):
#     # evaluate at each time point
#     avg_scores.append(scorer_spearman(y, y_pred[:, t]))
# plt.plot(epochs.times, np.array(avg_scores), label='all trials', lw=2, c='k')
# plt.legend()
# plt.show()
#
# # plot result
# lineObjects = plt.plot(epochs.times, scores.T)
# plt.legend(iter(lineObjects), search_terms)
# plt.show()


# ---------------------------------------------------------------------------
# PLOTTING

if decode_using == 'spatial':
    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(epochs.times, np.diag(scores), label='score')
    if score == 'AUC':
        ax.axhline(.5, color='k', linestyle='--', label='chance')
    else:
        ax.axhline(.0, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('%s'%(score))
    ax.legend()
    # ax.set_ylim(bottom=-0.035, top=0.16)
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    plt.show()
elif decode_using == 'temporal':
    scores_evoked = mne.read_evokeds(meg_dir+subj+'_shepard-evoked-ave.fif')[0]
    scores_evoked._data[0:157, 0] = scores
    if regressor == 'freq':
        scores_evoked.plot_topomap(times=scores_evoked.times[0])
    if regressor == 'condition':
        # center around 0 for plotting
        scores_evoked._data[0:157, 0] = scores_evoked._data[0:157, 0] - scores_evoked._data[0:157, 0].mean()
        scores_evoked.plot_topomap(times=scores_evoked.times[0])
else:
    print scores

# to plot matrix
plt.matshow(plt.matshow(scores, cmap=plt.cm.RdBu_r, origin='lower'))
plt.show()
