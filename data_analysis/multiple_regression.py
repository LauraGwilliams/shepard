
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols

# VIF
# https://etav.github.io/python/vif_factor_python.html

explanatory_vars = ['Active_Engagement', 'Emotions', 'General_Sophistication',
                    'Musical_Training', 'Perceptual_Abilities',
                    'Singing_Abilities', 'same_diff','up_down','PLV']

# take out PLV
explanatory_vars.remove('PLV')

# gather features
features = "+".join(explanatory_vars)

# https://www.statsmodels.org/devel/gettingstarted.html
# get y and X dataframes based on this regression (X is features, y is exogenous var)
y,X = dmatrices('PLV ~' + features,df,return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)

# greater than 5 = highly correlated --> get rid of general sophistication

#__________________________________________________________
# multiple regression
explanatory_vars = ['Active_Engagement', 'Emotions',
                    'Musical_Training', 'Perceptual_Abilities',
                    'Singing_Abilities', 'same_diff','up_down']
pred_var = ['High_Low']


# extract data
X = df[explanatory_vars]
y = df[pred_var]

# fit the multiple regression
regr = make_pipeline(StandardScaler(),
                     LinearRegression())
regr.fit(X, y)

# get beta coeficients of the regression
betas = regr.steps[1][1].coef_

#__________________________________________________________
# multiple regression on decoding accuracy at each timepoint
explanatory_vars = ['Active_Engagement', 'Emotions',
                    'Musical_Training', 'Perceptual_Abilities',
                    'Singing_Abilities', 'same_diff','up_down','High_Low']

explanatory_vars = ['Musical_Training', 'Perceptual_Abilities',
                    'up_down','High_Low']


# explanatory_vars = ['PLV']

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'

regressor = 'condition'
subset= 'purepartial'
sensors = 'all'
scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))

scores = pd.DataFrame(scores)

# extract data
X = df[explanatory_vars]
y = scores

# fit the multiple regression
regr = make_pipeline(StandardScaler(),
                     LinearRegression())
regr.fit(X, y)

# get beta coeficients of the regression
betas = regr.steps[1][1].coef_

s = scores.mean(0)
times = np.linspace(-200,600,161)
for ii, var in enumerate(explanatory_vars):
    plt.plot(times, betas[:, ii], label=var)
plt.plot(times, s-s.mean(), c='k', lw=4)
plt.legend()
plt.show()

#__________________________________________________________
# multiple correlation
# https://www.statsmodels.org/devel/gettingstarted.html
scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'

explanatory_vars = ['Active_Engagement', 'Emotions',
                    'Musical_Training', 'Perceptual_Abilities',
                    'Singing_Abilities', 'same_diff',
                    'up_down','High_Low']

explanatory_vars = ['Perceptual_Abilities', 'same_diff', 'High_Low']

regressor = 'condition'
subset= 'purepartial'
sensors = 'all'
scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))

plt.plot(scores[df['High_Low'].values == 0, :].mean(0), label='low')
plt.plot(scores[df['High_Low'].values == 1, :].mean(0), label='high')
plt.legend()
plt.show()

# scores = pd.DataFrame(scores)

# time_crop = np.logical_and(times > 50, times <= 150)
# M100 = scores[:, time_crop].mean(1)
# M100 = scores[:, 51:59].mean(1)

m100_peak = 40 + np.argmax(scores[:, 40:80], axis=1)
M100 = np.diag(scores[:, m100_peak])
df['M100'] = M100

features = "+".join(explanatory_vars)


model = ols('M100 ~' + features, data=df).fit()
print model.params
print model.summary()

# plotting the condition diff
sensor_list = ['lh', 'rh', 'all']
regressor = 'condition'
subset= 'purepartial'

meas = 'up_down'
median = df[meas].median()

fig, axs = plt.subplots(1, 3, figsize=(30, 6))
for si, sensors in enumerate(sensor_list):

    scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))
    axs[si].plot(scores[df[meas].values < median, :].mean(0), label='low')
    axs[si].plot(scores[df[meas].values >= median, :].mean(0), label='high')
    # axs[si].plot(scores[df['High_Low'].values == 0, :].mean(0), label='low')
    # axs[si].plot(scores[df['High_Low'].values == 1, :].mean(0), label='high')
    axs[si].set_title(sensors)
axs[-1].legend()
plt.show()

# plot the difference between groups
# plotting the condition diff
sensor_list = ['lh', 'rh', 'all']
regressor = 'condition'
subset= 'purepartial'

meases = ['up_down', 'Musical_Training', 'High_Low', 'Perceptual_Abilities']

fig, axs = plt.subplots(3, len(meases), figsize=(15, 10))
for si, sensors in enumerate(sensor_list):
    scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))

    for mi, meas in enumerate(meases):
        median = df[meas].median()
        low = scores[df[meas].values < median, :].mean(0)
        high = scores[df[meas].values >= median, :].mean(0)
        axs[si, mi].plot(low, label='low')
        axs[si, mi].plot(high, label='high')
        axs[si, mi].set_title('%s, %s' % (sensors, meas))
axs[-1, -1].legend()
plt.show()

# tvalue plot over time

scores_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/group'

explanatory_vars = ['up_down', 'Musical_Training', 'High_Low', 'Perceptual_Abilities','Male_Female']
                    # 'Age','Singing_Abilities','Male_Female']
var_cols = {'up_down': 'Orange',
            'Musical_Training': 'y',
            'High_Low': 'g',
            'Perceptual_Abilities': 'Purple',
            'Male_Female': 'Grey'}
            # 'Age': 'b',
            # 'Singing_Abilities': 'Purple',



regressor = 'condition'
subset= 'purepartial'
sensor_list = ['lh','rh','all']
plots_dir = '/Users/ea84/Dropbox/shepard_decoding/_FIGS'

# params
features = "+".join(explanatory_vars)

n_features = len(explanatory_vars)+1
n_times = scores.shape[-1]

fig,axs = plt.subplots(1,3,figsize=(30,6))
for si, sensors in enumerate(sensor_list):

    scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))

    all_tvals = np.zeros([n_features, n_times])
    all_pvals = np.zeros([n_features, n_times])
    # fit the model at each time point
    for tt in range(n_times):

        # add this time point to the df
        df['tt_score'] = scores[:, tt]
        model = ols('tt_score ~' + features, data=df).fit()
        all_tvals[:, tt] = model.tvalues
        all_pvals[:, tt] = model.pvalues

    # plot those tvals
    for vi, var in enumerate(explanatory_vars):
        # plt.plot(times, (all_pvals[vi+1, :] < .05)*1, label=var)
        axs[si].plot(times, all_tvals[vi+1, :], label=var)
axs[-1].legend()
plt.show()

if subset == 'pure':
    cmap = plt.cm.Reds
elif subset == 'partial':
    cmap = plt.cm.Blues
elif subset == 'purepartial':
    cmap = plt.cm.rainbow

# [plt.plot(times, scores[ii, :], color=cmap(np.linspace(0, 1, 28)[ii])) for ii in range(28)]
plt.plot(times, scores.mean(0), lw=4, color='k')
plt.title('Decoding %s' % regressor)
# plt.savefig('%s/%s_%s-_scores.png' % (plots_dir, regressor, subset))
plt.close()

# find potential clusters
if regressor == 'condition':
    t_thresh = 1.2
    min_cluster_len = 5
    avg_acc_scaler = 100

elif regressor == 'freq':
    t_thresh = 1.5
    min_cluster_len = 5
    avg_acc_scaler = 60

# init cluster bin for all vars
# all_clusters = np.zeros([n_features, n_times])
all_clusters = list()

# loop through vars
for vi in range(n_features):

    # init bin for this var
    clusters = list()

    # init potential cluster list
    potential_cluster = list()

    # init counter
    counter = 0

    # loop through time
    for tii in range(n_times):

        # if tval exceeds threshold, add this timepoint to the cluster bin
        if np.abs(all_tvals[vi, tii]) > t_thresh:
            counter = counter + 1
            # mark as potenital cluster
            potential_cluster.append(tii)
        else:
            # once cluster ends, check the length
            if counter >= min_cluster_len:
                # if long enough, add to the bin
                # all_clusters[vi, np.array(potential_cluster)] = 1
                clusters.append(potential_cluster)
            counter = 0

            # reset potenital cluster bin
            potential_cluster = list()
    all_clusters.append(clusters)


sensors = 'all'
scores = np.load('%s/group_%s_%s_%s.npy'%(scores_dir,regressor,subset,sensors))

labels = ['Pitch Acuity','Musical Training','High/Low Synchronizer',
            'Perceptual Abilities','Male/Female']
#,'Age','Singing Abilities','Male/Female'
# params
plt.close()
fig, axs = plt.subplots(len(labels), 1, figsize=(10, 12))
# plot those tvals
for vi, var in enumerate(explanatory_vars):
    cluster = all_clusters[vi+1]
    axs[vi].plot(times, all_tvals[vi+1, :], label=labels[vi],
                 color=var_cols[var], lw=4, alpha=0)
    axs[vi].plot(times,(scores.mean(0)-scores[:, 0:40].mean())*avg_acc_scaler,
                 color='Grey', lw=2, linestyle='--')
    axs[vi].axhline(y=0,color='Black',linestyle='--')
    axs[vi].set_xlim([-0.1,0.4])
    for cluster_idxs in cluster:

        # only plot the cluster if it is between 0-400 ms
        if sum(np.logical_and(times[cluster_idxs] > 0,
                              times[cluster_idxs] < 300)) == len(cluster_idxs):

            axs[vi].fill_between(x=times[cluster_idxs],
                                 y1=0,
                                 y2=all_tvals[vi+1, :][cluster_idxs],
                                 color=var_cols[var],
                                 alpha=0)
    axs[vi].legend(loc='upper right')
plt.savefig('%s/%s_%s-indiv_diff_line_empty.png' % (plots_dir, regressor, subset))
plt.close()

# TODO - permutation cluster test

# # get null distribution
# n_perms = 20
# all_tvals = np.zeros([n_features, n_times, n_perms])
#
# # fit the model at each time point
# for pp in range(n_perms):
#     for tt in range(n_times):
#
#         # add this time point to the df
#         df['tt_score'] = scores[:, tt]
#         model = ols('tt_score ~' + features, data=df).fit()
#         all_tvals[:, tt, pp] = model.tvalues
#         print(pp, tt)
