# leg5@nyu.edu
# make psychometric curve for shepard pre-tests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# paths
data_path = '/Users/lauragwilliams/Documents/experiments/shepard/pilot_pretest'
files = glob.glob('%s/_pitchdiscrim/*' % (data_path))

# plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cols = plt.cm.rainbow(np.linspace(0.2, 1, len(files)))

accuracy_dict = {}
dfs = list()
for f in files:

	# get ptp code
	ptp = f.split('scrim/')[1].split('_')[0]

	# load data
	pitch_df = pd.read_csv(f, sep='\t')
	pitch_df['same_diff'] = (pitch_df['resp_button'].values != 0 )*1
	pitch_df['up_resp'] = (pitch_df['resp_button'].values == 2 )*1
	pitch_df['direction'] = pitch_df['up_down']
	pitch_df['direction'][pitch_df['up_down'].values==2] = -1
	pitch_df['direction_distance'] = pitch_df['abs_dist']*pitch_df['direction']
	pitch_df['subject'] = [ptp]*len(pitch_df)

	# add accuracy measures of different precisions
	pitch_df['same_diff_acc'] = ((pitch_df['up_down']==0)*1==(pitch_df['resp_button']==0)*1)*1

	# add accuracy info to the dictionary
	full_accuracy = pitch_df['accuracy'].values.mean()
	same_diff_accuracy = pitch_df['same_diff_acc'].values.mean()
	accuracy_dict.update({ptp: [full_accuracy, same_diff_accuracy]})

	# to plot just a subset of subjects, based on accuracy
	# if full_accuracy < 0.6:
	# 	continue

	# add to list
	dfs.append(pitch_df)
df = pd.concat(dfs)

# plot proportion of "different" responses as a function of cent difference
data = df.groupby(['direction_distance', 'subject']).mean()['same_diff'].unstack()
data.plot(ax=axs[0], lw=4, colormap=plt.cm.rainbow, legend=False)
axs[0].set_title('Overall performance')
axs[0].set_ylabel('Proportion "different" responses')
axs[0].set_xlabel('Distance between tones (cents)')
axs[0].set_ylim([0, 1])

# and RT
data = df.groupby(['direction_distance', 'subject']).mean()['RT'].unstack()
data.plot(ax=axs[1], lw=4, colormap=plt.cm.rainbow, legend=True)
axs[1].set_title('Reaction Time')
axs[1].set_ylabel('Reaction Time (ms)')
axs[1].set_xlabel('Distance between tones (cents)')
# axs[0].set_ylim([0, 1])
plt.show()



# # load data
# amp_df = pd.read_csv('%s/test_ampdiscrim.txt' % (data_path), sep='\t')
# fig, ax = plt.subplots(figsize=(15,7))
# amp_df.groupby(['amp_level', 'freq']).mean()['tone_present'].unstack().plot(ax=ax)
# plt.show()