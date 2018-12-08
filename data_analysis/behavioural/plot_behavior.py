# leg5@nyu.edu
# make psychometric curve for shepard pre-tests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# params
ptps = ['R1460', 'laura', 'ellie']
task = 'pitchdiscrim'
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# paths
data_path = '/Users/lauragwilliams/Documents/experiments/shepard/pilot_pretest'

dfs = list()
for ptp in ptps:

	# load data
	pitch_df = pd.read_csv('%s/%s_%s.txt' % (data_path, ptp, task), sep='\t')

	# change names of columns so i don't get confused
	pitch_df['response'] = pitch_df['same_diff']
	pitch_df['same_diff'] = (pitch_df['response'] != 0 )*1
	pitch_df['y_true'] = (pitch_df['freq1'] != pitch_df['freq2'])*1
	pitch_df['accuracy'] = (pitch_df['y_true'] == pitch_df['same_diff'])*1
	pitch_df['up_down'] = (pitch_df['freq1'] < pitch_df['freq2'])*1
	pitch_df['up_down'][pitch_df['up_down'].values==0] = -1
	pitch_df['direction_distance'] = pitch_df['distance']*pitch_df['up_down']
	pitch_df['order'] = (pitch_df['abs_dist']*pitch_df['up_down'] < 0)*1
	pitch_df['subject'] = [ptp]*len(pitch_df)

	# add to list
	dfs.append(pitch_df)
df = pd.concat(dfs)


# plot
data = df.groupby(['direction_distance', 'subject']).mean()['same_diff'].unstack()
data.plot(ax=axs[0], lw=4, colormap=plt.cm.Paired, legend=False)
axs[0].set_title('Overall performance')
axs[0].set_ylabel('Proportion "different" responses')
axs[0].set_xlabel('Distance from semitone (cents)')

for direction in [-1, 1]:
	data = df.query("up_down == @direction and distance != 0 and response != 0").groupby(['direction_distance', 'subject']).mean()['response'].unstack() - 1
	data.plot(ax=axs[1], lw=4, colormap=plt.cm.Paired, legend=False)
data = df.query("distance == 0 and response != 0").groupby(['direction_distance', 'subject']).mean()['response'].unstack() - 1
data.plot(ax=axs[1], lw=4, colormap=plt.cm.Paired, marker='o')
axs[1].set_title('Distribution of errors')
axs[1].set_ylabel('Proportion "up" responses')
axs[1].set_xlabel('Distance from semitone (cents)')

plt.show()



# # load data
# amp_df = pd.read_csv('%s/test_ampdiscrim.txt' % (data_path), sep='\t')
# fig, ax = plt.subplots(figsize=(15,7))
# amp_df.groupby(['amp_level', 'freq']).mean()['tone_present'].unstack().plot(ax=ax)
# plt.show()