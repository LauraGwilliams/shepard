# leg5@nyu.edu
# apply MDS to shepard tones

# modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import glob
from scipy.io.wavfile import read


# paths
base_dir = '/Users/lauragwilliams/Documents/experiments/shepard/shepard/'
tone_dir = '%s/stimuli/main_exp/db_normed_tones' % (base_dir)
out_dir = '%s/plots' % (base_dir)

# params
conditions = ['partial', 'pure', 'shepard']
key = 'A'
cols = [plt.cm.Blues(np.linspace(0.2, 1, 12)),
		plt.cm.Oranges(np.linspace(0.2, 1, 12)),
		plt.cm.Greens(np.linspace(0.2, 1, 12))]

# load data
ffts = list()
freqs = list()
tcs = list()
for cond in conditions:

	# init
	cond_fft = list()
	cond_freq = list()
	cond_tc = list()

	# get all files for this condition and key
	search_term = '%s/%s-%s_*-min-60_max-20.wav' % (tone_dir, cond, key)
	fnames = glob.glob(search_term)

	# load data in turn
	for fii, fname in enumerate(fnames):
		fs, t = read(fname)

		# add the freq of this tone to the list
		tone_freq = fname.split('%s_' % (key))[1].split('-min')[0]
		if tone_freq == '440':
			continue
		cond_freq.append(tone_freq)

		# get fft
		fft_data = np.abs(np.fft.fft(t))[0:1000]
		cond_fft.append(fft_data)
		cond_tc.append(t)

	# add to total list
	ffts.append(np.array(cond_fft))
	freqs.append(np.array(cond_freq))
	tcs.append(np.array(cond_tc))

freqs = np.array(freqs)
ffts = np.array(ffts).astype(float)
tcs = np.array(tcs)

# normalise fft
# ffts = np.array(ffts < ffts.mean())*1
norm_fft = np.zeros_like(ffts)

# plot fft
for cii, cond in enumerate(conditions):
	for fii, f in enumerate(np.sort(freqs[cii])):
		idx = np.where(freqs[cii] == f)[0][0]
		print(f, idx)
		plt.plot(ffts[cii, idx, :], label=f,
				 c=cols[cii][fii])
		plt.hlines(ffts[cii, idx, :].mean()*50, 0, 800)
		norm_fft[cii, idx, :] = (ffts[cii, idx, :] > ffts[cii, idx, :].mean()*50)
	plt.legend()
	plt.show()

# scale fft
for c in range(3):
	for t in range(12):
		norm_fft[c, t, :] = norm_fft[c, t, :] * w

for cii, cond in enumerate(conditions):
	for fii, f in enumerate(np.sort(freqs[cii])):
		idx = np.where(freqs[cii] == f)[0][0]
		print(f, idx)
		plt.plot(norm_fft[cii, idx, :], label=f,
				 c=cols[cii][fii])
		plt.hlines(norm_fft[cii, idx, :].mean()*50, 0, 800)
	plt.legend()
	plt.show()

# reshape the data
data = norm_fft
fft_reshaped = np.reshape(data, [data.shape[0]*data.shape[1], data.shape[2]])

# fit MDS
trans_data = MDS().fit_transform(fft_reshaped)

# shape back the data
trans_data = np.reshape(trans_data, [data.shape[0], data.shape[1], 2])

# plot responses as a scatter figure
for cii, cond in enumerate(conditions):
	for fii, f in enumerate(np.sort(freqs[cii])):
		print(f)
		idx = np.where(freqs[cii] == f)[0][0]
		plt.scatter(trans_data[cii, idx, 0], trans_data[cii, idx, 1], 
					label='%s%s' % (f, cond), c=cols[cii][fii], s=120, alpha=0.8)
	plt.legend()
	plt.show()

# fit MDS on just one condition
cond_i = 1
trans_data = MDS().fit_transform(data[cond_i])

# plot responses as a scatter figure
for fii, f in enumerate(np.sort(freqs[cond_i])):
	print(f)
	idx = np.where(freqs[cii] == f)[0][0]
	plt.scatter(trans_data[idx, 0], trans_data[idx, 1], 
				label='%s%s' % (f, conditions[cond_i]), c=cols[cond_i][fii], s=120, alpha=0.8)
plt.legend()
plt.show()
