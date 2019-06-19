# leg5@nyu.edu & ea84@nyu.edu
# parse logfiles for shepard experiment

import pandas as pd
import numpy as np
import csv
import mne
import glob

# paths
subject = 'P015'
base_path = '/Users/ea84/Dropbox/shepard_preproc/%s'%(subject) # change to local meg dir
return_path = '/Users/ea84/Dropbox/shepard_decoding/%s'%(subject)

# use time stamps to create condition order
blocks = ['shepard', 'pure', 'partials']

# grab time stamps, order list
time_stamps = list()
for block in blocks:
	time_stamp = ([int(i.split('_%s_' %(block))[1].split('.txt')[0])
					for i in glob.glob('%s/*%s*[0-9].txt' % (base_path, block))])
	time_stamps.append(time_stamp)
time_stamps = np.array(time_stamps).flatten().tolist()
time_stamps.sort()

# func
def txt_to_pandas(fname):
	with open(fname) as inputfile:
	    results = [l[0].split('\t') for l in list(csv.reader(inputfile))]
	return pd.DataFrame(results[1:], columns=results[0])

dfs = list()
for time_stamp in time_stamps:
	fname = glob.glob('%s/*%s.txt' % (base_path, time_stamp))
	assert(len(fname) == 1)
	fname = fname[0]  # glob gives a list of files
	dfs.append(txt_to_pandas(fname))
	print(fname)

df = pd.concat(dfs)

df.to_csv('%s/'%(return_path) + '%s_shepard_trialinfo.csv'%(subject))
