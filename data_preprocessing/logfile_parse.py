# leg5@nyu.edu & ea84@nyu.edu
# parse logfiles for shepard experiment
# concatenate fifs

import pandas as pd
import numpy as np
import csv
import mne

# params
conditions = ['pure_0', 'partials_0', 'pure_1', 'partials_1', 'pure_2', 'partials_2']
subj = 'A0316'

# paths
data_path = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/' + subj

fifs = []

# funcs
def txt_to_pandas(fname):
	with open(fname) as inputfile:
	    results = [l[0].split('\t') for l in list(csv.reader(inputfile))]
	return pd.DataFrame(results[1:], columns=results[0])

dfs = list()
for condition in conditions:
	fname = '%s/'%(data_path)+subj+'_%s.txt' %(condition)
	dfs.append(txt_to_pandas(fname))
	# fif = mne.io.read_raw_fif('%s/'%(data_path) + subj + '_%s-raw.fif'%(condition))
	# fifs.append(fif)
	# print fifs

# all_fifs = mne.concatenate_raws(fifs)
df = pd.concat(dfs)

# df.to_csv('%s/'%(data_path) + subj + '_purepar_trialinfo.csv')
# all_fifs.save('%s/'%(data_path) + subj + '_shepard-raw.fif', overwrite=True)
