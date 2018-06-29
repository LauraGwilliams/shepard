# leg5@nyu.edu
# parse logfiles for shepard experiment

import pandas as pd
import numpy as np
import csv

# paths
data_path = '/Users/lauragwilliams/Documents/experiments/shepard/pilot'

# params
conditions = ['shepard_0', 'pure_0', 'partials_0', 'shepard_1']

# funcs
def txt_to_pandas(fname):
	with open(fname) as inputfile:
	    results = [l[0].split('\t') for l in list(csv.reader(inputfile))]
	return pd.DataFrame(results[1:], columns=results[0])


dfs = list()
for condition in conditions:
	fname = '%s/R1201_%s.txt' % (data_path, condition)
	dfs.append(txt_to_pandas(fname))
df = pd.concat(dfs)