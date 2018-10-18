import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import mne
from meg_preprocessing import (evoked,  stc_evoked,
                                inverse_operator, inverse_operator_signed)
from mne import read_labels_from_annot
# difference_evoked_A, difference_evoked_C, difference_evoked_Eb, difference_evoked_hi_lo,
                                # partials_evoked, pure_evoked,

# paths
mri_dir='/Users/ellieabrams/Desktop/Projects/Shepard/analysis/mri/'
stc_dir = '/Users/ellieabrams/Desktop/Projects/Shepard/analysis/meg/stc/'
subject='R1201'


# evoked topomaps
ts_args = dict(gfp=True)
topomap_args = dict(sensors=False)
evoked.plot_joint(title='evoked', times=[.075,.125,.2], ts_args=ts_args,
                topomap_args=topomap_args)

difference_evoked_A.plot_joint(title='diff_A', times=[.075,.1,.125,.15,.17,.2],
                ts_args=ts_args, topomap_args=topomap_args)

difference_evoked_C.plot_joint(title='diff_C', times=[.075,.1,.125,.15,.17,.2],
                ts_args=ts_args, topomap_args=topomap_args)

difference_evoked_Eb.plot_joint(title='diff_Eb', times=[.04, .05, .075,.125],
                ts_args=ts_args, topomap_args=topomap_args)

difference_evoked_hi_lo.plot_joint(title='diff_highest_lowest',
                times=[.075,.1,.125,.15,.17,.2],ts_args=ts_args,
                topomap_args=topomap_args)

-------------------------------------------------------------------------------
# SET UP: labels, stc lists, means to plot
# extract vertices within label per condition

labels = read_labels_from_annot(subject, parc='aparc', subjects_dir=mri_dir)

# HG and STC labels
hg_lh = [label for label in labels if label.name == 'transversetemporal-lh'][0]
hg_rh = [label for label in labels if label.name == 'transversetemporal-rh'][0]
stg_lh = [label for label in labels if label.name == 'superiortemporal-lh'][0]
stg_rh = [label for label in labels if label.name == 'superiortemporal-rh'][0]

src = inverse_operator['src']

# SETS
freqs = ['156', '175', '196', '208', '220', '233', '247', '262', '277', '294',
'312', '330', '349', '370', '392', '415', '440', '494', '524']

A_tones = ['A_220', 'A_247', 'A_277', 'A_294', 'A_330', 'A_370', 'A_415', 'A_440']
C_tones = ['C_262', 'C_294', 'C_330', 'C_349', 'C_392', 'C_440', 'C_494', 'C_524']
Eb_tones = ['D_156', 'D_175', 'D_196', 'D_208', 'D_233', 'D_262', 'D_294', 'D_312']

# NOTE change this!
current_label = stg_lh

# NOTE change this!
current_set = A_tones
stc_list = []
stc_labels = []
stc_means = []

# NOTE for entire list of frequencies (collapsed over scale)
# for i in range(len(freqs)):
#     #temp_stc = mne.read_source_estimate(stc_dir + '%sHz/R1201_shepard_%s'%(freqs[i],freqs[i]))
#     temp_stc = mne.read_source_estimate(stc_dir + '%sHzPure/R1201_shepard_%spure'%(freqs[i],freqs[i]))
#     #temp_stc = mne.read_source_estimate(stc_dir + '%sHzPartials/R1201_shepard_%spar'%(freqs[i],freqs[i]))
#     stc_list.append(temp_stc)
#     stc_labels.append(temp_stc.in_label(current_label))
#     stc_means.append(temp_stc.extract_label_time_course(current_label, src, mode='mean'))

# NOTE for frequencies within scales
for i in range(8):
    #temp_stc = mne.read_source_estimate(stc_dir + '%sHz/R1201_shepard_%s'%(A_tones[i],A_tones[i]))
    #temp_stc = mne.read_source_estimate(stc_dir + '%sHz/R1201_shepard_%s'%(C_tones[i],C_tones[i]))
    temp_stc = mne.read_source_estimate(stc_dir + '%sHz/R1201_shepard_%s'%(Eb_tones[i],Eb_tones[i]))
    stc_list.append(temp_stc)
    stc_labels.append(temp_stc.in_label(current_label))
    stc_means.append(temp_stc.extract_label_time_course(current_label, src, mode='mean'))

#-------------------------------------------------------------------------------
# PLOT THAT SHIT!

# NOTE for entire evoked stc all trials
# current_stc = stc_evoked
# stc_label = current_stc.in_label(current_label)
# mean = current_stc.extract_label_time_course(current_label, src, mode='mean')
# h0 = plt.plot(1e3 * stc_label.times, mean.T, 'r', linewidth=2)

# create color lists
color=iter(cm.rainbow(np.linspace(0,1,8))) #for each scale
#color=iter(cm.rainbow(np.linspace(0,1,19))) #for all freqs
plt.figure()
for j in range(len(stc_means)):
    c=next(color)
    freq_plots.append(plt.plot(1e3 * stc_labels[j].times, stc_means[j].T, c=c, label=freqs[j],linewidth=1,))
plt.legend(current_set) # NOTE SET FOR CURRENT SET
plt.xlabel('Time (ms)')
plt.ylabel('Source amplitude')
plt.title('Activations in Label : ' + current_label.name)
plt.show()
