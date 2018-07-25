import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from meg_preprocessing import (evoked, difference_evoked_A, difference_evoked_C,
                                difference_evoked_Eb, difference_evoked_hi_lo,
                                partials_evoked, pure_evoked, stc_evoked,
                                inverse_operator, inverse_operator_signed)

mri_dir='/Users/ellieabrams/Desktop/Projects/Shepard/analysis/mri/'
subject='R1201'

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

difference_evoked_hi_lo.plot_joint(title='diff_highest_lowest', times=[.075,.1,.125,.15,.17,.2],
                ts_args=ts_args, topomap_args=topomap_args)

labels = read_labels_from_annot(subject, parc='aparc', subjects_dir=mri_dir)

# HG and STC labels
hg_lh = [label for label in labels if label.name == 'transversetemporal-lh'][0]
hg_rh = [label for label in labels if label.name == 'transversetemporal-rh'][0]
stg_lh = [label for label in labels if label.name == 'superiortemporal-lh'][0]
stg_rg = [label for label in labels if label.name == 'superiortemporal-rh'][0]

src = inverse_operator['src']

# extract vertices within label per condition
stc_label = stc_evoked.in_label(hg_lh)
mean = stc_evoked.extract_label_time_course(hg_lh, src, mode='mean')
plt.figure()
plt.plot(1e3 * stc_label.times, stc_label.data.T, 'k', linewidth=0.5)
h0, = plt.plot(1e3 * stc_label.times, mean.T, 'r', linewidth=3)
plt.legend([h0], ['mean'])
plt.xlabel('Time (ms)')
plt.ylabel('Source amplitude')
plt.title('Activations in Label : %s' % label)
plt.show()
