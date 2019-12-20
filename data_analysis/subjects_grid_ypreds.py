import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


# paths
base_dir = '/Users/ea84/Dropbox/shepard_decoding/_GRP_SCORES/n=28/indiv/ypreds'
info_dir = '/Users/ea84/Dropbox/shepard_decoding/_DOCS'

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

info = pd.read_csv('%s/MSI_breakdown.csv'%(info_dir))
columns = info.columns.tolist()

high_low = info['High_Low']
gen_soph = info['General_Sophistication']

fig,axs=plt.subplots(4,7,figsize=(40, 15))
fig.subplots_adjust(wspace=0.2,hspace=0.4)

axs = axs.ravel()
times = np.linspace(-200,800,161)

for trained_on in ['partial']:
    condition = 'scale'
    dir = 'down'
    fig.suptitle('%s_%s'%(condition,dir))

    if condition == 'circular':
        num_tones = 7
    else:
        num_tones = 8
    cmap = cm.rainbow(np.linspace(0, 1, num_tones))
    for j in range(len(subjects)):
        print subjects[j]
        if high_low[j] == 1:
            sync = 'high'
        else:
            sync = 'low'

        np_preds = list()

        info_fname = '%s/%s/%s_freq_train%s_ypreds.csv' % (base_dir, subjects[j], subjects[j],
                                                      trained_on)
        trial_info = pd.read_csv(info_fname)

        ypred_fname = '%s/%s/%s_freq_train%s_ypreds.npy' % (base_dir, subjects[j], subjects[j],
                                                      trained_on)
        ypred = np.load(ypred_fname)

        # sanity
        assert(trial_info.shape[0] == ypred.shape[1])

        # dims
        n_times, n_trials = ypred.shape

        idx = np.logical_and(trial_info['circscale'].values == condition,
                                trial_info['updown'].values == dir)
        trial_info = trial_info[idx]
        ypred = ypred[:, idx]

        for i in range(num_tones):
            print i
            # get ypreds for note position
            idx_preds = list()
            idx_np = trial_info['note_position'].values == i+1
            idx_preds.append(ypred[:, idx_np].mean(1))
            np_preds.append(idx_preds)
        np_preds = np.array(np_preds)
        np_preds=np_preds.mean(1)
        for i in range(num_tones):
            axs[j].plot(times,np_preds[i],color=cmap[i],linewidth=0.8)
            axs[j].set_title('%s, %s, MSI: %s'%(subjects[j],sync,gen_soph[j]))
    handles=[]
    for i in range(num_tones):
        label = mpatches.Patch(color=cmap[i], label='%s'%(i+1))
        handles.append(label)
    fig.legend(handles=handles[0:num_tones])
    # fig.savefig('/Users/ea84/Dropbox/shepard_decoding/_GRP_PLOTS/n=28/group/ypreds/by_subj/group_bysubj_%s_%s_%s'%(trained_on,condition,dir))
    fig.show()
