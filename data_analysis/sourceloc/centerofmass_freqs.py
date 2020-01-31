import os
import numpy as np
import pandas as pd
from mne import read_epochs, read_source_estimate, compute_source_morph, vertex_to_mni
from mne.minimum_norm import read_inverse_operator, apply_inverse
from functools import reduce
from operator import add
from matplotlib import cm
from surfer import Brain

subjects = ['A0216','A0270','A0280','A0305','A0306','A0307','A0314',
            'A0323','A0326','A0344','A0345','A0353','A0354','A0355',
            'A0357','A0358','A0362','A0364','A0365','A0367','A0368',
            'A0369','A0370','P010','P011','P014','P015','P022']

snr = 2.0  # Standard assumption for single trial
lambda2 = 1.0 / snr ** 2

mri_dir = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri'
meg_dir = '/Users/ea84/Dropbox/shepard_sourceloc/'
os.environ["SUBJECTS_DIR"] = '/Volumes/Server/MORPHLAB/Users/Ellie/Shepard/mri/'

tonetypes = ['pure','partial']


# plot ALL frequencies of given tonetype:
frequencies = [220, 247, 262, 277, 294, 312, 330, 349, 370, 392,
                415, 440, 466, 494, 523, 587, 624]

tonetype = 'pure'
hemi = 'rh'
hem = 1

cmap = cm.rainbow(np.linspace(0,1,len(frequencies)))

t_bin = []

b = Brain('fsaverage', hemi = hemi , surf = 'pial',background='white')

for f,freq in enumerate(frequencies):
    vtx, _, t = np.load(meg_dir + '_STCS/%s/%s/_CM_%s_%s_%s_cropped.npy' % (tonetype,freq,tonetype,freq,hemi))
    # vtx, _, t = np.load(meg_dir + '_STCS/%s/%s/_CM_%s_%s_%s_cropped.npy' % (tonetype,freq,tonetype,freq,hemi))
    # get the mni co-ordinates of the center of mass
    coords = vertex_to_mni(int(vtx), hemis=hem, subject='fsaverage')
    print (coords)
    # plot result
    b.add_foci(coords, color=cmap[f], map_surface='pial', scale_factor=0.5)
    t_bin.append(t)



key = 'A'

if key == 'A':
    frequencies = [220, 247, 277, 294, 330, 370, 415, 440]
elif key == 'C':
    frequencies = [262, 294, 330, 349, 392, 440, 494, 523]
elif key == 'Eb':
    frequencies = [312, 349, 392, 415, 466, 523, 587, 624]

for freq in frequencies:
    if not os.path.exists('/Users/ea84/Dropbox/shepard_sourceloc/_STCS/partial/%s/%s'%(key,freq)):
        os.makedirs('/Users/ea84/Dropbox/shepard_sourceloc/_STCS/partial/%s/%s'%(key,freq))

# params
for subject in subjects:
    print (subject)
    subj_dir = '%s%s/'%(meg_dir,subject)

    # paths
    epochs_fname = subj_dir + subject+ '_shepard-epo.fif'
    inv_fname = subj_dir+subject+'_shepard-inv.fif'


    print ("Loading variables...")
    epochs = read_epochs(epochs_fname)
    epochs.metadata['freq'].replace(524,523,inplace=True)

    inverse_operator = read_inverse_operator(inv_fname)
    src = inverse_operator['src']

    # make stcs for each frequency for each tonetype
    for tonetype in tonetypes:
        print("Beginning %s tones..."%(tonetype))
        # frequencies = np.sort(epochs.metadata[np.logical_and(epochs.metadata['key']==key,
        #                                                 epochs.metadata['condition']==tonetype)].freq.unique())
        for freq in frequencies:
            print(freq)

            freq_epochs = epochs[(epochs.metadata['condition']==tonetype) &
                            (epochs.metadata['freq']==freq) &
                            (epochs.metadata['key']==key)]
            evoked = freq_epochs.average()

            # apply inverse to evoked for frequency
            print ("Creating stcs...")
            stcs = apply_inverse(evoked, inverse_operator, lambda2,
                                              method='dSPM')
            print("Morphing %s to fsaverage..."%(subject))
            morph = compute_source_morph(stcs, subject_from=subject,
                                     subject_to='fsaverage', spacing=4)
            stc_fsaverage = morph.apply(stcs)
            print ("Saving!")
            print (meg_dir+'_STCS/%s/%s/%s/%s_%s_%s_morphed'%(tonetype,key,freq,subject,tonetype,freq))
            stc_fsaverage.save(meg_dir+'_STCS/%s/%s/%s/%s_%s_%s_morphed'%(tonetype,key,freq,subject,tonetype,freq))


# average stcs for each frequency and get center of mass
for tonetype in tonetypes:
    print("COM for %s"%(tonetype))
    for freq in frequencies:
        print('freq=%s'%(freq))
        stcs = []
        for subject in subjects:
            stc = read_source_estimate(meg_dir+'_STCS/%s/%s/%s/%s_%s_%s_morphed'%(tonetype,key,freq,subject,tonetype,freq))
            stcs.append(stc)
        stc_avg = reduce(add, stcs)
        stc_avg /= len(stcs)

        stc_avg._data = np.abs(stc_avg._data)
        stc_avg_cropped = stc_avg.copy().crop(0,0.3)

        print("Computing COM for %s %sHz"%(tonetype,freq))
        for hem,h in zip(['lh','rh'],[0,1]):
            vtx, _, t = stc_avg_cropped.center_of_mass(subject = 'fsaverage',hemi=h, restrict_vertices=False)
            # vtx2,_2,t2 = stc_avg_cropped.center_of_mass(subject = 'fsaverage',hemi=hem, restrict_vertices=False)
            print(vtx)
            np.save(meg_dir + '_STCS/%s/%s/%s/_CM_%s_%s_%s_%s_cropped.npy' % (tonetype,key,freq,tonetype,key,freq,hem), (vtx, _, t))

tonetype = 'pure'
hemi = 'rh'
hem = 1

cmap = cm.rainbow(np.linspace(0,1,len(frequencies)))

t_bin = []

b = Brain('fsaverage', hemi = hemi , surf = 'pial',background='white',alpha=0.5)

for f,freq in enumerate(frequencies):
    vtx, _, t = np.load(meg_dir + '_STCS/%s/%s/%s/_CM_%s_%s_%s_%s_cropped.npy' % (tonetype,key,freq,tonetype,key,freq,hemi))
    # vtx, _, t = np.load(meg_dir + '_STCS/%s/%s/_CM_%s_%s_%s_cropped.npy' % (tonetype,freq,tonetype,freq,hemi))
    # get the mni co-ordinates of the center of mass
    coords = vertex_to_mni(int(vtx), hemis=hem, subject='fsaverage')
    print (coords)
    # plot result
    b.add_foci(coords, color=cmap[f], map_surface='pial', scale_factor=0.5)
    t_bin.append(t)

b.save_image(meg_dir + '_STCS/%s/%s/_CM_%s.png'%(tonetype,key,key))

brain = stc_avg.plot(subject='fsaverage',hemi='rh',colormap='rainbow',smoothing_steps=1)
brain.scale_data_colormap(fmin=220, fmid=370, fmax=624,transparent=True)
