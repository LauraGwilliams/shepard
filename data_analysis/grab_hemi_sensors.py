import mne

def grab_hemi_sensors(epochs,exclude_center=False):

    # grab info and channels from epochs
    info = epochs.info
    chs = info['chs'] # this is a dictionary of channels and their info

    if exclude_center:
        thresh = 0.03
    else:
        thresh = 0

    rh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] > thresh]]
    lh_ch_names = [ch['ch_name'] for ch in [i for i in chs if i['loc'][0] < -thresh]]

    rh_picks = mne.pick_types(epochs.info,selection=rh_ch_names)
    lh_picks = mne.pick_types(epochs.info,selection=lh_ch_names)

    return rh_picks, lh_picks
