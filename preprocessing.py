import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

## Functions used to create STIM channel and get true labels
def create_stim_channel(x):
    """
    Create a STIM channel by combining existing channels.
    """
    dat = x.get_data()
    stim = np.zeros(shape = (1,40064), dtype = int)
    for i in range(40,112):
        # Find all the time indices when this character channel is 'ON'
        all_ones_idx = np.where(dat[i] == 1)
        # We know we would expect to see 120 flashes per character per round(6*10*2).
        # When we zoom in our data, we find that each flash is 'ON' for 16 units(duration).
        # We only care about the time of onset, so we use the step size 16 to keep the onset time only and get rid of the duration time.
        # This results in an array of shape 120.
        first_ones_idx = all_ones_idx[0][::16]
        stim[0][first_ones_idx] += 1 
    # After for-loop we have an np_array that has 1020 non_zero entries in total, each has value 8(row flash) or 9(column flash).
    STIM = np.zeros(shape = (1,40064), dtype = int)
    idx = np.where(stim[0] != 0)
    # Since we don't care about the difference between row flash and column flash, we encode them as 1
    STIM[0][idx] = 1
    return STIM

def get_true_label(x):
    """
    Get the true label for the training set.
    """
    # Need 1020 labels per round because we have 1020 flashes per round, 1 flash == 1 epoch, each epoch need a label
    label = np.zeros(1020)
    num = 0
    # Loop over indicies in STIM channel
    for i in np.where(x.get_data()[120] == 1)[0]:
        # At the time of each flash onset, find the target character
        true_label = x.get_data()[117][i]
        # Get the channel index of that target character
        index = int(true_label + 39)
        # Check if the target character's channel is ON at the time of flash onset.
        # If yes, it means this epoch should contain ERP
        label[num] = x.get_data()[index][i] == 1
        num += 1
    label = np.reshape(label, (1020, 1))
    return label
 
## Use preprocessing with plot_erp; use preprocessing_nonerp with plot_nonerp.
def preprocessing(subject, which_round):
    """
    Preprocess and pick epochs with ERP.
    """
    r = mne.io.read_raw_edf('/Users/sunxiaotan/Desktop/dat/Subject' + subject + '/Session001/Train/RowColumn/Subject_' + subject + '_Train_S001R' + which_round + '.edf')
    r.load_data()
    mne.channels.rename_channels(r.info, {'EEG_F3':'F3','EEG_Fz':'Fz','EEG_F4':'F4','EEG_T7':'T7','EEG_C3':'C3','EEG_Cz':'Cz','EEG_C4':'C4','EEG_T8':'T8','EEG_CP3':'CP3','EEG_CP4':'CP4','EEG_P3':'P3','EEG_Pz':'Pz','EEG_P4':'P4','EEG_PO7':'PO7','EEG_PO8':'PO8','EEG_Oz':'Oz','EEG_FP1':'Fp1','EEG_FP2':'Fp2','EEG_F7':'F7','EEG_F8':'F8','EEG_FC5':'FC5','EEG_FC1':'FC1','EEG_FC2':'FC2','EEG_FC6':'FC6','EEG_CPz':'CPz','EEG_P7':'P7','EEG_P5':'P5','EEG_PO3':'PO3','EEG_POz':'POz','EEG_PO4':'PO4','EEG_O1':'O1','EEG_O2':'O2'})
    r.set_channel_types({'IsGazeValid':'syst', 'EyeGazeX':'syst', 'EyeGazeY':'syst', 'PupilSizeLeft':'syst', 'PupilSizeRight':'syst', 'EyePosX':'syst', 'EyePosY':'syst', 'EyeDist':'syst', 'A_1_1':'syst', 'B_1_2':'syst', 'C_1_3':'syst', 'D_1_4':'syst', 'E_1_5':'syst', 'F_1_6':'syst', 'G_1_7':'syst', 'H_1_8':'syst', 'I_2_1':'syst', 'J_2_2':'syst', 'K_2_3':'syst', 'L_2_4':'syst', 'M_2_5':'syst', 'N_2_6':'syst', 'O_2_7':'syst', 'P_2_8':'syst', 'Q_3_1':'syst', 'R_3_2':'syst', 'S_3_3':'syst', 'T_3_4':'syst', 'U_3_5':'syst', 'V_3_6':'syst', 'W_3_7':'syst', 'X_3_8':'syst', 'Y_4_1':'syst', 'Z_4_2':'syst', 'Sp_4_3':'syst', '1_4_4':'syst', '2_4_5':'syst', '3_4_6':'syst', '4_4_7':'syst', '5_4_8':'syst', '6_5_1':'syst', '7_5_2':'syst', '8_5_3':'syst', '9_5_4':'syst', '0_5_5':'syst', 'Prd_5_6':'syst', 'Ret_5_7':'syst', 'Bs_5_8':'syst', '?_6_1':'syst', ',_6_2':'syst', ';_6_3':'syst', '\\_6_4':'syst', '/_6_5':'syst', '+_6_6':'syst', '-_6_7':'syst', 'Alt_6_8':'syst', 'Ctrl_7_1':'syst', '=_7_2':'syst', 'Del_7_3':'syst', 'Home_7_4':'syst', 'UpAw_7_5':'syst', 'End_7_6':'syst', 'PgUp_7_7':'syst', 'Shft_7_8':'syst', 'Save_8_1':'syst', "'_8_2":'syst', 'F2_8_3':'syst', 'LfAw_8_4':'syst', 'DnAw_8_5':'syst', 'RtAw_8_6':'syst', 'PgDn_8_7':'syst', 'Pause_8_8':'syst', 'Caps_9_1':'syst', 'F5_9_2':'syst', 'Tab_9_3':'syst', 'EC_9_4':'syst', 'Esc_9_5':'syst', 'email_9_6':'syst', '!_9_7':'syst', 'Sleep_9_8':'syst', 'StimulusType':'syst', 'SelectedTarget':'syst', 'SelectedRow':'syst', 'SelectedColumn':'syst', 'PhaseInSequence':'syst', 'CurrentTarget':'syst', 'BCISelection':'syst', 'Error':'syst'})
    r.set_eeg_reference('average', projection = True)
    STIM = create_stim_channel(r)
    info = mne.create_info(['STI'], r.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(STIM, info)
    r.add_channels([stim_raw], force_update_info = True)
    montage1020 = mne.channels.make_standard_montage('standard_1020') # set montage
    picks = ['F3','Fz','F4','T7','C3','Cz','C4','T8','CP3','CP4','P3','Pz','P4','PO7','PO8','Oz','Fp1','Fp2','F7','F8','FC5','FC1','FC2','FC6','CPz','P7','P5','PO3','POz','PO4','O1','O2']
    ind = [i for (i, channel) in enumerate(montage1020.ch_names) if channel in picks]
    montage1020_new = montage1020.copy()
    montage1020_new.ch_names = [montage1020.ch_names[x] for x in ind]
    kept_channel_info = [montage1020.dig[x+3] for x in ind]
    montage1020_new.dig = montage1020.dig[0:3]+kept_channel_info
    r.set_montage(montage1020_new)
    #ica = mne.preprocessing.ICA(n_components = 15, random_state = 1)
    #ica.fit(r)
    #ica.exclude = []
    #eog_indices, eog_scores = ica.find_bads_eog(r, ch_name = ['Fp1','Fp2'], measure = 'correlation', threshold = 'auto')
    #ica.exclude = eog_indices
    #ica.apply(r)
    tmin, tmax = 0.0, 0.8
    event_id = {'Flash': 1}
    baseline = None
    events = mne.find_events(r)
    epoch_r = mne.Epochs(r, events = events, event_id = event_id, tmin = tmin,
                    tmax = tmax, baseline = baseline, picks = ['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    label = get_true_label(r)

    np.set_printoptions(suppress=True)
    a = np.concatenate((events, label), axis=1)
    a = a[a[:,3] == 1,:]
    a = np.delete(a, 3, 1)
    epochs_with_ERP = a.astype(int)
    ERP = mne.Epochs(r, events = epochs_with_ERP, event_id = event_id, tmin = tmin,
                    tmax = tmax, baseline = baseline, picks = ['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    return ERP

def plot_erp(subject):
    """
    Plot the image of average epochs with ERP.
    """
    ERP1 = preprocessing(subject,'01')
    ERP2 = preprocessing(subject,'02')
    ERP3 = preprocessing(subject,'03')
    ERP4 = preprocessing(subject,'04')
    ERP5 = preprocessing(subject,'05')
    ERP6 = preprocessing(subject,'06')
    A = mne.concatenate_epochs([ERP1,ERP2,ERP3,ERP4,ERP5,ERP6])
    ERP_flash = A['Flash'].average(picks=['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    ERP_flash.plot()
    #ERP_flash.plot_topomap(times=[0.2, 0.3, 0.4], average=0.05)
    #ERP_flash.plot(gfp='only')
    #ERP_flash.plot_joint(times=[0.23, 0.38, 0.5])

def preprocessing_nonerp(subject, which_round):
    """
    Preprocess and pick epochs without ERP.
    """
    r = mne.io.read_raw_edf('/Users/sunxiaotan/Desktop/dat/Subject' + subject + '/Session001/Train/RowColumn/Subject_' + subject + '_Train_S001R' + which_round + '.edf')
    r.load_data()
    mne.channels.rename_channels(r.info, {'EEG_F3':'F3','EEG_Fz':'Fz','EEG_F4':'F4','EEG_T7':'T7','EEG_C3':'C3','EEG_Cz':'Cz','EEG_C4':'C4','EEG_T8':'T8','EEG_CP3':'CP3','EEG_CP4':'CP4','EEG_P3':'P3','EEG_Pz':'Pz','EEG_P4':'P4','EEG_PO7':'PO7','EEG_PO8':'PO8','EEG_Oz':'Oz','EEG_FP1':'Fp1','EEG_FP2':'Fp2','EEG_F7':'F7','EEG_F8':'F8','EEG_FC5':'FC5','EEG_FC1':'FC1','EEG_FC2':'FC2','EEG_FC6':'FC6','EEG_CPz':'CPz','EEG_P7':'P7','EEG_P5':'P5','EEG_PO3':'PO3','EEG_POz':'POz','EEG_PO4':'PO4','EEG_O1':'O1','EEG_O2':'O2'})
    r.set_channel_types({'IsGazeValid':'syst', 'EyeGazeX':'syst', 'EyeGazeY':'syst', 'PupilSizeLeft':'syst', 'PupilSizeRight':'syst', 'EyePosX':'syst', 'EyePosY':'syst', 'EyeDist':'syst', 'A_1_1':'syst', 'B_1_2':'syst', 'C_1_3':'syst', 'D_1_4':'syst', 'E_1_5':'syst', 'F_1_6':'syst', 'G_1_7':'syst', 'H_1_8':'syst', 'I_2_1':'syst', 'J_2_2':'syst', 'K_2_3':'syst', 'L_2_4':'syst', 'M_2_5':'syst', 'N_2_6':'syst', 'O_2_7':'syst', 'P_2_8':'syst', 'Q_3_1':'syst', 'R_3_2':'syst', 'S_3_3':'syst', 'T_3_4':'syst', 'U_3_5':'syst', 'V_3_6':'syst', 'W_3_7':'syst', 'X_3_8':'syst', 'Y_4_1':'syst', 'Z_4_2':'syst', 'Sp_4_3':'syst', '1_4_4':'syst', '2_4_5':'syst', '3_4_6':'syst', '4_4_7':'syst', '5_4_8':'syst', '6_5_1':'syst', '7_5_2':'syst', '8_5_3':'syst', '9_5_4':'syst', '0_5_5':'syst', 'Prd_5_6':'syst', 'Ret_5_7':'syst', 'Bs_5_8':'syst', '?_6_1':'syst', ',_6_2':'syst', ';_6_3':'syst', '\\_6_4':'syst', '/_6_5':'syst', '+_6_6':'syst', '-_6_7':'syst', 'Alt_6_8':'syst', 'Ctrl_7_1':'syst', '=_7_2':'syst', 'Del_7_3':'syst', 'Home_7_4':'syst', 'UpAw_7_5':'syst', 'End_7_6':'syst', 'PgUp_7_7':'syst', 'Shft_7_8':'syst', 'Save_8_1':'syst', "'_8_2":'syst', 'F2_8_3':'syst', 'LfAw_8_4':'syst', 'DnAw_8_5':'syst', 'RtAw_8_6':'syst', 'PgDn_8_7':'syst', 'Pause_8_8':'syst', 'Caps_9_1':'syst', 'F5_9_2':'syst', 'Tab_9_3':'syst', 'EC_9_4':'syst', 'Esc_9_5':'syst', 'email_9_6':'syst', '!_9_7':'syst', 'Sleep_9_8':'syst', 'StimulusType':'syst', 'SelectedTarget':'syst', 'SelectedRow':'syst', 'SelectedColumn':'syst', 'PhaseInSequence':'syst', 'CurrentTarget':'syst', 'BCISelection':'syst', 'Error':'syst'})
    r.set_eeg_reference('average', projection = True)
    STIM = create_stim_channel(r)
    info = mne.create_info(['STI'], r.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(STIM, info)
    r.add_channels([stim_raw], force_update_info = True)
    montage1020 = mne.channels.make_standard_montage('standard_1020')
    picks = ['F3','Fz','F4','T7','C3','Cz','C4','T8','CP3','CP4','P3','Pz','P4','PO7','PO8','Oz','Fp1','Fp2','F7','F8','FC5','FC1','FC2','FC6','CPz','P7','P5','PO3','POz','PO4','O1','O2']
    ind = [i for (i, channel) in enumerate(montage1020.ch_names) if channel in picks]
    montage1020_new = montage1020.copy()
    montage1020_new.ch_names = [montage1020.ch_names[x] for x in ind]
    kept_channel_info = [montage1020.dig[x+3] for x in ind]
    montage1020_new.dig = montage1020.dig[0:3]+kept_channel_info
    r.set_montage(montage1020_new)
    #ica = mne.preprocessing.ICA(n_components = 15, random_state = 1)
    #ica.fit(r)
    #ica.exclude = []
    #eog_indices, eog_scores = ica.find_bads_eog(r, ch_name = ['Fp1','Fp2'], measure = 'correlation', threshold = 'auto')
    #ica.exclude = eog_indices
    #ica.apply(r)
    tmin, tmax = 0.0, 0.8
    event_id = {'Flash': 1}
    baseline = None
    events = mne.find_events(r)
    epoch_r = mne.Epochs(r, events = events, event_id = event_id, tmin = tmin,
                    tmax = tmax, baseline = baseline, picks = ['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    label = get_true_label(r)

    np.set_printoptions(suppress=True)
    a = np.concatenate((events, label), axis=1)
    a = a[a[:,3] == 0,:]
    a = np.delete(a, 3, 1)
    epochs_without_ERP = a.astype(int)
    NONERP = mne.Epochs(r, events = epochs_without_ERP, event_id = event_id, tmin = tmin,
                    tmax = tmax, baseline = baseline, picks = ['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    return NONERP

def plot_nonerp(subject):
    """
    Plot the image of average epochs without ERP.
    """
    NERP1 = preprocessing_nonerp(subject,'01')
    NERP2 = preprocessing_nonerp(subject,'02')
    NERP3 = preprocessing_nonerp(subject,'03')
    NERP4 = preprocessing_nonerp(subject,'04')
    NERP5 = preprocessing_nonerp(subject,'05')
    NERP6 = preprocessing_nonerp(subject,'06')
    A = mne.concatenate_epochs([NERP1,NERP2,NERP3,NERP4,NERP5,NERP6])
    ERP_flash = A['Flash'].average(picks=['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    ERP_flash.plot()
    #ERP_flash.plot_topomap(times=[0.2, 0.3, 0.4], average=0.05)
    #ERP_flash.plot(gfp='only')
    #ERP_flash.plot_joint(times=[0.26,0.33])
