def feature_extraction(x):
    """
    Extract features from epoch data. Feature defined as the average of sub-window.
    """
    feature_arr = np.zeros((1020, 120))
    # ith flash (0-1019)
    for i in range(x.shape[0]):
        arr = np.array([])
        data = x[i]
        # Remove last 11 samples
        chopped = np.delete(data, np.s_[-11:], axis = 1)
        # Divide (by column) into equally spaced arrays
        splitted = np.hsplit(chopped, 15)
        # jth sub-window (0-14)
        for j in range(len(splitted)):
            # kth channel (0-7)
            for k in range(len(splitted[j])):
                feature = np.mean(splitted[j][k])
                arr = np.append(arr, feature)
        feature_arr[i] = arr
    return feature_arr

def get_feature_matrix(subject: str, which_round: str):
    """
    Get the feature matrix from the given subject's training data.
    Outputs 6120*120 array for each subject.
    """
    r = mne.io.read_raw_edf('/content/gdrive/MyDrive/dat/Subject' + subject + '/Session001/Train/RowColumn/Subject_' + subject + '_Train_S001R' + which_round + '.edf')
    r.load_data()
    
    # Rename channels
    mne.channels.rename_channels(r.info, {'EEG_F3':'F3','EEG_Fz':'Fz','EEG_F4':'F4','EEG_T7':'T7','EEG_C3':'C3','EEG_Cz':'Cz','EEG_C4':'C4','EEG_T8':'T8','EEG_CP3':'CP3','EEG_CP4':'CP4','EEG_P3':'P3','EEG_Pz':'Pz','EEG_P4':'P4','EEG_PO7':'PO7','EEG_PO8':'PO8','EEG_Oz':'Oz','EEG_FP1':'Fp1','EEG_FP2':'Fp2','EEG_F7':'F7','EEG_F8':'F8','EEG_FC5':'FC5','EEG_FC1':'FC1','EEG_FC2':'FC2','EEG_FC6':'FC6','EEG_CPz':'CPz','EEG_P7':'P7','EEG_P5':'P5','EEG_PO3':'PO3','EEG_POz':'POz','EEG_PO4':'PO4','EEG_O1':'O1','EEG_O2':'O2'})
    
    # Reset channel types
    r.set_channel_types({'IsGazeValid':'syst', 'EyeGazeX':'syst', 'EyeGazeY':'syst', 'PupilSizeLeft':'syst', 'PupilSizeRight':'syst', 'EyePosX':'syst', 'EyePosY':'syst', 'EyeDist':'syst', 'A_1_1':'syst', 'B_1_2':'syst', 'C_1_3':'syst', 'D_1_4':'syst', 'E_1_5':'syst', 'F_1_6':'syst', 'G_1_7':'syst', 'H_1_8':'syst', 'I_2_1':'syst', 'J_2_2':'syst', 'K_2_3':'syst', 'L_2_4':'syst', 'M_2_5':'syst', 'N_2_6':'syst', 'O_2_7':'syst', 'P_2_8':'syst', 'Q_3_1':'syst', 'R_3_2':'syst', 'S_3_3':'syst', 'T_3_4':'syst', 'U_3_5':'syst', 'V_3_6':'syst', 'W_3_7':'syst', 'X_3_8':'syst', 'Y_4_1':'syst', 'Z_4_2':'syst', 'Sp_4_3':'syst', '1_4_4':'syst', '2_4_5':'syst', '3_4_6':'syst', '4_4_7':'syst', '5_4_8':'syst', '6_5_1':'syst', '7_5_2':'syst', '8_5_3':'syst', '9_5_4':'syst', '0_5_5':'syst', 'Prd_5_6':'syst', 'Ret_5_7':'syst', 'Bs_5_8':'syst', '?_6_1':'syst', ',_6_2':'syst', ';_6_3':'syst', '\\_6_4':'syst', '/_6_5':'syst', '+_6_6':'syst', '-_6_7':'syst', 'Alt_6_8':'syst', 'Ctrl_7_1':'syst', '=_7_2':'syst', 'Del_7_3':'syst', 'Home_7_4':'syst', 'UpAw_7_5':'syst', 'End_7_6':'syst', 'PgUp_7_7':'syst', 'Shft_7_8':'syst', 'Save_8_1':'syst', "'_8_2":'syst', 'F2_8_3':'syst', 'LfAw_8_4':'syst', 'DnAw_8_5':'syst', 'RtAw_8_6':'syst', 'PgDn_8_7':'syst', 'Pause_8_8':'syst', 'Caps_9_1':'syst', 'F5_9_2':'syst', 'Tab_9_3':'syst', 'EC_9_4':'syst', 'Esc_9_5':'syst', 'email_9_6':'syst', '!_9_7':'syst', 'Sleep_9_8':'syst', 'StimulusType':'syst', 'SelectedTarget':'syst', 'SelectedRow':'syst', 'SelectedColumn':'syst', 'PhaseInSequence':'syst', 'CurrentTarget':'syst', 'BCISelection':'syst', 'Error':'syst'})
   
    # Set reference channel
    r.set_eeg_reference('average', projection = True)
    
    # Create and add STIM channel
    STIM = create_stim_channel(r)
    info = mne.create_info(['STI'], r.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(STIM, info)
    r.add_channels([stim_raw], force_update_info = True)
       
    # Set montage(electrode location on scalp)
    montage1020 = mne.channels.make_standard_montage('standard_1020')
    picks = ['F3','Fz','F4','T7','C3','Cz','C4','T8','CP3','CP4','P3','Pz','P4','PO7','PO8','Oz','Fp1','Fp2','F7','F8','FC5','FC1','FC2','FC6','CPz','P7','P5','PO3','POz','PO4','O1','O2']
    ind = [i for (i, channel) in enumerate(montage1020.ch_names) if channel in picks]
    montage1020_new = montage1020.copy()
    montage1020_new.ch_names = [montage1020.ch_names[x] for x in ind]
    kept_channel_info = [montage1020.dig[x+3] for x in ind]
    montage1020_new.dig = montage1020.dig[0:3]+kept_channel_info
    r.set_montage(montage1020_new)
       
    # ICA
    #ica = mne.preprocessing.ICA(n_components=15, random_state=1)
    #ica.fit(r)
    # Use Fp1 and Fp2 channels as proxies to the missing EOG channel. Base on observation, eye artifacts exist in every data so we want to auto-remove it from all independent components.
    #ica.exclude = []
    #eog_indices, eog_scores = ica.find_bads_eog(r, ch_name=['Fp1','Fp2'], measure='correlation', threshold='auto')
    #ica.exclude = eog_indices
    #ica.apply(r)
    
    # Define parameters
    tmin, tmax = 0.0, 0.8
    event_id = {'Flash': 1}
    baseline = None
    events = mne.find_events(r)
    
    # Epoching
    epoch_r = mne.Epochs(r, events = events, event_id = event_id, tmin = tmin,
                    tmax = tmax, baseline = baseline, picks = ['Cz', 'CPz', 'Fz', 'P7', 'PO7', 'O1', 'Oz', 'O2', 'PO8'])
    epoch_dat_r = epoch_r.get_data()
    
    # Feature extraction
    f = feature_extraction(epoch_dat_r)

    #feature_matrix = np.concatenate((f1, f2, f3, f4, f5, f6))

    return f

def feature_generator(subject: str):
    f1 = get_feature_matrix(subject,'01')
    f2 = get_feature_matrix(subject,'02')
    f3 = get_feature_matrix(subject,'03')
    f4 = get_feature_matrix(subject,'04')
    f5 = get_feature_matrix(subject,'05')
    f6 = get_feature_matrix(subject,'06')
    F = np.concatenate((f1, f2, f3, f4, f5, f6))
    return F
