!pip install mne
!pip install umap
!pip install -U ipykernel
import os
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
import pandas as pd
from mne import (io, compute_raw_covariance, read_events, pick_types, Epochs)
from mne.datasets import sample
from mne.preprocessing import Xdawn
from mne.viz import plot_epochs_image
import plotly.express as px
from sklearn.model_selection import train_test_split
from google.colab import drive
import math
import random 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm.notebook import tqdm, trange
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import scipy
import scipy.linalg
#drive.mount('/content/gdrive')

# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Train on GPU...")
else:
    print("Train on CPU...")

##### Zero-training CNN

# Generate training and test data for CNN
def create_stim_channel(x):
    """
    Create a STIM channel by combining existing channels.
    """
    dat = x.get_data()
    length = dat.shape[1]
    stim = np.zeros(shape = (1,length), dtype = int)
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
    STIM = np.zeros(shape = (1,length), dtype = int)
    idx = np.where(stim[0] != 0)
    # Since we don't care about the difference between row flash and column flash, we encode them as 1
    STIM[0][idx] = 1
    return STIM

def get_true_label(x, epoch_dat_x):
    """
    Get the true label for the training set.
    """
    # Need 1020 labels per round because we have 1020 flashes per round, 1 flash == 1 epoch, each epoch need a label
    length = epoch_dat_x.shape[0]
    label = np.zeros(length)
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
    label = np.reshape(label, (length, 1))
    return label

def data_for_EEGNET(subject: str, which_round: str, training: bool):
    """
    Get the input feature matrix and the label of the given subject's data.
    """
    if training is True:
        r = mne.io.read_raw_edf('/content/gdrive/MyDrive/dat/Subject' + subject + '/Session001/Train/RowColumn/Subject_' + subject + '_Train_S001R' + which_round + '.edf')
    else:
        r = mne.io.read_raw_edf('/content/gdrive/MyDrive/dat/Subject' + subject + '/Session001/Test/DynamicStopping/Subject_' + subject + '_Ds_S001R' + which_round + '.edf')
    r.load_data()

    # Rename channels
    mne.channels.rename_channels(r.info, {'EEG_F3':'F3','EEG_Fz':'Fz','EEG_F4':'F4','EEG_T7':'T7','EEG_C3':'C3','EEG_Cz':'Cz','EEG_C4':'C4','EEG_T8':'T8','EEG_CP3':'CP3','EEG_CP4':'CP4','EEG_P3':'P3','EEG_Pz':'Pz','EEG_P4':'P4','EEG_PO7':'PO7','EEG_PO8':'PO8','EEG_Oz':'Oz','EEG_FP1':'Fp1','EEG_FP2':'Fp2','EEG_F7':'F7','EEG_F8':'F8','EEG_FC5':'FC5','EEG_FC1':'FC1','EEG_FC2':'FC2','EEG_FC6':'FC6','EEG_CPz':'CPz','EEG_P7':'P7','EEG_P5':'P5','EEG_PO3':'PO3','EEG_POz':'POz','EEG_PO4':'PO4','EEG_O1':'O1','EEG_O2':'O2'})
    
    # Reset channel types
    r.set_channel_types({'IsGazeValid':'syst', 'EyeGazeX':'syst', 'EyeGazeY':'syst', 'PupilSizeLeft':'syst', 'PupilSizeRight':'syst', 'EyePosX':'syst', 'EyePosY':'syst', 'EyeDist':'syst', 'A_1_1':'syst', 'B_1_2':'syst', 'C_1_3':'syst', 'D_1_4':'syst', 'E_1_5':'syst', 'F_1_6':'syst', 'G_1_7':'syst', 'H_1_8':'syst', 'I_2_1':'syst', 'J_2_2':'syst', 'K_2_3':'syst', 'L_2_4':'syst', 'M_2_5':'syst', 'N_2_6':'syst', 'O_2_7':'syst', 'P_2_8':'syst', 'Q_3_1':'syst', 'R_3_2':'syst', 'S_3_3':'syst', 'T_3_4':'syst', 'U_3_5':'syst', 'V_3_6':'syst', 'W_3_7':'syst', 'X_3_8':'syst', 'Y_4_1':'syst', 'Z_4_2':'syst', 'Sp_4_3':'syst', '1_4_4':'syst', '2_4_5':'syst', '3_4_6':'syst', '4_4_7':'syst', '5_4_8':'syst', '6_5_1':'syst', '7_5_2':'syst', '8_5_3':'syst', '9_5_4':'syst', '0_5_5':'syst', 'Prd_5_6':'syst', 'Ret_5_7':'syst', 'Bs_5_8':'syst', '?_6_1':'syst', ',_6_2':'syst', ';_6_3':'syst', '\\_6_4':'syst', '/_6_5':'syst', '+_6_6':'syst', '-_6_7':'syst', 'Alt_6_8':'syst', 'Ctrl_7_1':'syst', '=_7_2':'syst', 'Del_7_3':'syst', 'Home_7_4':'syst', 'UpAw_7_5':'syst', 'End_7_6':'syst', 'PgUp_7_7':'syst', 'Shft_7_8':'syst', 'Save_8_1':'syst', "'_8_2":'syst', 'F2_8_3':'syst', 'LfAw_8_4':'syst', 'DnAw_8_5':'syst', 'RtAw_8_6':'syst', 'PgDn_8_7':'syst', 'Pause_8_8':'syst', 'Caps_9_1':'syst', 'F5_9_2':'syst', 'Tab_9_3':'syst', 'EC_9_4':'syst', 'Esc_9_5':'syst', 'email_9_6':'syst', '!_9_7':'syst', 'Sleep_9_8':'syst', 'StimulusType':'syst', 'SelectedTarget':'syst', 'SelectedRow':'syst', 'SelectedColumn':'syst', 'PhaseInSequence':'syst', 'CurrentTarget':'syst', 'BCISelection':'syst', 'Error':'syst'})
    
    # Set Common Average Reference(CAR) to denoise 
    r.set_eeg_reference('average', projection = True)

    # Create and add STIM channel
    STIM = create_stim_channel(r)
    info = mne.create_info(['STI'], r.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(STIM, info)
    r.add_channels([stim_raw], force_update_info = True)
       
    # Set montage(electrode location on scalp) to be standard 10-20 montage
    montage1020 = mne.channels.make_standard_montage('standard_1020')
    picks = ['F3','Fz','F4','T7','C3','Cz','C4','T8','CP3','CP4','P3','Pz','P4','PO7','PO8','Oz','Fp1','Fp2','F7','F8','FC5','FC1','FC2','FC6','CPz','P7','P5','PO3','POz','PO4','O1','O2']
    ind = [i for (i, channel) in enumerate(montage1020.ch_names) if channel in picks]
    montage1020_new = montage1020.copy()
    montage1020_new.ch_names = [montage1020.ch_names[x] for x in ind]
    kept_channel_info = [montage1020.dig[x+3] for x in ind]
    montage1020_new.dig = montage1020.dig[0:3]+kept_channel_info
    r.set_montage(montage1020_new)
    
    # ICA
    ica = mne.preprocessing.ICA(n_components = 0.99, method = 'fastica')
    ica.fit(r)
    # Use Fp1 and Fp2 channels as bipolar referencing channels to simulate EOG channel. By doing this, we can drop the eye movement artifacts ica component.
    eog_indices, eog_scores = ica.find_bads_eog(r, ch_name = ['Fp1','Fp2'], measure = 'correlation', threshold = 'auto')
    ica.exclude = eog_indices
    # Visualize the differences after removing the blinks
    #ica.plot_overlay(r, exclude = eog_indices, picks = ['Fz','Cz','P3','Pz','P4','PO7','PO8','Oz'])
    ica.apply(r)
    
    # Define parameters
    tmin, tmax = 0.0, 0.8
    event_id = {'Flash': 1}
    baseline = None
    events = mne.find_events(r)
    
    # Epoching
    epoch_r = mne.Epochs(r, events = events, event_id = event_id, tmin = tmin, tmax = tmax, baseline = baseline, picks = picks)
    epoch_dat_r = epoch_r.get_data()
   
    # Get label
    label = get_true_label(r, epoch_dat_r)
    
    return epoch_dat_r, label

def data_generator_train(subject: str):
    dat1, label1 = data_for_EEGNET(subject,'01',True)
    dat2, label2 = data_for_EEGNET(subject,'02',True)
    dat3, label3 = data_for_EEGNET(subject,'03',True)
    dat4, label4 = data_for_EEGNET(subject,'04',True)
    dat5, label5 = data_for_EEGNET(subject,'05',True)
    dat6, label6 = data_for_EEGNET(subject,'06',True)
    dat = np.concatenate((dat1, dat2, dat3, dat4, dat5, dat6))
    label = np.concatenate((label1, label2, label3, label4, label5, label6))
    return dat, label

def data_generator_test(subject: str):
    dat1, label1 = data_for_EEGNET(subject,'01',False)
    dat2, label2 = data_for_EEGNET(subject,'02',False)
    dat3, label3 = data_for_EEGNET(subject,'03',False)
    dat4, label4 = data_for_EEGNET(subject,'04',False)
    dat5, label5 = data_for_EEGNET(subject,'05',False)
    dat6, label6 = data_for_EEGNET(subject,'06',False)
    dat = np.concatenate((dat1, dat2, dat3, dat4, dat5, dat6))
    label = np.concatenate((label1, label2, label3, label4, label5, label6))
    return dat, label

# Training data and labels
dat_01, label_01 = data_generator('01')
dat_02, label_02 = data_generator('02')
dat_03, label_03 = data_generator('03')
dat_04, label_04 = data_generator('04')
dat_05, label_05 = data_generator('05')
dat_06, label_06 = data_generator('06')
dat_07, label_07 = data_generator('07')
dat_08, label_08 = data_generator('08')
dat_09, label_09 = data_generator('09')
dat_10, label_10 = data_generator('10')
dat_11, label_11 = data_generator('11')
dat_13, label_13 = data_generator('13')
dat_14, label_14 = data_generator('14')
dat_15, label_15 = data_generator('15')
dat_16, label_16 = data_generator('16')
dat_17, label_17 = data_generator('17')

# Test data and labels
dat_01_test, label_01_test = data_generator_test('01')
dat_02_test, label_02_test = data_generator_test('02')
dat_03_test, label_03_test = data_generator_test('03')
dat_04_test, label_04_test = data_generator_test('04')
dat_05_test, label_05_test = data_generator_test('05')
dat_06_test, label_06_test = data_generator_test('06')
dat_07_test, label_07_test = data_generator_test('07')
dat_08_test, label_08_test = data_generator_test('08')
dat_09_test, label_09_test = data_generator_test('09')
dat_10_test, label_10_test = data_generator_test('10')
dat_11_test, label_11_test = data_generator_test('11')
dat_13_test, label_13_test = data_generator_test('13')
dat_14_test, label_14_test = data_generator_test('14')
dat_15_test, label_15_test = data_generator_test('15')
dat_16_test, label_16_test = data_generator_test('16')
dat_17_test, label_17_test = data_generator_test('17')

def unlist(lst):
    flattened_list = []
    for ele in lst:
      if type(ele) is list:
        for item in ele:
          flattened_list.append(item)
      else:
        flattened_list.append(ele)
    return flattened_list

def data_reader(subject):
    x = []
    data = np.load('/content/gdrive/MyDrive/BCI matrices/training/dat_' + subject + '.npy')
    data = data.astype(np.float32)
    label = np.load('/content/gdrive/MyDrive/BCI matrices/training/label_' + subject + '.npy')
    label = label.astype(np.float32)
    length = data.shape[0]
    subject = np.repeat(int(subject), length)
    subject = np.reshape(subject, (length, -1))
    subject = subject.astype(np.float32)
    x.append(data)
    x.append(label)
    x.append(subject)
    return x

def data_reader_test(subject):
    x = []
    data = np.load('/content/gdrive/MyDrive/BCI matrices/test/dat_' + subject + '_test.npy')
    data = data.astype(np.float32)
    label = np.load('/content/gdrive/MyDrive/BCI matrices/test/label_' + subject + '_test.npy')
    label = label.astype(np.float32)
    length = data.shape[0]
    subject = np.repeat(int(subject), length)
    subject = np.reshape(subject, (length, -1))
    subject = subject.astype(np.float32)
    x.append(data)
    x.append(label)
    x.append(subject)
    return x
  
### EEGNet architecture
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.C = 32
        self.F1 = 8
        self.D = 2
        self.F2 = self.F1 * self.D # Number of filters for separable conv, set to be F2
        
        self.conv1 = nn.Conv2d(1, self.F1, (1, 104), padding = 'same', bias = False) #because sampling frequency is 256Hz --> 206s
        self.conv11 = nn.Conv2d(1, self.F1, (1, 52), padding = 'same', bias = False) #half of previous
        self.conv12 = nn.Conv2d(1, self.F1, (1, 27), padding = 'same', bias = False) #half of previous
        self.conv1_bn = nn.BatchNorm2d(self.F1*3)
        self.conv2 = nn.Conv2d(self.F1*3, self.D*self.F1*3, (self.C, 1), groups = self.F1*3, padding = 'valid', bias = False) #https://pytorch.org/docs/0.3.1/nn.html#torch.nn.Conv2d
        self.conv2_bn = nn.BatchNorm2d(self.D*self.F1*3)
        self.depthwise = nn.Conv2d(self.D*self.F1*3, self.F2*3, (1, 5), padding = 'same', groups = self.D*self.F1*3, bias = False)
        self.pointwise = nn.Conv2d(self.F2*3, self.F2*3, (1, 1), bias = False) #https://www.analyticsvidhya.com/blog/2021/11/an-introduction-to-separable-convolutions/#:~:text=A%20Separable%20Convolution%20is%20a,to%20achieve%20the%20same%20effect.
        self.block2_bn = nn.BatchNorm2d(self.F2*3)
        self.flatten = nn.Flatten(0, -1)

        self.fc1 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc2 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc3 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc4 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc5 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc6 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc7 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc8 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc9 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc10 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc11 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc13 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc14 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc15 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc16 = nn.Linear(self.F2*30, 1, bias = True)
        self.fc17 = nn.Linear(self.F2*30, 1, bias = True)      
        
    def forward(self, x, subject):
        x1 = self.conv1(x) # 1st temporal filter
        x2 = self.conv11(x) # 2nd temporal filter
        x3 = self.conv12(x) # 3rd temporal filter
        x = torch.cat((x1, x2, x3), dim = 1) # stack feature maps
        x = self.conv1_bn(x) # batch normalization
        x = self.conv2(x) # depthwise filter
        x = self.conv2_bn(x) # batch normalization
        x = F.elu(x) # activation
        x = F.avg_pool2d(x, (1, 4)) # average pooling
        x = F.dropout(x, p = 0.25) # dropout
        x = self.depthwise(x) # separable part1
        x = self.pointwise(x) # separable part2
        x = self.block2_bn(x) # batch normalization
        x = F.elu(x) # activation
        x = F.avg_pool2d(x, (1, 5)) # average pooling
        x = F.dropout(x, p = 0.25) # dropout
        x = self.flatten(x) # flatten

        if subject == 1:
            x = self.fc1(x)
        elif subject == 2:
            x = self.fc2(x)
        elif subject == 3:
            x = self.fc3(x)
        elif subject == 4:
            x = self.fc4(x)
        elif subject == 5:
            x = self.fc5(x)
        elif subject == 6:
            x = self.fc6(x)
        elif subject == 7:
            x = self.fc7(x)
        elif subject == 8:
            x = self.fc8(x)
        elif subject == 9:
            x = self.fc9(x)
        elif subject == 10:
            x = self.fc10(x)
        elif subject == 11:
            x = self.fc11(x)
        elif subject == 13:
            x = self.fc13(x)
        elif subject == 14:
            x = self.fc14(x)
        elif subject == 15:
            x = self.fc15(x)
        elif subject == 16:
            x = self.fc16(x)
        elif subject == 17:
            x = self.fc17(x)
        x = torch.sigmoid(x)
        return x

# Initiate hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Print model
model = EEGNet()
print(model)

# Cholesky decomposition
def lower_t(feature):
  cov = np.cov(feature, rowvar = False)
  L = scipy.linalg.cholesky(cov, lower = True)
  return L

def KL(mu1, mu0, l1, l0, k):
  M = np.linalg.solve(l1, l0) #k*k
  y = np.linalg.solve(l1, (mu1-mu0)) #k
  A = 0
  C = 0
  for i in range(k):
    A += M[i,i]**2
    C += (np.log(l1[i,i])-np.log(l0[i,i]))
  B = y.dot(y)
  return 1/2*(A-k+B+2*C)

def wasserstein(mu1,mu0,l1,l0):
  return np.linalg.norm(mu1-mu0)**2 + np.linalg.norm(l1-l0,'fro')**2

def train_model(BATCH, Learning_rate, EPOCH, Leave_out, Feature_dim):
    # A dictionary of all patients' data, labels, subject names
    subjects = {'01': data_reader('01'), '02': data_reader('02'), '03': data_reader('03'), '04': data_reader('04'),
                '05': data_reader('05'), '06': data_reader('06'), '07': data_reader('07'), '08': data_reader('08'), 
                '09': data_reader('09'), '10': data_reader('10'), '11': data_reader('11'), '13': data_reader('13'), 
                '14': data_reader('14'), '15': data_reader('15'), '16': data_reader('16'), '17': data_reader('17')}
    del subjects[Leave_out] #remove test subject from dictionary
    
    # Training set
    train_data = []
    train_label = []
    train_subject = []
    for i in subjects:
        train_data.append(subjects[i][0])
        train_label.append(subjects[i][1])
        train_subject.append(subjects[i][2])
    x_tr = np.concatenate(train_data)
    x_tr = torch.from_numpy(np.expand_dims(x_tr, axis = 1))
    y_tr = np.concatenate(train_label)
    y_tr = torch.from_numpy(np.reshape(y_tr, (y_tr.size,))).long()
    sub_tr = np.concatenate(train_subject)
    sub_tr = torch.from_numpy(np.reshape(sub_tr, (sub_tr.size,))).long()
    
    # Create data loaders
    trainset = TensorDataset(x_tr, y_tr, sub_tr)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH, shuffle = True)
    
    # Instantiate model
    model = EEGNet()
    
    # Loss and Optimizer
    criterion = nn.BCELoss() #https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/2
    optimizer = torch.optim.NAdam(model.parameters(), lr = Learning_rate)
    
    # Set seed
    torch.manual_seed(1)
    
##### Model training
    for epoch in trange(EPOCH):
        for signals, labels, subject in tqdm(train_loader):

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            y = model(signals, subject)
            labels = labels.float()
            loss = criterion(y, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
    
    # Save trained model
    torch.save(model.state_dict(), f"/content/gdrive/MyDrive/BCI matrices/model{Leave_out}_weights.pth")
    torch.save(model, f"/content/gdrive/MyDrive/BCI matrices/model{Leave_out}.pth")

##### Extract latent features from pre-trained model
    feature_extract_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH, shuffle = False)
    model = EEGNet()
    model.load_state_dict(torch.load(f"/content/gdrive/MyDrive/BCI matrices/model{Leave_out}_weights.pth"))
    model.flatten.register_forward_hook(get_activation('flatten'))
    model.eval()
    features = []
    sub = []
    with torch.no_grad():
        for signals, labels, subject in tqdm(feature_extract_loader):
            y = model(signals, subject)
            features.append(activation['flatten'])
            sub.append(subject)

##### Compute similarity scores on feature space, determine the most similar sibject    
    encoded_feature = np.zeros(shape=(91800, Feature_dim))
    for i in range(len(features)):
      arr = features[i].cpu().detach().numpy().flatten()
      encoded_feature[i] = arr
    # Individual encoded feature matrix
    feature_sub1 = encoded_feature[0:6120,:]
    feature_sub2 = encoded_feature[6120:12240,:]
    feature_sub3 = encoded_feature[12240:18360,:]
    feature_sub4 = encoded_feature[18360:24480,:]
    feature_sub5 = encoded_feature[24480:30600,:]
    feature_sub6 = encoded_feature[30600:36720,:]
    feature_sub7 = encoded_feature[36720:42840,:]
    feature_sub8 = encoded_feature[42840:48960,:]
    feature_sub9 = encoded_feature[48960:55080,:]
    feature_sub10 = encoded_feature[55080:61200,:]
    feature_sub11 = encoded_feature[61200:67320,:]
    feature_sub12 = encoded_feature[67320:73440,:]
    feature_sub13 = encoded_feature[73440:79560,:]
    feature_sub14 = encoded_feature[79560:85680,:]
    feature_sub15 = encoded_feature[85680:91800,:]
    features_list = [feature_sub1,feature_sub2,feature_sub3,feature_sub4,feature_sub5,
                     feature_sub6,feature_sub7,feature_sub8,feature_sub9,feature_sub10,
                     feature_sub11,feature_sub12,feature_sub13,feature_sub14,feature_sub15]
    f_list = [[np.mean(feature_matrix, axis=0), lower_t(feature_matrix)] for feature_matrix in features_list]
    
    for j, key in enumerate(subjects.keys()):
        subjects[key] = f_list[j] #{training subject: [mean, lower triangular]}
    
    return subjects

def fit_model(Leave_out, BATCH, features_dict, Feature_dim):
    length = data_reader_test(Leave_out)[0].shape[0]
    # Test set
    test_data = torch.from_numpy(np.expand_dims(data_reader_test(Leave_out)[0], axis = 1))
    test_label = torch.from_numpy(np.reshape(data_reader_test(Leave_out)[1], (data_reader_test(Leave_out)[1].size,))).long()
    testset = TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH, shuffle = False)
    
    ##### Compute AUC and AP based on suggested optimal subject
    model = EEGNet()
    model.load_state_dict(torch.load(f"/content/gdrive/MyDrive/BCI matrices/model{Leave_out}_weights.pth"))
    model.flatten.register_forward_hook(get_activation('flatten'))
    model.eval()
    
    features = []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader):
            y = model(signals, Leave_out)
            features.append(activation['flatten'])
    
    encoded_feature = np.zeros(shape=(length, Feature_dim))
    for i in range(len(features)):
        arr = features[i].cpu().detach().numpy().flatten()
        encoded_feature[i] = arr
    
    reference_mean = np.mean(encoded_feature, axis=0) #Mean of the test subject's encoded features
    reference_l = lower_t(encoded_feature) #Lower triangle of the test subject's encoded features

    KL_list = [KL(features_dict[key][0], reference_mean, features_dict[key][1], reference_l, Feature_dim) for key in features_dict]
    wass_list = [wasserstein(features_dict[key][0], reference_mean, features_dict[key][1], reference_l) for key in features_dict]
    similarity_scores = [[KL_list[k], wass_list[k]] for k in range(len(KL_list))]
    training_keys = list(features_dict.keys())
    similarity_dict = dict(zip(training_keys, similarity_scores)) #{subject: [KL, Wasserstein]}
    # Most similar subject
    most_similar_sub_KL = min(similarity_dict.items(), key = lambda x:x[1][0])[0]
    most_similar_sub_wass = min(similarity_dict.items(), key = lambda x:x[1][1])[0]
    print('KL most similar:', most_similar_sub_KL)
    print('Wass most similar:', most_similar_sub_wass)
    others = list(similarity_dict.keys())
    predicted_prob_1 = []
    predicted_prob_2 = []
    predicted_prob_3 = []
    predicted_prob_4 = []
    predicted_prob_5 = []
    predicted_prob_6 = []
    predicted_prob_7 = []
    predicted_prob_8 = []
    predicted_prob_9 = []
    predicted_prob_10 = []
    predicted_prob_11 = []
    predicted_prob_12 = []
    predicted_prob_13 = []
    predicted_prob_14 = []
    predicted_prob_15 = []
    true_label = []
    with torch.no_grad():
      # Iterate through test set minibatchs 
      for signals, labels in tqdm(test_loader):
          y1 = model(signals, int(others[0]))
          predicted_prob_1.append(y1.data.numpy())
          y2 = model(signals, int(others[1]))
          predicted_prob_2.append(y2.data.numpy())
          y3 = model(signals, int(others[2]))
          predicted_prob_3.append(y3.data.numpy())
          y4 = model(signals, int(others[3]))
          predicted_prob_4.append(y4.data.numpy())
          y5 = model(signals, int(others[4]))
          predicted_prob_5.append(y5.data.numpy())
          y6 = model(signals, int(others[5]))
          predicted_prob_6.append(y6.data.numpy())
          y7 = model(signals, int(others[6]))
          predicted_prob_7.append(y7.data.numpy())
          y8 = model(signals, int(others[7]))
          predicted_prob_8.append(y8.data.numpy())
          y9 = model(signals, int(others[8]))
          predicted_prob_9.append(y9.data.numpy())
          y10 = model(signals, int(others[9]))
          predicted_prob_10.append(y10.data.numpy())
          y11 = model(signals, int(others[10]))
          predicted_prob_11.append(y11.data.numpy())
          y12 = model(signals, int(others[11]))
          predicted_prob_12.append(y12.data.numpy())
          y13 = model(signals, int(others[12]))
          predicted_prob_13.append(y13.data.numpy())
          y14 = model(signals, int(others[13]))
          predicted_prob_14.append(y14.data.numpy())
          y15 = model(signals, int(others[14]))
          predicted_prob_15.append(y15.data.numpy())
          # True label
          true_label.append(labels.data.numpy())

    predicted_prob_1 = unlist(predicted_prob_1)
    predicted_prob_2 = unlist(predicted_prob_2)
    predicted_prob_3 = unlist(predicted_prob_3)
    predicted_prob_4 = unlist(predicted_prob_4)
    predicted_prob_5 = unlist(predicted_prob_5)
    predicted_prob_6 = unlist(predicted_prob_6)
    predicted_prob_7 = unlist(predicted_prob_7)
    predicted_prob_8 = unlist(predicted_prob_8)
    predicted_prob_9 = unlist(predicted_prob_9)
    predicted_prob_10 = unlist(predicted_prob_10)
    predicted_prob_11 = unlist(predicted_prob_11)
    predicted_prob_12 = unlist(predicted_prob_12)
    predicted_prob_13 = unlist(predicted_prob_13)
    predicted_prob_14 = unlist(predicted_prob_14)
    predicted_prob_15 = unlist(predicted_prob_15)
    true_label = unlist(true_label)

    auc_1 = roc_auc_score(true_label, predicted_prob_1)
    auc_2 = roc_auc_score(true_label, predicted_prob_2)
    auc_3 = roc_auc_score(true_label, predicted_prob_3)
    auc_4 = roc_auc_score(true_label, predicted_prob_4)
    auc_5 = roc_auc_score(true_label, predicted_prob_5)
    auc_6 = roc_auc_score(true_label, predicted_prob_6)
    auc_7 = roc_auc_score(true_label, predicted_prob_7)
    auc_8 = roc_auc_score(true_label, predicted_prob_8)
    auc_9 = roc_auc_score(true_label, predicted_prob_9)
    auc_10 = roc_auc_score(true_label, predicted_prob_10)
    auc_11 = roc_auc_score(true_label, predicted_prob_11)
    auc_12 = roc_auc_score(true_label, predicted_prob_12)
    auc_13 = roc_auc_score(true_label, predicted_prob_13)
    auc_14 = roc_auc_score(true_label, predicted_prob_14)
    auc_15 = roc_auc_score(true_label, predicted_prob_15)
    auc_list = [auc_1,auc_2,auc_3,auc_4,auc_5,auc_6,auc_7,auc_8,
                auc_9,auc_10,auc_11,auc_12,auc_13,auc_14,auc_15]
    auc_dict = dict(zip(training_keys, auc_list))
    auc_dict_values = list(auc_dict.values())
    print('KL AUC score: {}'.format(auc_dict[most_similar_sub_KL]))
    print('Wass AUC score: {}'.format(auc_dict[most_similar_sub_wass]))
    print('Max AUC score: {}'.format(np.max(auc_dict_values)))
    print('Min AUC score: {}'.format(np.min(auc_dict_values)))
    print('Median AUC score: {}'.format(np.median(auc_dict_values)))

    return auc_list

##### Within-training LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SequentialFeatureSelector

def feature_extraction(dat):
    """Extract features from epoch data.
    """
    length = dat.shape[0]
    feature_arr = np.zeros((length, 120))
    # ith flash (0-1019)
    for i in range(length):
        arr = np.array([])
        data = dat[i]
        # Remove last 11 samples
        chopped = np.delete(data, np.s_[-11:], axis = 1)
        # Divide (by column) into equally spaced arrays
        splitted = np.hsplit(chopped, 15)
        # jth sub-window (0-14)
        for j in range(len(splitted)):
            # kth channel (0-7)
            for k in range(len(splitted[j])):
                #feature = np.max(splitted[j][k]) - np.min(splitted[j][k])
                feature = np.mean(splitted[j][k])
                arr = np.append(arr, feature)
        feature_arr[i] = arr
    return feature_arr

def get_feature_matrix(subject: str, which_round: str, training: bool):
    """
    Get the feature matrix and the label of the given subject's training data.
    """
    if training is True:
        r = mne.io.read_raw_edf('/content/gdrive/MyDrive/dat/Subject' + subject + '/Session001/Train/RowColumn/Subject_' + subject + '_Train_S001R' + which_round + '.edf')
    else:
        r = mne.io.read_raw_edf('/content/gdrive/MyDrive/dat/Subject' + subject + '/Session001/Test/DynamicStopping/Subject_' + subject + '_Ds_S001R' + which_round + '.edf')
    r.load_data()
  
    # Rename channels
    mne.channels.rename_channels(r.info, {'EEG_F3':'F3','EEG_Fz':'Fz','EEG_F4':'F4','EEG_T7':'T7','EEG_C3':'C3','EEG_Cz':'Cz','EEG_C4':'C4','EEG_T8':'T8','EEG_CP3':'CP3','EEG_CP4':'CP4','EEG_P3':'P3','EEG_Pz':'Pz','EEG_P4':'P4','EEG_PO7':'PO7','EEG_PO8':'PO8','EEG_Oz':'Oz','EEG_FP1':'Fp1','EEG_FP2':'Fp2','EEG_F7':'F7','EEG_F8':'F8','EEG_FC5':'FC5','EEG_FC1':'FC1','EEG_FC2':'FC2','EEG_FC6':'FC6','EEG_CPz':'CPz','EEG_P7':'P7','EEG_P5':'P5','EEG_PO3':'PO3','EEG_POz':'POz','EEG_PO4':'PO4','EEG_O1':'O1','EEG_O2':'O2'})
    
    # Reset channel types
    r.set_channel_types({'IsGazeValid':'syst', 'EyeGazeX':'syst', 'EyeGazeY':'syst', 'PupilSizeLeft':'syst', 'PupilSizeRight':'syst', 'EyePosX':'syst', 'EyePosY':'syst', 'EyeDist':'syst', 'A_1_1':'syst', 'B_1_2':'syst', 'C_1_3':'syst', 'D_1_4':'syst', 'E_1_5':'syst', 'F_1_6':'syst', 'G_1_7':'syst', 'H_1_8':'syst', 'I_2_1':'syst', 'J_2_2':'syst', 'K_2_3':'syst', 'L_2_4':'syst', 'M_2_5':'syst', 'N_2_6':'syst', 'O_2_7':'syst', 'P_2_8':'syst', 'Q_3_1':'syst', 'R_3_2':'syst', 'S_3_3':'syst', 'T_3_4':'syst', 'U_3_5':'syst', 'V_3_6':'syst', 'W_3_7':'syst', 'X_3_8':'syst', 'Y_4_1':'syst', 'Z_4_2':'syst', 'Sp_4_3':'syst', '1_4_4':'syst', '2_4_5':'syst', '3_4_6':'syst', '4_4_7':'syst', '5_4_8':'syst', '6_5_1':'syst', '7_5_2':'syst', '8_5_3':'syst', '9_5_4':'syst', '0_5_5':'syst', 'Prd_5_6':'syst', 'Ret_5_7':'syst', 'Bs_5_8':'syst', '?_6_1':'syst', ',_6_2':'syst', ';_6_3':'syst', '\\_6_4':'syst', '/_6_5':'syst', '+_6_6':'syst', '-_6_7':'syst', 'Alt_6_8':'syst', 'Ctrl_7_1':'syst', '=_7_2':'syst', 'Del_7_3':'syst', 'Home_7_4':'syst', 'UpAw_7_5':'syst', 'End_7_6':'syst', 'PgUp_7_7':'syst', 'Shft_7_8':'syst', 'Save_8_1':'syst', "'_8_2":'syst', 'F2_8_3':'syst', 'LfAw_8_4':'syst', 'DnAw_8_5':'syst', 'RtAw_8_6':'syst', 'PgDn_8_7':'syst', 'Pause_8_8':'syst', 'Caps_9_1':'syst', 'F5_9_2':'syst', 'Tab_9_3':'syst', 'EC_9_4':'syst', 'Esc_9_5':'syst', 'email_9_6':'syst', '!_9_7':'syst', 'Sleep_9_8':'syst', 'StimulusType':'syst', 'SelectedTarget':'syst', 'SelectedRow':'syst', 'SelectedColumn':'syst', 'PhaseInSequence':'syst', 'CurrentTarget':'syst', 'BCISelection':'syst', 'Error':'syst'})
    
    # Set Common Average Reference(CAR) to denoise 
    r.set_eeg_reference('average', projection = True)

    # Create and add STIM channel
    STIM = create_stim_channel(r)
    info = mne.create_info(['STI'], r.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(STIM, info)
    r.add_channels([stim_raw], force_update_info = True)
       
    # Set montage(electrode location on scalp) to be standard 10-20 montage
    montage1020 = mne.channels.make_standard_montage('standard_1020')
    picks = ['F3','Fz','F4','T7','C3','Cz','C4','T8','CP3','CP4','P3','Pz','P4','PO7','PO8','Oz','Fp1','Fp2','F7','F8','FC5','FC1','FC2','FC6','CPz','P7','P5','PO3','POz','PO4','O1','O2']
    ind = [i for (i, channel) in enumerate(montage1020.ch_names) if channel in picks]
    montage1020_new = montage1020.copy()
    montage1020_new.ch_names = [montage1020.ch_names[x] for x in ind]
    kept_channel_info = [montage1020.dig[x+3] for x in ind]
    montage1020_new.dig = montage1020.dig[0:3]+kept_channel_info
    r.set_montage(montage1020_new)
    
    # ICA
    ica = mne.preprocessing.ICA(n_components = 0.99, method = 'fastica')
    ica.fit(r)
    # Use Fp1 and Fp2 channels as bipolar referencing channels to simulate EOG channel. By doing this, we can drop the eye movement artifacts ica component.
    eog_indices, eog_scores = ica.find_bads_eog(r, ch_name = ['Fp1','Fp2'], measure = 'correlation', threshold = 'auto')
    ica.exclude = eog_indices
    ica.apply(r)

    # Define parameters
    tmin, tmax = 0.0, 0.8
    events = mne.find_events(r)
    event_id = {'Flashes': 1}
    new_picks = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
    baseline = None
    
    # Epoching
    epoch_r = mne.Epochs(r, events = events, event_id = event_id, tmin = tmin, tmax = tmax, baseline = baseline, picks = new_picks)
    epoch_dat_r = epoch_r.get_data()
    
    # Feature extraction
    feature_matrix = feature_extraction(epoch_dat_r)
    
    # Get label
    label = get_true_label(r, epoch_dat_r)

    return feature_matrix, label

def feature_generator_train(subject: str):
    f1, label1 = get_feature_matrix(subject, '01', True)
    f2, label2 = get_feature_matrix(subject, '02', True)
    f3, label3 = get_feature_matrix(subject, '03', True)
    f4, label4 = get_feature_matrix(subject, '04', True)
    f5, label5 = get_feature_matrix(subject, '05', True)
    f6, label6 = get_feature_matrix(subject, '06', True)
    f = np.concatenate((f1, f2, f3, f4, f5, f6))
    label = np.concatenate((label1, label2, label3, label4, label5, label6))
    return f, label

def feature_generator_test(subject: str):
    f1, label1 = get_feature_matrix(subject, '01', False)
    f2, label2 = get_feature_matrix(subject, '02', False)
    f3, label3 = get_feature_matrix(subject, '03', False)
    f4, label4 = get_feature_matrix(subject, '04', False)
    f5, label5 = get_feature_matrix(subject, '05', False)
    f6, label6 = get_feature_matrix(subject, '06', False)
    f = np.concatenate((f1, f2, f3, f4, f5, f6))
    label = np.concatenate((label1, label2, label3, label4, label5, label6))
    return f, label

fm1, l1 = feature_generator_train('01')
fm2, l2 = feature_generator_train('02')
fm3, l3 = feature_generator_train('03')
fm4, l4 = feature_generator_train('04')
fm5, l5 = feature_generator_train('05')
fm6, l6 = feature_generator_train('06')
fm7, l7 = feature_generator_train('07')
fm8, l8 = feature_generator_train('08')
fm9, l9 = feature_generator_train('09')
fm10, l10 = feature_generator_train('10')
fm11, l11 = feature_generator_train('11')
fm13, l13 = feature_generator_train('13')
fm14, l14 = feature_generator_train('14')
fm15, l15 = feature_generator_train('15')
fm16, l16 = feature_generator_train('16')
fm17, l17 = feature_generator_train('17')

fm1_test, l1_test = feature_generator_test('01')
fm2_test, l2_test = feature_generator_test('02')
fm3_test, l3_test = feature_generator_test('03')
fm4_test, l4_test = feature_generator_test('04')
fm5_test, l5_test = feature_generator_test('05')
fm6_test, l6_test = feature_generator_test('06')
fm7_test, l7_test = feature_generator_test('07')
fm8_test, l8_test = feature_generator_test('08')
fm9_test, l9_test = feature_generator_test('09')
fm10_test, l10_test = feature_generator_test('10')
fm11_test, l11_test = feature_generator_test('11')
fm13_test, l13_test = feature_generator_test('13')
fm14_test, l14_test = feature_generator_test('14')
fm15_test, l15_test = feature_generator_test('15')
fm16_test, l16_test = feature_generator_test('16')
fm17_test, l17_test = feature_generator_test('17')

sub = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
for i in sub:
  lda = LinearDiscriminantAnalysis()
  exec(f"lda.fit(fm{i}, l{i}.flatten())")
  exec(f"y_pred = lda.predict(fm{i}_test)")
  exec(f"y_pred_prob = lda.predict_proba(fm{i}_test)")
  exec(f"auc = roc_auc_score(l{i}_test.flatten(), y_pred_prob[:,1])")
  print(f"AUC{i}:", auc)
