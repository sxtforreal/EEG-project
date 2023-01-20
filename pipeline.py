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
#from google.colab import drive
#drive.mount('/content/gdrive')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Train on GPU...")
else:
    print("Train on CPU...")
    
    
    
### Data preparation    
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
    
def data_for_EEGNET(subject: str, which_round: str):
    """
    Prepare the input data to CNN model.
    """
    r = mne.io.read_raw_edf('/content/gdrive/MyDrive/dat/Subject' + subject + '/Session001/Train/RowColumn/Subject_' + subject + '_Train_S001R' + which_round + '.edf')
    r.load_data()

    # Rename channels
    mne.channels.rename_channels(r.info, {'EEG_F3':'F3','EEG_Fz':'Fz','EEG_F4':'F4','EEG_T7':'T7','EEG_C3':'C3','EEG_Cz':'Cz','EEG_C4':'C4','EEG_T8':'T8','EEG_CP3':'CP3','EEG_CP4':'CP4','EEG_P3':'P3','EEG_Pz':'Pz','EEG_P4':'P4','EEG_PO7':'PO7','EEG_PO8':'PO8','EEG_Oz':'Oz','EEG_FP1':'Fp1','EEG_FP2':'Fp2','EEG_F7':'F7','EEG_F8':'F8','EEG_FC5':'FC5','EEG_FC1':'FC1','EEG_FC2':'FC2','EEG_FC6':'FC6','EEG_CPz':'CPz','EEG_P7':'P7','EEG_P5':'P5','EEG_PO3':'PO3','EEG_POz':'POz','EEG_PO4':'PO4','EEG_O1':'O1','EEG_O2':'O2'})
    
    # Reset channel types
    r.set_channel_types({'IsGazeValid':'syst', 'EyeGazeX':'syst', 'EyeGazeY':'syst', 'PupilSizeLeft':'syst', 'PupilSizeRight':'syst', 'EyePosX':'syst', 'EyePosY':'syst', 'EyeDist':'syst', 'A_1_1':'syst', 'B_1_2':'syst', 'C_1_3':'syst', 'D_1_4':'syst', 'E_1_5':'syst', 'F_1_6':'syst', 'G_1_7':'syst', 'H_1_8':'syst', 'I_2_1':'syst', 'J_2_2':'syst', 'K_2_3':'syst', 'L_2_4':'syst', 'M_2_5':'syst', 'N_2_6':'syst', 'O_2_7':'syst', 'P_2_8':'syst', 'Q_3_1':'syst', 'R_3_2':'syst', 'S_3_3':'syst', 'T_3_4':'syst', 'U_3_5':'syst', 'V_3_6':'syst', 'W_3_7':'syst', 'X_3_8':'syst', 'Y_4_1':'syst', 'Z_4_2':'syst', 'Sp_4_3':'syst', '1_4_4':'syst', '2_4_5':'syst', '3_4_6':'syst', '4_4_7':'syst', '5_4_8':'syst', '6_5_1':'syst', '7_5_2':'syst', '8_5_3':'syst', '9_5_4':'syst', '0_5_5':'syst', 'Prd_5_6':'syst', 'Ret_5_7':'syst', 'Bs_5_8':'syst', '?_6_1':'syst', ',_6_2':'syst', ';_6_3':'syst', '\\_6_4':'syst', '/_6_5':'syst', '+_6_6':'syst', '-_6_7':'syst', 'Alt_6_8':'syst', 'Ctrl_7_1':'syst', '=_7_2':'syst', 'Del_7_3':'syst', 'Home_7_4':'syst', 'UpAw_7_5':'syst', 'End_7_6':'syst', 'PgUp_7_7':'syst', 'Shft_7_8':'syst', 'Save_8_1':'syst', "'_8_2":'syst', 'F2_8_3':'syst', 'LfAw_8_4':'syst', 'DnAw_8_5':'syst', 'RtAw_8_6':'syst', 'PgDn_8_7':'syst', 'Pause_8_8':'syst', 'Caps_9_1':'syst', 'F5_9_2':'syst', 'Tab_9_3':'syst', 'EC_9_4':'syst', 'Esc_9_5':'syst', 'email_9_6':'syst', '!_9_7':'syst', 'Sleep_9_8':'syst', 'StimulusType':'syst', 'SelectedTarget':'syst', 'SelectedRow':'syst', 'SelectedColumn':'syst', 'PhaseInSequence':'syst', 'CurrentTarget':'syst', 'BCISelection':'syst', 'Error':'syst'})
    
    # Set reference
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
    #ica = mne.preprocessing.ICA(n_components = 15, random_state = 1)
    #ica.fit(r)
    # Use Fp1 and Fp2 channels as proxies to the missing EOG channel. Base on observation, eye artifacts exist in every data so we want to auto-remove it from all independent components.
    #ica.exclude = []
    #eog_indices, eog_scores = ica.find_bads_eog(r, ch_name = ['Fp1','Fp2'], measure = 'correlation', threshold = 'auto')
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
   
    # Get label
    label = get_true_label(r)
    
    return epoch_dat_r, label

def data_generator(subject: str):
    dat1, label1 = data_for_EEGNET(subject,'01')
    dat2, label2 = data_for_EEGNET(subject,'02')
    dat3, label3 = data_for_EEGNET(subject,'03')
    dat4, label4 = data_for_EEGNET(subject,'04')
    dat5, label5 = data_for_EEGNET(subject,'05')
    dat6, label6 = data_for_EEGNET(subject,'06')
    dat = np.concatenate((dat1, dat2, dat3, dat4, dat5, dat6))
    label = np.concatenate((label1, label2, label3, label4, label5, label6))
    return dat, label
  
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


### Neural Network
import math
import random 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm.notebook import tqdm, trange
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

# Unlist a nested list
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
    data = np.load('/content/gdrive/MyDrive/BCI matrices/dat_' + subject + '.npy')
    data = data.astype(np.float32)
    label = np.load('/content/gdrive/MyDrive/BCI matrices/label_' + subject + '.npy')
    label = label.astype(np.float32)
    subject = np.repeat(int(subject), 6120)
    subject = np.reshape(subject, (6120, -1))
    subject = subject.astype(np.float32)
    x.append(data)
    x.append(label)
    x.append(subject)
    return x
  
 ### EEGNet architecture
 class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.C = 9
        self.F1 = 9
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
        
    def forward(self, x, subject):
        # Block 1
        x1 = self.conv1(x)
        x2 = self.conv11(x)
        x3 = self.conv12(x)
        x = torch.cat((x1, x2, x3), dim = 1)
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = F.dropout(x, p = 0.25)
        # Block 2
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.block2_bn(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 5))
        x = F.dropout(x, p = 0.25)
        # Classifier
        x = self.flatten(x)
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
        x = torch.sigmoid(x)
        return x

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

### Model training
# Use 15 subjects to train the encoder and the fully connected layers
def train_model(BATCH, Learning_rate, EPOCH, seed):
    # A dictionary of all training patients' data and labels
    subjects = {'01': data_reader('01'), '02': data_reader('02'), '03': data_reader('03'), '04': data_reader('04'), '05': data_reader('05'), '06': data_reader('06'), '07': data_reader('07'), '08': data_reader('08'), '09': data_reader('09'), '10': data_reader('10'), '11': data_reader('11'), '13': data_reader('13'), '14': data_reader('14'), '15': data_reader('15'), '16': data_reader('16')}
    
    # Training set
    data = []
    label = []
    train_subject = []
    for i in subjects:
        data.append(subjects[i][0])
        label.append(subjects[i][1])
        train_subject.append(subjects[i][2])
    x_tr = np.concatenate(data)
    x_tr = torch.from_numpy(np.expand_dims(x_tr, axis = 1))
    y_tr = np.concatenate(label)
    y_tr = torch.from_numpy(np.reshape(y_tr, (y_tr.size,))).long()
    sub_tr = np.concatenate(train_subject)
    sub_tr = torch.from_numpy(np.reshape(sub_tr, (sub_tr.size,))).long()
    
    # Create data loader
    trainset = TensorDataset(x_tr, y_tr, sub_tr)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH, shuffle = True)
    
    # Instantiate model
    model = EEGNet()
    
    # Loss and Optimizer
    criterion = nn.BCELoss() #https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/2
    optimizer = torch.optim.Adam(model.parameters(), lr = Learning_rate)
    
    # Set seed
    torch.manual_seed(seed)
    
    # Model training
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
    
    torch.save(model.state_dict(), '/content/gdrive/MyDrive/BCI matrices/model_weights.pth')
    torch.save(model, '/content/gdrive/MyDrive/BCI matrices/model.pth')

 train_model(1, 0.002, 1, 0)

### Collect pre-trained features from pre-trained model
subjects = {'01': data_reader('01'), '02': data_reader('02'), '03': data_reader('03'), '04': data_reader('04'), '05': data_reader('05'), '06': data_reader('06'), '07': data_reader('07'), '08': data_reader('08'), '09': data_reader('09'), '10': data_reader('10'), '11': data_reader('11'), '13': data_reader('13'), '14': data_reader('14'), '15': data_reader('15'), '16': data_reader('16'), '17': data_reader('17')}
data = []
label = []
train_subject = []
for i in subjects:
    data.append(subjects[i][0])
    label.append(subjects[i][1])
    train_subject.append(subjects[i][2])
x_tr = np.concatenate(data)
x_tr = torch.from_numpy(np.expand_dims(x_tr, axis = 1))
y_tr = np.concatenate(label)
y_tr = torch.from_numpy(np.reshape(y_tr, (y_tr.size,))).long()
sub_tr = np.concatenate(train_subject)
sub_tr = torch.from_numpy(np.reshape(sub_tr, (sub_tr.size,))).long()

trainset = TensorDataset(x_tr, y_tr, sub_tr)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle = False)

model = EEGNet()
model.load_state_dict(torch.load('/content/gdrive/MyDrive/BCI matrices/model_weights.pth'))
model.flatten.register_forward_hook(get_activation('flatten'))
model.eval()
features = []
sub = []
with torch.no_grad():
    # Iterate through test set minibatchs 
    for signals, labels, subject in tqdm(train_loader):
        # Forward pass, choose the most similar subject
        if subject == '17':
          y = model(signals, '01')
        else: 
          y = model(signals, subject)
        features.append(activation['flatten'])
        sub.append(subject)

# Encoded feature matrix (97920*540)
encoded_feature = np.zeros(shape=(97920, 540))
for i in range(len(features)):
  arr = features[i].cpu().detach().numpy().flatten()
  encoded_feature[i] = arr

# Individual encoded feature matrix (6120*540)
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
feature_sub13 = encoded_feature[67320:73440,:]
feature_sub14 = encoded_feature[73440:79560,:]
feature_sub15 = encoded_feature[79560:85680,:]
feature_sub16 = encoded_feature[85680:91800,:]
feature_sub17 = encoded_feature[91800:97920,:]

### We use KL divergence and Wasserstein distance to measure similarity
# Cholesky decomposition
import scipy
import scipy.linalg

# Cholesky decomposition
def lower_t(feature):
  cov = np.cov(feature, rowvar = False)
  L = scipy.linalg.cholesky(cov, lower = True)
  return L

mean1 = np.mean(feature_sub1, axis=0)
mean2 = np.mean(feature_sub2, axis=0)
mean3 = np.mean(feature_sub3, axis=0)
mean4 = np.mean(feature_sub4, axis=0)
mean5 = np.mean(feature_sub5, axis=0)
mean6 = np.mean(feature_sub6, axis=0)
mean7 = np.mean(feature_sub7, axis=0)
mean8 = np.mean(feature_sub8, axis=0)
mean9 = np.mean(feature_sub9, axis=0)
mean10 = np.mean(feature_sub10, axis=0)
mean11 = np.mean(feature_sub11, axis=0)
mean13 = np.mean(feature_sub13, axis=0)
mean14 = np.mean(feature_sub14, axis=0)
mean15 = np.mean(feature_sub15, axis=0)
mean16 = np.mean(feature_sub16, axis=0)
mean17 = np.mean(feature_sub17, axis=0)

l1 = lower_t(feature_sub1)
l2 = lower_t(feature_sub2)
l3 = lower_t(feature_sub3)
l4 = lower_t(feature_sub4)
l5 = lower_t(feature_sub5)
l6 = lower_t(feature_sub6)
l7 = lower_t(feature_sub7)
l8 = lower_t(feature_sub8)
l9 = lower_t(feature_sub9)
l10 = lower_t(feature_sub10)
l11 = lower_t(feature_sub11)
l13 = lower_t(feature_sub13)
l14 = lower_t(feature_sub14)
l15 = lower_t(feature_sub15)
l16 = lower_t(feature_sub16)
l17 = lower_t(feature_sub17)

def KL(mu1, mu0, l1, l0, k=540):
  M = np.linalg.solve(l1, l0) #540*540
  y = np.linalg.solve(l1, (mu1-mu0)) #540
  A = 0
  C = 0
  for i in range(k):
    A += M[i,i]**2
    C += (np.log(l1[i,i])-np.log(l0[i,i]))
  B = y.dot(y)
  return 1/2*(A-k+B+2*C)

def wasserstein(mu1,mu0,l1,l0):
  return np.linalg.norm(mu1-mu0)**2 + np.linalg.norm(l1-l0,'fro')**2

# AUCs of subject 17 using different pre-trained classifiers
def auc(which_sub):
  subject = {'17': data_reader('17')}
  x_tt = torch.from_numpy(np.expand_dims(subject['17'][0], axis = 1))
  y_tt = torch.from_numpy(np.reshape(subject['17'][1], (subject['17'][1].size,))).long()
  testset = TensorDataset(x_tt, y_tt)
  test_loader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False)

  model = EEGNet()
  model.load_state_dict(torch.load('/content/gdrive/MyDrive/BCI matrices/model_weights.pth'))
  model.eval()
  predicted_prob = []
  true_label = []
  with torch.no_grad():
    # Iterate through test set minibatchs 
    for signals, labels in tqdm(test_loader):
        y = model(signals, which_sub)
        predicted_prob.append(y)
        true_label.append(labels.data.numpy())

  predicted_prob_target = unlist(predicted_prob)
  true_label = unlist(true_label)

  auc = roc_auc_score(true_label, predicted_prob_target)
  print('AUC score: {}'.format(auc))
  ap = average_precision_score(true_label, predicted_prob_target)
  print('Average precision score: {}'.format(ap))
  return auc,ap

auc1,ap1 = auc(1)
auc2,ap2 = auc(2)
auc3,ap3 = auc(3)
auc4,ap4 = auc(4)
auc5,ap5 = auc(5)
auc6,ap6 = auc(6)
auc7,ap7 = auc(7)
auc8,ap8 = auc(8)
auc9,ap9 = auc(9)
auc10,ap10 = auc(10)
auc11,ap11 = auc(11)
auc13,ap13 = auc(13)
auc14,ap14 = auc(14)
auc15,ap15 = auc(15)
auc16,ap16 = auc(16)

### Visualization
auc = [auc1,auc2,auc3,auc4,auc5,auc6,auc7,auc8,auc9,auc10,auc11,auc13,auc14,auc15,auc16]
ap = [ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10,ap11,ap13,ap14,ap15,ap16]
# KL divergence with reference group = 17
KL = [KL(mean1,mean17,l1,l17),KL(mean2,mean17,l2,l17),KL(mean3,mean17,l3,l17),KL(mean4,mean17,l4,l17),KL(mean5,mean17,l5,l17),KL(mean6,mean17,l6,l17),KL(mean7,mean17,l7,l17),KL(mean8,mean17,l8,l17),KL(mean9,mean17,l9,l17),KL(mean10,mean17,l10,l17),KL(mean11,mean17,l11,l17),KL(mean13,mean17,l13,l17),KL(mean14,mean17,l14,l17),KL(mean15,mean17,l15,l17),KL(mean16,mean17,l16,l17)]
# Wasserstein distance with reference group = 17
wass = [wasserstein(mean1,mean17,l1,l17),wasserstein(mean2,mean17,l2,l17),wasserstein(mean3,mean17,l3,l17),wasserstein(mean4,mean17,l4,l17),wasserstein(mean5,mean17,l5,l17),wasserstein(mean6,mean17,l6,l17),wasserstein(mean7,mean17,l7,l17),wasserstein(mean8,mean17,l8,l17),wasserstein(mean9,mean17,l9,l17),wasserstein(mean10,mean17,l10,l17),wasserstein(mean11,mean17,l11,l17),wasserstein(mean13,mean17,l13,l17),wasserstein(mean14,mean17,l14,l17),wasserstein(mean15,mean17,l15,l17),wasserstein(mean16,mean17,l16,l17)]

plt.scatter(auc, KL)
plt.title('AUC vs KL')
plt.xlabel('AUC')
plt.ylabel('KL')
plt.show()

#plt.scatter(ap, KL)
#plt.title('AP vs KL')
#plt.xlabel('AP')
#plt.ylabel('KL')
#plt.show()

plt.scatter(auc, wass)
plt.title('AUC vs Wasserstein')
plt.xlabel('AUC')
plt.ylabel('Wass_d')
plt.show()

#plt.scatter(ap, wass)
#plt.title('AP vs Wasserstein')
#plt.xlabel('AP')
#plt.ylabel('Wass_d')
#plt.show()

## Check distribution
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})
sns.boxplot(data=feature_sub13)
