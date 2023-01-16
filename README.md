# Duke AI Health - BCI/EEG project

## Background
BCI applications are meant to provide controls for individuals with severe neuromuscular limitations such as late-stage Amyotrophic Lateral Sclerosis(ALS). The P300 speller is the most widely researched BCI application for communication purposes(Farwell and Donchin, 1988) which allows its users to communicate through their brain activity. 

From the raw EEG signal data collected through non-invasive electrodes, the P300 speller exploits adaptive algorithms to identify the presence of the P300 wave from each epoch. The P300 wave is a type of evoked Event-Related Potential(ERP) associated with human attention, which is widely used as an indicator of user's reaction to stimulus. In this sense, the P300 speller’s user experience completely depends on the performance of the algorithm, and the performance is evaluated on both accuracy and speed. Current P300 spellers(using SWLDA) are very slow due to noisy brain activity(drawback of non-invasive BCI) and variability across users(shape and latency of P300 wave). Our goal for this project is to come up with a new method that beats the benchmark Stepwise Linear Discriminant Analysis(SWLDA) method in both accuracy and speed. Our effort can better off the lives of patients who suffer from severe neuromuscular limitations and re-connect them with the world.

Due to the fact that P300 waves differ in amplitude and latency across subjects but meanwhile they are more or less similar to each other, traditional P300 speller adopts supervised learning that requires a calibration process, in which individual user’s data are extracted to train a user-specific machine learning classifier. To make a breakthrough in the trade-off between the accuracy and speed, we try to remove the calibration phase. In this sense, we are dealing with a Zero-shot learning task.

An ideal calibration-less algorithm should take one of two paths: (1) determine a set of perfectly generalized features that do not differ among subjects (generalization) or (2) store various and infinite forms of P300 signals for comparison (robustness).

To acquire generalization and robustness, we are building a similarity measurement in the high dimensional encoded feature space to support our model. After the model is trained with a library of labeled training data, the new unlabeled data is mapped into a pre-trained feature space and compare with the existing ones. Based on the similarity scores, we assign the most appropriate pre-trained classifier to use. 

Meanwhile, we will try other methods to accelerate the process including neural network architecture modification, dynamic stopping criteria for data acquisition, etc.

## Details
P300 speller users use a 9 by 8 spelling grid as 'keyboard'. They gaze at one of the grids to express their spelling intentions. In each round, the rows and columns of the spelling grid flash sequentially. When the gazed grid flashes, a P300 wave generates in the user’s EEG signal which can be characterized as a positive deflection with a latency of roughly 300 ms after the flash onset. 

In pre-processing, we crop the continuous raw EEG signal into epochs, each epoch corresponds to a 800 ms window after each flash onset. Techniques such as frequency filtering, Independent Component Analysis, and EOG/ECG channel simulation are carried out to increase the signal-to-noise ratio.

We use EEGNet(Lawhern et al., 2018) architecture as reference for our CNN model, which aims to detect the pattern of P300 wave in each epoch and make binary classification. Once we can successfully identify whether an epoch contains P300 wave, or equivalently whether a flash elicits ERP, we can infer subject’s spelling intentions with the known information of flashing order.

## Notice
Original dataset is presented as edf files.

Original dataset doesn't include STIM channel, EOG channel, ECG channel, and reference channel. We either infer or simulate.

Original dataset is confidential, I'm not uploading it to repo.

## Software
Python 3.8.16

Pre-processing: MNE

Neural Network: PyTorch

## References
1. Lawhern, Vernon J, Amelia J Solon, Nicholas R Waytowich, Stephen M Gordon, Chou P Hung, and Brent J Lance. “EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain–Computer Interfaces.” Journal of Neural Engineering 15, no. 5 (2018): 056013. https://doi.org/10.1088/1741-2552/aace8c.
2. Farwell, L.A., and E. Donchin. “Talking off the Top of Your Head: Toward a Mental Prosthesis Utilizing Event-Related Brain Potentials.” Electroencephalography and Clinical Neurophysiology 70, no. 6 (1988): 510–23. https://doi.org/10.1016/0013-4694(88)90149-6. 

